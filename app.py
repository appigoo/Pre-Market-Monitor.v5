import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date  # 正確寫法
import os
import json
import re

st.set_page_config(page_title="市場熱力圖", layout="wide")
st.title("每日市場熱力圖（yfinance 版）")
st.markdown("資料來源：Yahoo Finance，自動生成主要資產漲跌熱力表。")

# ======== 資產清單 ========
ASSETS = {
    "TESLA": "TSLA","APPLE": "AAPL","NVIDIA": "NVDA","GOOGLE": "GOOGL","META": "META","MSFT": "MSFT","AMAZON": "AMZN","NIO": "NIO","XPEV": "XPEV",
    "美元指數": "DX-Y.NYB", "2年美債收益率": "^IRX", "10年美債收益率": "^TNX",
    "TLT(長期美債ETF)": "TLT", "S&P500": "^GSPC", "那斯達克": "^IXIC",
    "道瓊工業": "^DJI", "羅素2000": "^RUT", "VIX(恐慌指數)": "^VIX",
    "黃金期貨": "GC=F", "WTI原油": "CL=F", "REITs指數ETF": "VNQ",
    "科技ETF": "XLK", "醫療ETF": "XLV", "金融ETF": "XLF", "能源ETF": "XLE",
    "非必需消費ETF": "XLY", "公用事業ETF": "XLU", "必需消費ETF": "XLP",
    "標普成長ETF": "SPYG", "標普價值ETF": "SPYV", "標普500 ETF": "SPY",
    "那斯達克100(大型科技)": "QQQ"
}

# ======== 側欄設定 ========
st.sidebar.header("設定")
start_of_fetch = st.sidebar.date_input(
    "最早抓取日期", value=date.today() - timedelta(days=400),
    min_value=date(2000, 1, 1), max_value=date.today()
)
save_folder = st.sidebar.text_input("CSV 存放資料夾", value="data")
os.makedirs(save_folder, exist_ok=True)

st.sidebar.header("AI 總結設定")
ai_focus = st.sidebar.text_input("重點關注標的（逗號分隔）", value="TSLA")
ai_style = st.sidebar.selectbox("總結風格", ["決策型（明確多空方向）", "簡報型（精簡重點）", "深度研究型"])
anthropic_key = st.sidebar.text_input("Anthropic API Key（選填，直接生成）", type="password")
anthropic_model = st.sidebar.text_input("模型", value="claude-sonnet-5-5")
self_improve = st.sidebar.checkbox("啟用自我改進回饋迴路", value=True,
    help="每次生成會記錄訊號與判斷，隔日以 SPY 實際漲跌評分，並自動調整各訊號權重")
JOURNAL_FILE = os.path.join(save_folder, "journal.jsonl")
RULES_FILE = os.path.join(save_folder, "learned_rules.txt")


# ======== 自選股票 ========
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    try:
        with open(WATCHLIST_FILE, encoding="utf-8") as f:
            return json.load(f).get("text", "")
    except Exception:
        return ""

def save_watchlist(text):
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump({"text": text}, f, ensure_ascii=False)
    except Exception as e:
        st.sidebar.warning(f"自選清單儲存失敗：{e}")

def parse_custom(text):
    """支援：AMD, PLTR  /  特斯拉:TSLA  /  騰訊:0700.HK  /  BTC-USD  /  GC=F（換行或逗號分隔）"""
    items = {}
    for part in re.split(r"[,\n，;；\s]+", text.strip()):
        if not part:
            continue
        if ":" in part:
            nm, tk = part.split(":", 1)
        else:
            nm, tk = "", part
        tk = tk.strip().upper()
        if not tk:
            continue
        items[(nm.strip() or tk)] = tk
    return items

st.sidebar.header("自選股票")
custom_text = st.sidebar.text_area(
    "輸入代號（逗號或換行分隔，可用「名稱:代號」）",
    value=load_watchlist(), height=100,
    placeholder="AMD, PLTR, 騰訊:0700.HK, BTC-USD"
)
remember_watchlist = st.sidebar.checkbox("記住自選清單", value=True)
hidden_defaults = st.sidebar.multiselect("隱藏預設資產", list(ASSETS.keys()))

ACTIVE_ASSETS = {k: v for k, v in ASSETS.items() if k not in hidden_defaults}
_seen = set(ACTIVE_ASSETS.values())
for _nm, _tk in parse_custom(custom_text).items():
    if _tk in _seen:
        continue
    if _nm in ACTIVE_ASSETS:
        _nm = f"{_nm}({_tk})"
    ACTIVE_ASSETS[_nm] = _tk
    _seen.add(_tk)
st.sidebar.caption(f"目前共 {len(ACTIVE_ASSETS)} 檔（自選 {len(ACTIVE_ASSETS) - len([k for k in ASSETS if k not in hidden_defaults])} 檔）")

# ======== 時區工具 ========
NY_TZ = 'America/New_York'

def to_ny_aware(dt):
    if dt is None:
        return None
    try:
        ts = pd.to_datetime(dt)
        if ts.tz is None:
            return ts.tz_localize(NY_TZ)
        else:
            return ts.tz_convert(NY_TZ)
    except Exception as e:
        st.warning(f"時間轉換錯誤: {e}")
        return None

# ======== 資料抓取 ========
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(tickers, start_date):
    data = {}
    start_ny = to_ny_aware(start_date)
    end_ny = to_ny_aware(date.today() + timedelta(days=1))

    for tk in tickers:
        try:
            hist = yf.Ticker(tk).history(
                start=start_ny.date() if start_ny else None,
                end=end_ny.date() if end_ny else None,
                raise_errors=False
            )
            if not hist.empty:
                if hist.index.tz is None:
                    hist.index = hist.index.tz_localize(NY_TZ)
                else:
                    hist.index = hist.index.tz_convert(NY_TZ)
            data[tk] = hist
        except Exception as e:
            st.warning(f"抓取 {tk} 失敗：{e}")
            data[tk] = pd.DataFrame()
    return data

# ======== 最近價格 ========
def nearest_price(hist_df, target_date):
    if hist_df is None or hist_df.empty:
        return np.nan
    target = to_ny_aware(target_date)
    if target is None:
        return np.nan
    idx = hist_df.index
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ)
    else:
        idx = idx.tz_convert(NY_TZ)
    past = idx[idx <= target]
    if past.empty:
        return np.nan
    return hist_df.loc[past.max(), "Close"]

# ======== 季度起點 ========
def quarter_start(today):
    q = (today.month - 1) // 3 + 1
    return date(today.year, 3*(q-1)+1, 1)

# ======== 計算漲跌 ========
def compute_changes(hist_map, assets):
    rows = []

    for name, tk in assets.items():
        hist = hist_map.get(tk, pd.DataFrame())
        has_data = not hist.empty and "Close" in hist.columns
        closes = hist["Close"].dropna() if has_data else pd.Series(dtype=float)
        latest = closes.iloc[-1] if len(closes) else np.nan

        changes = {}
        if len(closes):
            # 以「最後一根K線日期」為錨點，避免週末/休市/時區造成 0%
            anchor = closes.index[-1].date()
            targets = {
                "1週": anchor - timedelta(days=7),
                "1月": anchor - timedelta(days=30),
                "1年": anchor - timedelta(days=365),
                "QTD": quarter_start(anchor),
                "YTD": date(anchor.year, 1, 1),
            }
            # 1日 = 最新收盤 vs 前一根K線收盤
            prev1 = closes.iloc[-2] if len(closes) >= 2 else np.nan
            changes["1日"] = round((latest / prev1 - 1) * 100, 2) if pd.notna(prev1) and prev1 != 0 else np.nan
            for label, base_date in targets.items():
                prev = nearest_price(hist, base_date)
                changes[label] = round((latest / prev - 1) * 100, 2) if pd.notna(prev) and prev != 0 else np.nan
        else:
            changes = {k: np.nan for k in ["1日", "1週", "1月", "1年", "QTD", "YTD"]}

        rows.append({
            "資產": name, "Ticker": tk, "收盤": round(latest, 4) if pd.notna(latest) else np.nan,
            "1日": changes["1日"], "1週": changes["1週"], "1月": changes["1月"],
            "1年": changes["1年"], "QTD": changes["QTD"], "YTD": changes["YTD"]
        })

    return pd.DataFrame(rows).set_index("資產")

# ======== 自動分析 + AI Prompt ========
def _g(t, tk, col):
    try:
        v = t.loc[tk, col]
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan

def _fmt(v, suffix="%"):
    return "N/A" if pd.isna(v) else f"{v:+.2f}{suffix}"

def _avg(t, tks, col):
    vals = [_g(t, k, col) for k in tks]
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def df_to_markdown(df):
    d = df.reset_index()
    cols = list(d.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ("資產", "Ticker"):
                cells.append(str(v))
            elif pd.isna(v):
                cells.append("N/A")
            elif c == "收盤":
                cells.append(f"{v:.2f}")
            else:
                cells.append(f"{v:+.2f}%")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)

def build_analysis(df):
    """用規則自動算出板塊輪動、風險偏好等重點，附在 prompt 中"""
    t = df.reset_index().set_index("Ticker")
    names = df.reset_index().set_index("Ticker")["資產"]
    L = []

    # 1) 領漲/領跌（排除利率與 VIX）
    px = t.drop(index=[x for x in ("^IRX", "^TNX", "^VIX") if x in t.index])
    for col in ["1日", "1週", "1月", "YTD"]:
        s = px[col].dropna().sort_values(ascending=False)
        if len(s) >= 6:
            L.append(f"- {col} 領漲：" + "、".join(f"{names[k]}({v:+.2f}%)" for k, v in s.head(3).items()))
            L.append(f"- {col} 領跌：" + "、".join(f"{names[k]}({v:+.2f}%)" for k, v in s.tail(3)[::-1].items()))

    # 2) 板塊輪動
    cyc, defn = ["XLK", "XLY", "XLF", "XLE"], ["XLU", "XLP", "XLV"]
    L.append("")
    for col in ["1週", "1月"]:
        c, d = _avg(t, cyc, col), _avg(t, defn, col)
        L.append(f"- {col} 週期板塊均值 {_fmt(c)} vs 防禦板塊均值 {_fmt(d)}，差距 {_fmt(c - d, 'pp') if pd.notna(c) and pd.notna(d) else 'N/A'}")
    sect = ["XLK", "XLV", "XLF", "XLE", "XLY", "XLU", "XLP"]
    sr = t.loc[[k for k in sect if k in t.index], "1月"].dropna().sort_values(ascending=False)
    if not sr.empty:
        L.append("- 板塊 1月強弱排名：" + " > ".join(f"{names[k]}({v:+.2f}%)" for k, v in sr.items()))

    # 3) 風格 / 市值 / 七巨頭
    gv = _g(t, "SPYG", "1月") - _g(t, "SPYV", "1月")
    sl = _g(t, "^RUT", "1月") - _g(t, "^GSPC", "1月")
    mag = _avg(t, ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"], "1月")
    L.append(f"- 成長 vs 價值（SPYG-SPYV，1月）：{_fmt(gv, 'pp')}")
    L.append(f"- 小型 vs 大型（羅素2000-S&P500，1月）：{_fmt(sl, 'pp')}")
    L.append(f"- 七巨頭 1月平均：{_fmt(mag)}")

    # 4) 宏觀
    L.append(f"- VIX 現值 {_g(t, '^VIX', '收盤'):.2f}，1日 {_fmt(_g(t, '^VIX', '1日'))}，1週 {_fmt(_g(t, '^VIX', '1週'))}")
    L.append(f"- 10年美債殖利率 {_g(t, '^TNX', '收盤'):.3f}%，1週 {_fmt(_g(t, '^TNX', '1週'))}（為殖利率本身的相對變化，非 bps）")
    L.append(f"- 美元指數 {_g(t, 'DX-Y.NYB', '收盤'):.2f}，1週 {_fmt(_g(t, 'DX-Y.NYB', '1週'))}｜TLT 1週 {_fmt(_g(t, 'TLT', '1週'))}")
    L.append(f"- 黃金 1週 {_fmt(_g(t, 'GC=F', '1週'))}｜WTI 原油 1週 {_fmt(_g(t, 'CL=F', '1週'))}")

    # 5) 風險偏好評分（加權，權重由歷史命中率自我校準）
    sigs = compute_signals(df)
    wmap = learned_weights(load_journal()) if self_improve else {}
    norm, label = score_from_signals(sigs, wmap)
    tag = "（已套用歷史學習權重）" if any(v["n"] >= 5 for v in wmap.values()) else "（權重尚在累積樣本，暫用等權）"
    L.append("")
    L.append(f"**規則引擎風險偏好評分：{norm:+.1f} / 5 → {label}** {tag}")
    L.append("- 訊號明細：" + "、".join(f"{k}={_fmt(v, 'pp')}" for k, v in sigs.items()))
    return "\n".join(L)

# ======== 自我改進：預測日誌 → 評分 → 權重 → 規則 ========
SIGNAL_NAMES = ["週期>防禦", "成長>價值", "小型>大型", "VIX下降", "美元走弱"]

def compute_signals(df):
    t = df.reset_index().set_index("Ticker")
    cyc, defn = ["XLK", "XLY", "XLF", "XLE"], ["XLU", "XLP", "XLV"]
    return {
        "週期>防禦": _avg(t, cyc, "1週") - _avg(t, defn, "1週"),
        "成長>價值": _g(t, "SPYG", "1週") - _g(t, "SPYV", "1週"),
        "小型>大型": _g(t, "^RUT", "1週") - _g(t, "^GSPC", "1週"),
        "VIX下降": -_g(t, "^VIX", "1週"),
        "美元走弱": -_g(t, "DX-Y.NYB", "1週"),
    }

def score_from_signals(sigs, wmap):
    num = den = 0.0
    for k, v in sigs.items():
        if v is None or pd.isna(v):
            continue
        w = wmap.get(k, {}).get("w", 1.0)
        num += w if v > 0 else -w
        den += w
    norm = num / den * 5 if den else 0.0
    label = "偏 Risk-On（偏多）" if norm >= 2 else "偏 Risk-Off（偏空/防守）" if norm <= -2 else "中性震盪"
    return norm, label

def load_journal():
    try:
        with open(JOURNAL_FILE, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []

def save_journal(entries):
    try:
        with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception as e:
        st.warning(f"日誌儲存失敗：{e}")

def learned_weights(entries):
    """每個訊號的方向命中率 → 權重（貝氏收縮到 50%，樣本<5 則維持 1.0，範圍 0.2~2.0）"""
    ev = [e for e in entries if e.get("fwd1") is not None]
    out = {}
    for name in SIGNAL_NAMES:
        hits = n = 0
        for e in ev:
            v = e.get("signals", {}).get(name)
            if v is None or v == 0:
                continue
            n += 1
            hits += int((v > 0) == (e["fwd1"] > 0))
        p = (hits + 5) / (n + 10)
        w = 1.0 if n < 5 else float(np.clip(1 + (p - 0.5) * 4, 0.2, 2.0))
        out[name] = {"n": n, "hit": (hits / n if n else None), "w": w}
    return out

def overall_accuracy(entries):
    hits = n = 0
    for e in entries:
        if e.get("fwd1") is None or abs(e.get("norm", 0)) < 2:
            continue
        n += 1
        hits += int((e["norm"] > 0) == (e["fwd1"] > 0))
    return hits, n

def record_and_evaluate(df, hist_map):
    spy = hist_map.get("SPY")
    if spy is None or spy.empty or "Close" not in spy.columns:
        return
    closes = spy["Close"].dropna()
    if len(closes) < 2:
        return
    entries = load_journal()
    dates = [d.date().isoformat() for d in closes.index]
    for e in entries:                       # 1) 評分舊預測：下一個交易日的 SPY 實際報酬
        if e.get("fwd1") is None:
            for k, d in enumerate(dates):
                if d > e["anchor"]:
                    e["fwd1"] = float(closes.iloc[k] / e["spy_close"] - 1)
                    break
    anchor = dates[-1]                       # 2) 記錄今天的判斷（同日未評分者以最新資料覆蓋）
    entries = [e for e in entries if not (e["anchor"] == anchor and e.get("fwd1") is None)]
    if not any(e["anchor"] == anchor for e in entries):
        sigs = compute_signals(df)
        norm, label = score_from_signals(sigs, learned_weights(entries) if self_improve else {})
        entries.append({
            "anchor": anchor, "spy_close": float(closes.iloc[-1]),
            "signals": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in sigs.items()},
            "norm": round(norm, 2), "label": label, "fwd1": None,
        })
    save_journal(entries)

def load_rules():
    try:
        with open(RULES_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def build_feedback_section():
    if not self_improve:
        return "（自我改進回饋迴路已停用）"
    entries = load_journal()
    ev = [e for e in entries if e.get("fwd1") is not None]
    wmap = learned_weights(entries)
    hits, n = overall_accuracy(entries)
    L = [f"- 已評分預測 {len(ev)} 筆（待評分 {len(entries) - len(ev)} 筆）；"
         + (f"方向性判斷命中率 {hits}/{n} = {hits / n:.0%}" if n else "方向性判斷尚無足夠樣本")]
    L += ["", "| 訊號 | 樣本數 | 命中率 | 目前權重 |", "|---|---|---|---|"]
    for k, v in wmap.items():
        hit = "N/A" if v["hit"] is None else f"{v['hit']:.0%}"
        L.append(f"| {k} | {v['n']} | {hit} | {v['w']:.2f} |")
    recent = ev[-5:]
    if recent:
        L += ["", "近期預測 vs 實際（SPY 隔日報酬）：", "| 日期 | 判斷 | 評分 | 隔日SPY |", "|---|---|---|---|"]
        for e in recent:
            L.append(f"| {e['anchor']} | {e['label']} | {e['norm']:+.1f} | {e['fwd1'] * 100:+.2f}% |")
    rules = load_rules()
    if rules:
        L += ["", "已採用的學習規則（請遵守）：", rules]
    return "\n".join(L)

def build_ai_prompt(df, focus, style, as_of):
    style_rule = {
        "決策型（明確多空方向）": "給出明確方向（偏多/偏空/觀望）與具體價位或百分比，不要含糊其辭。",
        "簡報型（精簡重點）": "總長度控制在 400 字內，只保留最關鍵的結論。",
        "深度研究型": "逐項深入解釋因果關係與跨資產傳導，並列出情境分析（基準/樂觀/悲觀）。",
    }[style]
    return f"""你是一位資深美股宏觀與日內交易策略分析師。請根據以下 {as_of} 的市場熱力表與自動分析結果，用繁體中文輸出完整的每日市場總結。

## 一、原始數據（收盤價與漲跌幅，美東時間；殖利率行的收盤值為 %）
{df_to_markdown(df)}

## 二、系統自動分析
{build_analysis(df)}

## 二之二、歷史績效回饋（系統自我校準）
{build_feedback_section()}

## 三、請依序輸出（使用表格與具體數字）
1. **一句話結論**：當前市場屬於 Risk-On / Risk-Off / 中性，並說明最關鍵的 2 個證據。
2. **指數與宏觀總覽表**：S&P500、那斯達克、道瓊、羅素2000、VIX、美元、10年債、黃金、原油的 1日/1週/1月/YTD 與一句判讀。
3. **板塊輪動分析**：資金流向哪些板塊、離開哪些板塊，週期 vs 防禦、成長 vs 價值是否改變。
4. **利率 / 美元 / 商品傳導**：對科技股與小型股的影響。
5. **重點個股（{focus}）**：結合上述環境，給出短線（1-3 日）與中線（1-4 週）方向，並提供支撐/壓力或風險點。
6. **風險提示**：列出 3 個最可能推翻你判斷的因素（含觸發條件）。
7. **明日交易計畫**：偏多/偏空/觀望、倉位建議（%）、止損原則。

## 四、輸出規則
- {style_rule}
- 所有判斷必須引用上方數據中的具體數字，不可憑空編造未提供的數據。
- 資料不足（N/A）時明確指出，不要猜測。
- 參考「歷史績效回饋」校準信心：命中率低於 50% 或權重偏低的訊號要降低依賴，並在結論中說明本次判斷的可信度。
- 結尾註明：以上僅為資料分析，非投資建議。"""

def call_claude(prompt, api_key, model):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model=model, max_tokens=3000, messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

# ======== 顏色 ========
def colorize(val):
    if pd.isna(val):
        return "background-color: #f0f0f0; color: #999;"
    v = float(val)
    if v > 0:
        intensity = min(abs(v)/15, 1)
        g = int(180 + 75*intensity)
        return f"background-color: rgba(0,{g},0,{0.2+0.3*intensity}); color: green; font-weight: bold;"
    elif v < 0:
        intensity = min(abs(v)/15, 1)
        r = int(255 - 55*intensity)
        return f"background-color: rgba({r},100,100,{0.2+0.3*intensity}); color: red; font-weight: bold;"
    else:
        return "background-color: #fff; color: #333;"

# ======== 主程式 ========
if st.button("生成市場熱力圖", type="primary"):
    with st.spinner("抓取資料中..."):
        hist_map = fetch_history(list(ACTIVE_ASSETS.values()), start_of_fetch)
        df_new = compute_changes(hist_map, ACTIVE_ASSETS)
    if self_improve:
        record_and_evaluate(df_new, hist_map)
    if remember_watchlist:
        save_watchlist(custom_text)
    missing = df_new[df_new["收盤"].isna()]["Ticker"].tolist()
    csv_file = f"market_heatmap_{date.today().isoformat()}.csv"
    csv_path = os.path.join(save_folder, csv_file)
    df_new.to_csv(csv_path, encoding="utf-8-sig", float_format="%.4f")
    st.session_state.update(df=df_new, csv_file=csv_file, csv_path=csv_path, ai_report=None, missing=missing)

if "df" in st.session_state:
    df = st.session_state["df"]
    style_cols = ["1日", "1週", "1月", "1年", "QTD", "YTD"]
    styled = df.style\
        .map(colorize, subset=style_cols)\
        .format({"收盤": "{:.4f}", **{col: "{:.2f}%" for col in style_cols}}, na_rep="N/A")\
        .set_properties(**{'text-align': 'center'}, subset=style_cols)

    if st.session_state.get("missing"):
        st.warning("以下代號找不到資料，請確認拼寫（港股 0700.HK、加密 BTC-USD、期貨 GC=F）：" + "、".join(st.session_state["missing"]))
    st.subheader(f"市場表現 {date.today()}")
    st.dataframe(styled, use_container_width=True)
    st.success(f"已儲存：`{st.session_state['csv_path']}`")
    with open(st.session_state["csv_path"], "rb") as f:
        st.download_button("下載 CSV", f, st.session_state["csv_file"], "text/csv")

    # ---- AI 總結 ----
    st.subheader("AI 自動總結 Prompt")
    prompt = build_ai_prompt(df, ai_focus, ai_style, str(date.today()))
    with st.expander("規則引擎自動分析預覽", expanded=True):
        st.markdown(build_analysis(df))
    st.caption("右上角複製按鈕可直接貼到 Claude / ChatGPT；或下載 txt。")
    st.code(prompt, language="markdown")
    st.download_button("下載 Prompt (.txt)", prompt, f"market_ai_prompt_{date.today().isoformat()}.txt", "text/plain")

    if anthropic_key:
        if st.button("直接用 Claude 生成總結"):
            with st.spinner("Claude 分析中..."):
                try:
                    st.session_state["ai_report"] = call_claude(prompt, anthropic_key, anthropic_model)
                except Exception as e:
                    st.error(f"呼叫失敗：{e}（需先 pip install anthropic）")
        if st.session_state.get("ai_report"):
            st.subheader("Claude 市場總結")
            st.markdown(st.session_state["ai_report"])
    # ---- 自我改進 ----
    if self_improve:
        st.subheader("自我改進回饋迴路")
        entries = load_journal()
        hits, n = overall_accuracy(entries)
        c1, c2, c3 = st.columns(3)
        c1.metric("日誌筆數", len(entries))
        c2.metric("已評分", len([e for e in entries if e.get("fwd1") is not None]))
        c3.metric("方向命中率", f"{hits / n:.0%}" if n else "N/A")
        with st.expander("訊號權重與近期績效", expanded=False):
            st.markdown(build_feedback_section())
        if entries:
            jdf = pd.DataFrame([{
                "日期": e["anchor"], "判斷": e["label"], "評分": e["norm"],
                "隔日SPY%": None if e.get("fwd1") is None else round(e["fwd1"] * 100, 2)} for e in entries[::-1]])
            st.dataframe(jdf, use_container_width=True, hide_index=True)

        st.markdown("**元層改進：讓 Claude 檢討判斷錯誤並提出新規則（需你審核後才會生效）**")
        if not anthropic_key:
            st.caption("在側欄填入 Anthropic API Key 後可使用。")
        elif len([e for e in entries if e.get("fwd1") is not None]) < 5:
            st.info("至少累積 5 筆已評分預測後才開放（每個交易日收盤後執行一次）。")
        else:
            if st.button("讓 Claude 檢討並提出新規則"):
                meta = (
                    "你是量化策略的檢討教練。以下是一個『每日市場風險偏好判斷系統』的歷史績效與目前規則。\n"
                    "請找出系統性偏誤（例如哪些訊號在什麼情境下失準、是否在高 VIX 時過度自信），"
                    "提出最多 5 條可直接加入 prompt 的『具體可執行規則』，每條一行、繁體中文、以數字條件表述。\n"
                    "限制：只能依據下列資料推論，不得編造；樣本少時要明說不確定；不要更動輸出格式。\n\n"
                    + build_feedback_section()
                )
                with st.spinner("Claude 檢討中..."):
                    try:
                        st.session_state["proposed_rules"] = call_claude(meta, anthropic_key, anthropic_model)
                    except Exception as e:
                        st.error(f"呼叫失敗：{e}")
            if st.session_state.get("proposed_rules"):
                edited = st.text_area("建議規則（可編輯後採用）", st.session_state["proposed_rules"], height=200)
                cc1, cc2 = st.columns(2)
                if cc1.button("採用並儲存規則"):
                    with open(RULES_FILE, "w", encoding="utf-8") as f:
                        f.write(edited.strip())
                    st.session_state["proposed_rules"] = None
                    st.success("已儲存，下次生成的 Prompt 會包含這些規則。")
                if cc2.button("清空已學規則"):
                    if os.path.exists(RULES_FILE):
                        os.remove(RULES_FILE)
                    st.success("已清空。")
else:
    st.info("點擊按鈕生成熱力圖")

st.markdown("---\n**綠漲紅跌**｜資料：Yahoo Finance｜時間：美東時間")
