import os
import sys
import time
import warnings
from pathlib import Path

# ── Add project root to path so utils/ is importable ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Suppress warnings BEFORE any heavy imports ───────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"]       = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_hf_token = os.getenv("HF_TOKEN", "")
if _hf_token:
    os.environ["HF_TOKEN"]               = _hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

# ── Streamlit import + page config MUST come before any st.* call ────────────
import streamlit as st

st.set_page_config(
    page_title="AI Product Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Remaining imports ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from datetime import datetime

from utils.matcher import (
    load_data,
    load_model,
    build_index,
    clean_text,
    find_match,
    compare_two_products,
    extract_title_from_url,
    run_batch,
    save_result_with_aht,
    generate_batch_excel,
    get_domain,
    resolve_target_domain,
    MATCH_THRESHOLD,
    RESULTS_FILE,
    RETAILER_DOMAIN_MAP,
    DATA_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# cache_data  → dataframe  (serialisable, can be hashed by Streamlit)
# cache_resource → model & index  (heavy objects, must NOT be re-created)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    """Load and enrich the product Excel — runs once, result is cached."""
    data_path = DATA_DIR / "Matching List - Example 1 (1).xlsx"
    if not data_path.exists():
        xlsx_files = list(DATA_DIR.glob("*.xlsx")) if DATA_DIR.exists() else []
        if xlsx_files:
            data_path = xlsx_files[0]
            st.warning(
                f"⚠️ Default data file not found. Using: **{data_path.name}**"
            )
        else:
            st.error(
                f"❌ **Data file not found.**\n\n"
                f"Expected: `{data_path}`\n\n"
                f"Place your Excel file in the `data/` folder and restart the app."
            )
            st.stop()
    return load_data(data_path)


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the SentenceTransformer model — downloads once, then uses .model_cache/."""
    return load_model()


@st.cache_resource(show_spinner=False)
def get_index(_model, _n_rows: int):
    """Build the FAISS index — tied to model + row count so it rebuilds if data changes."""
    _df = get_data()
    return build_index(_df, _model, show_progress=False)


# ─────────────────────────────────────────────────────────────────────────────
# ONE-TIME STARTUP — show a spinner only on the very first page load
# After this, every interaction is sub-second because everything is cached.
# ─────────────────────────────────────────────────────────────────────────────

if "app_ready" not in st.session_state:
    with st.spinner("⚙️ Loading AI model and building product index — takes ~10s on first run, then instant…"):
        _df_init    = get_data()
        _model_init = get_model()
        _           = get_index(_model_init, len(_df_init))
    st.session_state.app_ready = True

# All subsequent calls hit the cache immediately
df    = get_data()
model = get_model()
index = get_index(model, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "aht_log"         not in st.session_state: st.session_state.aht_log         = []
if "pending_results" not in st.session_state: st.session_state.pending_results = []


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def show_results_table(df_display: pd.DataFrame, link_cols: list = None):
    """Display a DataFrame with URL columns rendered as clickable links."""
    if link_cols is None:
        link_cols = [c for c in df_display.columns
                     if "link" in c.lower() or "url" in c.lower()]

    col_config = {
        col: st.column_config.LinkColumn(col, display_text="🔗 Open", help="Click to open")
        for col in link_cols
        if col in df_display.columns
    }
    display = df_display.copy()
    for col in link_cols:
        if col in display.columns:
            display[col] = display[col].astype(str).replace("nan", "")

    st.dataframe(display, column_config=col_config,
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔍 AI Product Matching System")
st.markdown(
    f"**Find identical products across Saudi e-commerce — Arabic · English · Turkish**"
    f"&nbsp;&nbsp;&nbsp;`{len(df)} products indexed`"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_aht = st.tabs([
    "🔍 Single Product Search",
    "📂 Batch Upload (Excel / CSV)",
    "📈 AHT Dashboard",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE SEARCH
# ════════════════════════════════════════════════════════════════════════════

with tab_single:
    st.subheader("📥 Enter URL and/or Product Title")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔗 Source Product URL** *(optional)*")
        source_url = st.text_input(
            "source_url_input", label_visibility="collapsed",
            placeholder="https://hepsiburada.com/... or https://noon.com/..."
        )
        st.caption("Paste a product URL — title is auto-extracted in any language")

        st.markdown("**📦 Product Title** *(optional if URL given)*")
        product_title = st.text_input(
            "product_title_input", label_visibility="collapsed",
            placeholder="e.g. Xiaomi 14T 256GB / شاومي 14 تي / Samsung mavi"
        )
        st.caption("Full title, partial info, or any language — all supported")

    with col2:
        st.markdown("**🌐 Target Website or Target Product URL** *(optional)*")
        target_site = st.text_input(
            "target_site_input", label_visibility="collapsed",
            placeholder="www.noon.com  OR  https://extra.com/en-sa/product/..."
        )
        st.caption("Domain → filters to that site only | Full URL → direct comparison")

        st.markdown("**🎯 Match Threshold**")
        threshold = st.slider(
            "threshold_slider", label_visibility="collapsed",
            min_value=0.5, max_value=1.0, value=MATCH_THRESHOLD, step=0.01,
            help="Lower = more lenient. Use ~0.65 for partial queries or translated titles.",
        )
        st.caption("💡 ~0.65 works better for partial or translated input")

    st.caption("ℹ️ **(A)** URL only → title auto-extracted  |  **(B)** Title only  |  **(C)** Both → typed title takes priority")

    with st.expander("ℹ️ What is AHT and how is it calculated?"):
        st.markdown("""
**Average Handling Time (AHT)** = time from clicking "Find Match" to result displayed.

| Metric | Description |
|---|---|
| **AHT per search** | Wall-clock time for one search |
| **Session Avg AHT** | Running average across all searches this session |
| **AHT Reduction** | Manual team baseline (3–10 min) vs AI (<1 second) |
        """)

    if st.button("🔍 Find Matching Product", type="primary",
                 use_container_width=True, key="single_search_btn"):

        final_title            = product_title.strip()
        target_input           = target_site.strip()
        target_is_url          = target_input.startswith("http")
        target_extracted_title = None

        # ── Optional: extract title from source URL ───────────────────────────
        if source_url.strip() and not final_title:
            with st.spinner("🌐 Fetching title from source URL (may take a few seconds for geo-blocked sites)…"):
                extracted_info = extract_title_from_url(source_url.strip())
            if extracted_info["success"]:
                final_title = extracted_info["title"]
                st.success(f"✅ **Source title extracted** ({extracted_info['source']}): `{final_title}`")
            else:
                st.warning(
                    f"⚠️ Could not auto-extract: *{extracted_info['error']}*  \n"
                    "Please type the product title manually."
                )

        # ── Optional: extract title from target URL ───────────────────────────
        if target_is_url:
            with st.spinner("🌐 Fetching title from target URL…"):
                target_extracted = extract_title_from_url(target_input)
            if target_extracted["success"]:
                target_extracted_title = target_extracted["title"]
                st.success(f"✅ **Target title extracted** ({target_extracted['source']}): `{target_extracted_title}`")
            else:
                st.warning(
                    f"⚠️ Could not extract from target URL: *{target_extracted['error']}*  \n"
                    "Falling back to domain-only filter."
                )

        if not final_title:
            st.warning("⚠️ Please enter a Product Title or a valid Source URL.")

        # ── CASE A: Direct URL-to-URL comparison ─────────────────────────────
        elif target_extracted_title:
            with st.spinner("🔎 Comparing products…"):
                comparison = compare_two_products(final_title, target_extracted_title, model)

            st.session_state.aht_log.append(comparison["elapsed"])

            st.divider()
            st.subheader("🎯 Direct URL-to-URL Comparison")
            c1, c2, c3 = st.columns(3)
            c1.metric("Similarity Score", f"{comparison['score']:.4f}")
            c2.metric("Search Time",      f"{comparison['elapsed']:.3f}s")
            c3.metric("Result", "✅ Match" if comparison["is_match"] else "❌ No Match")

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**📦 Source Product**")
                st.info(final_title)
                if source_url.strip():
                    st.markdown(f"[Open Source Link]({source_url.strip()})")
            with cc2:
                st.markdown("**🎯 Target Product**")
                st.info(target_extracted_title)
                st.markdown(f"[Open Target Link]({target_input})")

            if comparison["is_match"]:
                st.success(f"✅ **Same product!** Confidence: {comparison['score']:.1%}")
            else:
                st.error(f"❌ **Different products.** Score {comparison['score']:.4f} below threshold {threshold}.")

            with st.spinner("Running broader search on target site…"):
                results = find_match(final_title, model, index, df, threshold,
                                     target_filter=target_input)
            disp = results[["crawling title", "Link", "Similarity Score", "Match Status"]].copy()
            disp.columns = ["Product Title", "Link", "Similarity Score", "Match Status"]
            st.subheader("📊 Other Top Matches from Target Site")
            show_results_table(disp, link_cols=["Link"])

            st.session_state.pending_results.append({
                "source_title": final_title,
                "source_url":   source_url.strip() or "N/A",
                "target_site":  target_input,
                "match_title":  target_extracted_title,
                "target_link":  target_input,
                "score":        comparison["score"],
                "aht_seconds":  comparison["elapsed"],
            })

        # ── CASE B: Title search ──────────────────────────────────────────────
        else:
            with st.spinner("🔎 Searching…"):
                t0      = time.time()
                results = find_match(
                    final_title, model, index, df,
                    threshold=threshold,
                    target_filter=target_input if target_input else None,
                )
                elapsed = time.time() - t0

            st.session_state.aht_log.append(elapsed)

            if target_input and results.attrs.get("domain_filtered"):
                st.info(f"🔍 Results filtered to **{results.attrs['domain_used']}** only")
            elif target_input and not results.attrs.get("domain_filtered"):
                st.warning(f"⚠️ No products from **{get_domain(target_input)}** in dataset — showing all sites.")

            best  = results.iloc[0]
            score = float(best["Similarity Score"])

            if len(set(clean_text(final_title).split())) <= 4:
                st.info("🔎 **Partial query detected** — keyword boost applied.")

            st.subheader("📊 Top 5 Matches")
            disp = results[["crawling title", "Link", "Similarity Score", "Match Status"]].copy()
            disp.columns = ["Product Title", "Link", "Similarity Score", "Match Status"]
            show_results_table(disp, link_cols=["Link"])

            st.divider()
            ca, cb, cc = st.columns(3)
            ca.metric("🎯 Best Score",    f"{score:.4f}")
            cb.metric("⏱️ Search Time",    f"{elapsed:.3f}s")
            cc.metric("📈 Session Avg",
                      f"{sum(st.session_state.aht_log)/len(st.session_state.aht_log):.3f}s")

            if score >= threshold:
                st.success(f"✅ **Match Found!** Confidence: {score:.1%}")
                st.markdown(f"### 🔗 Best Match\n[Open Product]({best['Link']})")
            else:
                st.error(f"❌ **No Match Found.** Best score ({score:.4f}) is below threshold ({threshold}).")
                st.markdown(f"*Closest: [{best['crawling title']}]({best['Link']})*")

            st.session_state.pending_results.append({
                "source_title": final_title,
                "source_url":   source_url.strip() or "N/A",
                "target_site":  target_input or "N/A",
                "match_title":  best["crawling title"],
                "target_link":  best["Link"],
                "score":        score,
                "aht_seconds":  elapsed,
            })

    # Save-to-disk is on-demand, not automatic on every search
    if st.session_state.pending_results:
        n = len(st.session_state.pending_results)
        if st.button(f"💾 Save {n} unsaved result(s) to Excel log", key="save_btn"):
            for r in st.session_state.pending_results:
                save_result_with_aht(**r)
            st.session_state.pending_results = []
            st.success(f"✅ Saved to `{RESULTS_FILE}`")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("📂 Batch Product Matching — Upload Excel or CSV")

    with st.expander("📋 Download input template first", expanded=False):
        st.markdown("Your file needs **at least one** of these columns:")
        st.markdown("""
| Column | Required? | Description |
|---|---|---|
| `product_title` | ✅ OR source_url | Product name in any language |
| `source_url` | ✅ OR product_title | URL — title auto-extracted if no title column |
| `target_site` | Optional | Domain or retailer code to filter against |
        """)
        template_df = pd.DataFrame({
            "product_title": ["Xiaomi 14T 256GB", "Samsung Galaxy S24", "شاومي ريدمي 13C"],
            "source_url":    ["https://hepsiburada.com/...", "", "https://noon.com/..."],
            "target_site":   ["www.trendyol.com", "www.noon.com", ""],
        })
        st.download_button(
            "⬇️ Download CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_template.csv",
            mime="text/csv",
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your Excel (.xlsx) or CSV (.csv) file",
        type=["xlsx", "csv"],
    )

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        batch_threshold = st.slider(
            "🎯 Match Threshold for Batch",
            min_value=0.5, max_value=1.0, value=MATCH_THRESHOLD, step=0.01,
            key="batch_threshold",
        )
    with b_col2:
        batch_target_filter = st.text_input(
            "🌐 Global Target Website (optional — applies to rows without their own target)",
            placeholder="www.noon.com  OR  retailer code: NN_KSA",
            key="batch_target",
        )

    input_df = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, header=0)
            else:
                raw_df = pd.read_excel(uploaded_file, header=0)

            # Normalise column names
            seen, new_cols = {}, []
            for col in raw_df.columns:
                cc = str(col).lower().strip().replace(" ", "_")
                if cc in seen:
                    seen[cc] += 1
                    new_cols.append(f"{cc}_{seen[cc]}")
                else:
                    seen[cc] = 0
                    new_cols.append(cc)
            raw_df.columns = new_cols

            st.success(f"✅ File loaded — **{len(raw_df)} rows**, **{len(raw_df.columns)} columns**")

            with st.expander("👁️ Preview (first 5 rows)", expanded=True):
                show_results_table(raw_df.head(5).astype(str))
                st.caption(f"Columns: `{', '.join(raw_df.columns)}`")

            st.markdown("### 🗂️ Map Your Columns")
            all_cols = ["— None —"] + list(raw_df.columns)

            def best_guess(keywords):
                for kw in keywords:
                    for col in raw_df.columns:
                        if kw in col:
                            return col
                return "— None —"

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                title_col = st.selectbox(
                    "📦 Product Title", all_cols,
                    index=all_cols.index(best_guess(["crawling_title", "title", "product", "name"])),
                    key="map_title",
                )
            with mc2:
                url_col = st.selectbox(
                    "🔗 Source URL", all_cols,
                    index=all_cols.index(best_guess(["source_url", "url", "link", "source"])),
                    key="map_url",
                )
            with mc3:
                target_col = st.selectbox(
                    "🌐 Target Retailer Code", all_cols,
                    index=all_cols.index(best_guess(["target_retailer_code", "target_retailer", "target_site", "target"])),
                    key="map_target",
                )
            with mc4:
                domain_col = st.selectbox(
                    "🔗 Target Domain (url.1)", all_cols,
                    index=all_cols.index(best_guess(["url_1", "url.1", "target_domain", "domain"])),
                    key="map_domain",
                )

            with st.expander("📋 Supported retailer codes"):
                st.dataframe(
                    pd.DataFrame([(k, v) for k, v in RETAILER_DOMAIN_MAP.items()],
                                 columns=["Retailer Code", "Domain"]),
                    hide_index=True, use_container_width=True,
                )

            input_df = pd.DataFrame()
            if title_col  != "— None —": input_df["product_title"] = raw_df[title_col].astype(str).str.strip()
            if url_col    != "— None —": input_df["source_url"]    = raw_df[url_col].astype(str).str.strip()
            if target_col != "— None —": input_df["target_site"]   = raw_df[target_col].astype(str).str.strip()
            if domain_col != "— None —": input_df["url_1"]         = raw_df[domain_col].astype(str).str.strip()

            if input_df.empty or (
                "product_title" not in input_df.columns and
                "source_url"    not in input_df.columns
            ):
                st.warning("⚠️ Map at least **Product Title** or **Source URL**.")
                input_df = None
            else:
                st.success(f"✅ Ready to process **{len(input_df)} products**.")

        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

    if input_df is not None and st.button(
        "🚀 Run Batch Matching", type="primary",
        use_container_width=True, key="batch_run_btn"
    ):
        st.divider()
        progress_bar = st.progress(0)
        status_text  = st.empty()
        batch_start  = time.time()

        def st_progress(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(f"Processing {current}/{total} — {message}")

        batch_results = run_batch(
            input_df, model, index, df,
            batch_threshold=batch_threshold,
            global_target=batch_target_filter.strip(),
            progress_callback=st_progress,
        )

        total_time = time.time() - batch_start
        progress_bar.progress(1.0)
        status_text.text("✅ Done!")

        st.divider()
        st.subheader("📊 Batch Results")

        total_r   = len(batch_results)
        matched_r = len(batch_results[batch_results["Match Status"] == "Match Found"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Products", total_r)
        m2.metric("✅ Matched",      matched_r)
        m3.metric("❌ Not Matched",  total_r - matched_r)
        m4.metric("Match Rate",     f"{matched_r/total_r*100:.1f}%" if total_r else "0%")

        a1, a2 = st.columns(2)
        a1.metric("⏱️ Total Time",        f"{total_time:.1f}s")
        a2.metric("⚡ Avg AHT / product", f"{batch_results['AHT (seconds)'].mean():.3f}s")

        show_results_table(batch_results, link_cols=["Source URL", "Target Link"])

        excel_bytes = generate_batch_excel(batch_results)
        st.download_button(
            label="⬇️ Download Results as Excel",
            data=excel_bytes,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        for t in batch_results["AHT (seconds)"].tolist():
            st.session_state.aht_log.append(t)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — AHT DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

with tab_aht:
    st.subheader("📈 Session AHT Dashboard")

    if not st.session_state.aht_log:
        st.info("No searches yet — run a single search or batch upload to see AHT stats here.")
    else:
        aht_data = st.session_state.aht_log
        aht_df   = pd.DataFrame({
            "Search #":      range(1, len(aht_data) + 1),
            "AHT (seconds)": [round(x, 4) for x in aht_data],
        })

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Searches", len(aht_data))
        c2.metric("Avg AHT",        f"{np.mean(aht_data):.3f}s")
        c3.metric("Min AHT",        f"{np.min(aht_data):.3f}s")
        c4.metric("Max AHT",        f"{np.max(aht_data):.3f}s")

        st.line_chart(aht_df.set_index("Search #"))

        st.divider()
        st.markdown("### 🧮 AHT Reduction Calculator")
        manual_aht_min = st.number_input(
            "Enter your team's manual AHT per product (minutes):",
            min_value=1, max_value=60, value=5, key="manual_aht",
        )
        manual_sec  = manual_aht_min * 60
        avg_ai      = np.mean(aht_data)
        reduction   = (manual_sec - avg_ai) / manual_sec * 100
        daily_saves = (manual_sec - avg_ai) * 100 / 60

        st.success(
            f"🚀 **AHT Reduction: {reduction:.1f}%** — "
            f"AI: {avg_ai:.3f}s  vs  Manual: {manual_sec}s per product"
        )
        st.caption(
            f"Processing 100 products/day → saves **{daily_saves:.0f} minutes** daily."
        )

        if st.button("🗑️ Clear session history", key="clear_aht"):
            st.session_state.aht_log = []
            st.rerun()
