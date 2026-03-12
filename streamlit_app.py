"""
app/streamlit_app.py
====================
UI layer only — all AI logic lives in utils/matcher.py

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# ── Add project root to sys.path so utils/ is importable ────────────────────
# Works whether streamlit_app.py is at the repo root (Streamlit Cloud)
# or inside an app/ subfolder (local dev layout).
_this_file = os.path.abspath(__file__)
_this_dir  = os.path.dirname(_this_file)           # folder containing streamlit_app.py
_parent    = os.path.dirname(_this_dir)            # one level up

for _p in [_this_dir, _parent]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Suppress warnings before any heavy imports ───────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── HF Token: local .env OR Streamlit Cloud secrets ─────────────────────────
_hf_token = os.getenv("HF_TOKEN", "")
if not _hf_token:
    # Streamlit Cloud: set token via App Settings → Secrets as:  HF_TOKEN = "hf_..."
    try:
        import streamlit as _st_tmp
        _hf_token = _st_tmp.secrets.get("HF_TOKEN", "")
    except Exception:
        pass

if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = _hf_token

# ── Import all logic from utils ───────────────────────────────────────────────
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
)

import streamlit as st
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Product Matcher",
    page_icon="🔍",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def show_results_table(df: pd.DataFrame, link_cols: list = None):
    """Display a DataFrame where URL columns are rendered as clickable links."""
    if link_cols is None:
        link_cols = [
            col for col in df.columns
            if df[col].astype(str).str.startswith("http").any()
        ]

    col_config = {}
    for col in link_cols:
        col_config[col] = st.column_config.LinkColumn(
            col,
            display_text="🔗 Open Link",
            help="Click to open product page"
        )

    display = df.copy()
    for col in link_cols:
        display[col] = display[col].astype(str).replace("nan", "")

    st.dataframe(display, column_config=col_config, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA, MODEL & INDEX  (cached — only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_index(_df, _model):
    # Underscore prefix tells Streamlit not to hash these args
    return build_index(_df, _model)


df    = get_data()
model = get_model()
index = get_index(df, model)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "aht_log" not in st.session_state:
    st.session_state.aht_log = []


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔍 AI Product Matching System")
st.markdown("**Find identical products across Saudi e-commerce websites — supports Arabic, English, Turkish & more**")
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_aht = st.tabs([
    "🔍 Single Product Search",
    "📂 Batch Upload (Excel / CSV)",
    "📈 AHT Dashboard"
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
        st.caption("Paste a product page URL — title will be auto-extracted in any language")

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
        st.caption("Domain → filters results to that site only | Full URL → direct comparison")

        st.markdown("**🎯 Match Threshold**")
        threshold = st.slider(
            "threshold_slider", label_visibility="collapsed",
            min_value=0.5, max_value=1.0, value=MATCH_THRESHOLD, step=0.01,
            help="Lower = more lenient. Use ~0.65 for partial queries or translated titles."
        )
        st.caption("💡 Lower threshold (~0.65) works better for partial or translated input")

    st.caption("ℹ️ **(A)** URL only → title auto-extracted  |  **(B)** Title only  |  **(C)** Both → typed title takes priority")

    with st.expander("ℹ️ What is AHT and how is it calculated?"):
        st.markdown("""
        **Average Handling Time (AHT)** measures how long it takes to complete one product match.

        | Metric | Description |
        |---|---|
        | **AHT per search** | Time from clicking "Find Match" to result shown |
        | **Average AHT** | Total time ÷ number of searches |
        | **AHT Reduction** | Compare manual team's time vs this tool's AHT |

        *Manual team baseline AHT is typically 3–10 minutes per product. This tool targets < 5 seconds.*
        """)

    if st.button("🔍 Find Matching Product", type="primary", use_container_width=True, key="single_search_btn"):

        final_title          = product_title.strip()
        target_input         = target_site.strip()
        target_is_url        = target_input.startswith("http")
        target_extracted_title = None

        # Auto-extract title from source URL if no title typed
        if source_url.strip() and not final_title:
            with st.spinner("🌐 Extracting product title from source URL..."):
                extracted_info = extract_title_from_url(source_url.strip())
            if extracted_info["success"]:
                final_title = extracted_info["title"]
                st.success(f"✅ **Source title extracted** ({extracted_info['source']}): `{final_title}`")
            else:
                st.warning(f"⚠️ Could not extract from source URL: *{extracted_info['error']}* — please type the title manually.")

        # Auto-extract title from target URL if target is a full URL
        if target_is_url:
            with st.spinner("🌐 Extracting product title from target URL..."):
                target_extracted = extract_title_from_url(target_input)
            if target_extracted["success"]:
                target_extracted_title = target_extracted["title"]
                st.success(f"✅ **Target title extracted** ({target_extracted['source']}): `{target_extracted_title}`")
            else:
                st.warning(f"⚠️ Could not extract from target URL: *{target_extracted['error']}* — will filter by domain only.")

        if not final_title:
            st.warning("⚠️ Please enter a Product Title or a valid Source URL to extract from.")

        # ── CASE A: Direct URL-to-URL comparison ─────────────────────────────
        elif target_extracted_title and final_title:
            with st.spinner("🔎 Comparing source and target products directly..."):
                comparison = compare_two_products(final_title, target_extracted_title, model)

            st.session_state.aht_log.append(comparison["elapsed"])

            st.divider()
            st.subheader("🎯 Direct URL-to-URL Comparison Result")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Similarity Score", f"{comparison['score']:.4f}")
            col_r2.metric("Search Time",       f"{comparison['elapsed']:.2f}s")
            col_r3.metric("Result", "✅ Match" if comparison["is_match"] else "❌ No Match")

            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown("**📦 Source Product**")
                st.info(final_title)
                if source_url.strip():
                    st.markdown(f"[Open Source Link]({source_url.strip()})")
            with comp_col2:
                st.markdown("**🎯 Target Product**")
                st.info(target_extracted_title)
                st.markdown(f"[Open Target Link]({target_input})")

            if comparison["is_match"]:
                st.success(f"✅ **These are the SAME product!** Confidence: {comparison['score']:.1%}")
            else:
                st.error(f"❌ **Different products.** Score {comparison['score']:.4f} is below threshold {threshold}.")

            st.divider()
            st.subheader("📊 Other Top Matches from Target Site")
            with st.spinner("Running broader search..."):
                results     = find_match(final_title, model, index, df, threshold, target_filter=target_input)
            display_df  = results[["crawling title", "Link", "Similarity Score", "Match Status"]].copy()
            display_df.columns = ["Product Title", "Link", "Similarity Score", "Match Status"]
            show_results_table(display_df, link_cols=["Link"])

            save_result_with_aht(
                source_title=final_title,
                source_url=source_url.strip() if source_url.strip() else "N/A",
                target_site=target_input,
                match_title=target_extracted_title,
                target_link=target_input,
                score=comparison["score"],
                aht_seconds=comparison["elapsed"]
            )

        # ── CASE B: Title search ──────────────────────────────────────────────
        else:
            with st.spinner("🔎 Searching for matches..."):
                start_time = time.time()
                results    = find_match(
                    final_title, model, index, df,
                    threshold=threshold,
                    target_filter=target_input if target_input else None
                )
                elapsed = time.time() - start_time

            st.session_state.aht_log.append(elapsed)

            if target_input and results.attrs.get("domain_filtered"):
                st.info(f"🔍 Results filtered to **{results.attrs['domain_used']}** only")
            elif target_input and not results.attrs.get("domain_filtered"):
                st.warning(f"⚠️ No products from **{get_domain(target_input)}** found in dataset — showing all sites.")

            best_match = results.iloc[0]
            score      = float(best_match["Similarity Score"])
            is_match   = score >= threshold

            save_result_with_aht(
                source_title=final_title,
                source_url=source_url.strip() if source_url.strip() else "N/A",
                target_site=target_input if target_input else "N/A",
                match_title=best_match["crawling title"],
                target_link=best_match["Link"],
                score=score,
                aht_seconds=elapsed
            )

            query_tokens = set(clean_text(final_title).split())
            if len(query_tokens) <= 4:
                st.info(f"🔎 **Partial query detected** ({len(query_tokens)} words) — keyword boost applied.")

            st.subheader("📊 Top 5 Matches")
            display_df = results[["crawling title", "Link", "Similarity Score", "Match Status"]].copy()
            display_df.columns = ["Product Title", "Link", "Similarity Score", "Match Status"]
            show_results_table(display_df, link_cols=["Link"])

            st.divider()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🎯 Best Similarity Score", f"{score:.4f}")
            col_b.metric("⏱️ Search Time (AHT)",      f"{elapsed:.2f}s")
            col_c.metric("📈 Session Avg AHT",
                         f"{sum(st.session_state.aht_log)/len(st.session_state.aht_log):.2f}s")

            if is_match:
                st.success(f"✅ **Match Found!** Confidence: {score:.1%}")
                st.markdown(f"### 🔗 Best Match Link\n[Open Product]({best_match['Link']})")
            else:
                st.error(f"❌ **No Match Found.** Best score ({score:.4f}) is below threshold ({threshold}).")
                st.markdown(f"*Closest result: [{best_match['crawling title']}]({best_match['Link']})*")

        st.success(f"💾 Result saved to `{RESULTS_FILE}` with clickable links!")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("📂 Batch Product Matching — Upload Excel or CSV")

    with st.expander("📋 Download input template first", expanded=True):
        st.markdown("Your file must have **at least one of these columns** (column names are flexible):")
        st.markdown("""
        | Column | Required? | Description |
        |---|---|---|
        | `product_title` | ✅ OR source_url | Product name in any language |
        | `source_url` | ✅ OR product_title | URL — title auto-extracted if no title column |
        | `target_site` | Optional | Domain or retailer code to filter/compare against |
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
            mime="text/csv"
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your Excel (.xlsx) or CSV (.csv) file",
        type=["xlsx", "csv"],
        help="File must have product_title and/or source_url columns"
    )

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        batch_threshold = st.slider(
            "🎯 Match Threshold for Batch",
            min_value=0.5, max_value=1.0, value=MATCH_THRESHOLD, step=0.01,
            key="batch_threshold"
        )
    with b_col2:
        batch_target_filter = st.text_input(
            "🌐 Global Target Website (optional — applies to all rows without their own target)",
            placeholder="www.noon.com / www.extra.com  OR  retailer code: NN_KSA",
            key="batch_target"
        )

    if uploaded_file is not None:
        try:
            # ── Read raw file ──────────────────────────────────────────────────
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, header=0)
            else:
                raw_df = pd.read_excel(uploaded_file, header=0)

            # ── Fix duplicate column names ─────────────────────────────────────
            seen, new_cols = {}, []
            for col in raw_df.columns:
                col_clean = str(col).lower().strip().replace(" ", "_")
                if col_clean in seen:
                    seen[col_clean] += 1
                    new_cols.append(f"{col_clean}_{seen[col_clean]}")
                else:
                    seen[col_clean] = 0
                    new_cols.append(col_clean)
            raw_df.columns = new_cols

            st.success(f"✅ File loaded — **{len(raw_df)} rows**, **{len(raw_df.columns)} columns** detected")

            # Sanitize mixed-type columns for display
            display_df = raw_df.head(5).copy()
            for col in display_df.columns:
                if display_df[col].dtype == object or display_df[col].apply(type).nunique() > 1:
                    display_df[col] = display_df[col].astype(str)

            with st.expander("👁️ Preview uploaded file (first 5 rows)", expanded=True):
                show_results_table(display_df)
                st.caption(f"All columns found: `{', '.join(raw_df.columns)}`")

            # ── Column Mapping UI ──────────────────────────────────────────────
            st.markdown("### 🗂️ Map Your Columns")
            st.caption("Tell the app which column contains what. Select **None** to skip.")

            all_cols = ["— None —"] + list(raw_df.columns)

            def best_guess(keywords):
                for kw in keywords:
                    for col in raw_df.columns:
                        if kw in col:
                            return col
                return "— None —"

            map_col1, map_col2, map_col3, map_col4 = st.columns(4)
            with map_col1:
                title_col = st.selectbox(
                    "📦 Product Title column", all_cols,
                    index=all_cols.index(best_guess(["crawling_title", "title", "product", "name"])),
                    key="map_title"
                )
            with map_col2:
                url_col = st.selectbox(
                    "🔗 Source URL column", all_cols,
                    index=all_cols.index(best_guess(["source_url", "url", "link", "source"])),
                    key="map_url"
                )
            with map_col3:
                target_col = st.selectbox(
                    "🌐 Target Retailer Code column", all_cols,
                    index=all_cols.index(best_guess(["target_retailer_code", "target_retailer", "target_site", "target"])),
                    key="map_target",
                    help="Retailer code (TY, NN_KSA) OR domain (trendyol.com)"
                )
            with map_col4:
                domain_col = st.selectbox(
                    "🔗 Target Domain column (url.1)", all_cols,
                    index=all_cols.index(best_guess(["url_1", "url.1", "target_domain", "domain"])),
                    key="map_domain"
                )

            with st.expander("📋 Supported retailer codes"):
                code_df = pd.DataFrame(
                    [(k, v) for k, v in RETAILER_DOMAIN_MAP.items()],
                    columns=["Retailer Code", "Domain"]
                )
                st.dataframe(code_df, hide_index=True, use_container_width=True)

            # Build working DataFrame from mapped columns
            input_df = pd.DataFrame()
            if title_col  != "— None —": input_df["product_title"] = raw_df[title_col].astype(str).str.strip()
            if url_col    != "— None —": input_df["source_url"]    = raw_df[url_col].astype(str).str.strip()
            if target_col != "— None —": input_df["target_site"]   = raw_df[target_col].astype(str).str.strip()
            if domain_col != "— None —": input_df["url_1"]         = raw_df[domain_col].astype(str).str.strip()

            if input_df.empty or ("product_title" not in input_df.columns and "source_url" not in input_df.columns):
                st.warning("⚠️ Please map at least the **Product Title** or **Source URL** column to proceed.")
                input_df = None
            else:
                st.success(f"✅ Ready to process **{len(input_df)} products** using mapped columns.")

        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            input_df = None

        if input_df is not None and st.button("🚀 Run Batch Matching", type="primary",
                                               use_container_width=True, key="batch_run_btn"):
            st.divider()
            st.subheader("⚙️ Processing...")

            progress_bar = st.progress(0)
            status_text  = st.empty()
            batch_start  = time.time()

            # Wrap Streamlit progress into the callback signature
            def st_progress(current, total, message):
                progress_bar.progress(current / total)
                status_text.text(f"Processing {current}/{total} — {message}")

            batch_results = run_batch(
                input_df, model, index, df,
                batch_threshold=batch_threshold,
                global_target=batch_target_filter.strip(),
                progress_callback=st_progress
            )

            total_time = time.time() - batch_start
            progress_bar.progress(1.0)
            status_text.text("✅ Done!")

            st.divider()
            st.subheader("📊 Batch Results")

            total_r   = len(batch_results)
            matched_r = len(batch_results[batch_results["Match Status"] == "Match Found"])
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Total Products",  total_r)
            m_col2.metric("✅ Matched",       matched_r)
            m_col3.metric("❌ Not Matched",   total_r - matched_r)
            m_col4.metric("Match Rate", f"{matched_r/total_r*100:.1f}%" if total_r else "0%")

            aht_col1, aht_col2 = st.columns(2)
            aht_col1.metric("⏱️ Total Processing Time", f"{total_time:.1f}s")
            aht_col2.metric("⚡ Avg AHT per Product",   f"{batch_results['AHT (seconds)'].mean():.3f}s")

            show_results_table(batch_results, link_cols=["Source URL", "Target Link"])

            st.divider()
            excel_bytes = generate_batch_excel(batch_results)
            st.download_button(
                label="⬇️ Download Results as Excel (with clickable links)",
                data=excel_bytes,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            st.caption("Excel file includes clickable links and a Summary sheet with match rate & AHT stats.")

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
        aht_df = pd.DataFrame({
            "Search #":      range(1, len(st.session_state.aht_log) + 1),
            "AHT (seconds)": [round(x, 3) for x in st.session_state.aht_log]
        })
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Searches", len(st.session_state.aht_log))
        col2.metric("Avg AHT",        f"{np.mean(st.session_state.aht_log):.3f}s")
        col3.metric("Min AHT",        f"{np.min(st.session_state.aht_log):.3f}s")
        col4.metric("Max AHT",        f"{np.max(st.session_state.aht_log):.3f}s")
        st.line_chart(aht_df.set_index("Search #"))

        st.divider()
        st.markdown("### 🧮 AHT Reduction Calculator")
        manual_aht_min = st.number_input(
            "Enter your team's manual AHT per product (minutes):",
            min_value=1, max_value=60, value=5, key="manual_aht"
        )
        manual_sec = manual_aht_min * 60
        avg_ai     = np.mean(st.session_state.aht_log)
        reduction  = ((manual_sec - avg_ai) / manual_sec) * 100
        st.success(f"🚀 **AHT Reduction: {reduction:.1f}%** — AI: {avg_ai:.3f}s vs Manual: {manual_sec}s per product")
        st.caption(
            f"If your team processes 100 products/day, this tool saves "
            f"**{(manual_sec - avg_ai) * 100 / 60:.0f} minutes** daily."
        )