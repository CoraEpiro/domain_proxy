import streamlit as st
import pandas as pd
from datetime import datetime, date
from pathlib import Path

from tiktok_utils import (
    load_views_bsr_data,
    load_current_views_data,
    load_current_views_data_from_google_sheets,
    parse_google_sheets_url,
    get_date_to_tiktok_urls_from_google_sheets,
    parse_tiktok_video_id,
    get_tiktok_embed_url,
    merge_with_manual_tiktok_links,
    create_views_vs_bsr_chart,
    create_views_line_chart,
    create_bsr_line_chart,
    TIKTOK_DATA_PATH,
    TIKTOK_FILENAME,
    CURRENT_VIEWS_PATH,
    save_manual_bsr_entry,
    load_manual_bsr_entries,
    delete_manual_bsr_entry,
    get_file_modification_time,
)

APP_VERSION = "2025-11-16-1"

st.set_page_config(
    page_title="True Sea Moss Performance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_bsr_edit_modal(edit_date: str = None, edit_bsr: float = None):
    """Render BSR add/edit modal using Streamlit's form and columns."""
    expanded = edit_date is not None
    title = f"✏️ Edit BSR ({edit_date})" if edit_date else "➕ Add/Edit BSR"
    
    with st.expander(title, expanded=expanded):
        with st.form("bsr_entry_form"):
            col1, col2 = st.columns(2)
            with col1:
                if edit_date:
                    edit_date_obj = datetime.strptime(edit_date, "%Y-%m-%d").date()
                    bsr_date = st.date_input("Date", value=edit_date_obj)
                else:
                    bsr_date = st.date_input("Date", value=date.today())
            with col2:
                initial_bsr = edit_bsr if edit_bsr else 100.0
                bsr_value = st.number_input("BSR Value", min_value=0.0, value=initial_bsr, step=1.0)
            
            submitted = st.form_submit_button("Save BSR", use_container_width=True)
            if submitted:
                date_str = bsr_date.strftime("%Y-%m-%d")
                if save_manual_bsr_entry(date_str, bsr_value):
                    st.success(f"BSR {bsr_value} saved for {date_str}")
                    # Clear edit state
                    for key in list(st.session_state.keys()):
                        if key.startswith("edit_bsr_"):
                            del st.session_state[key]
                    st.rerun()
                else:
                    st.error("Failed to save BSR entry")


def render_current_mode_dashboard():
    """Render dashboard for current mode."""
    st.title("True Sea Moss Performance Dashboard - Current Mode")
    st.caption("Daily updated view counts with manual BSR entries.")
    
    # Initialize session state for current views file path and Google Sheets URL
    if "current_views_file_path" not in st.session_state:
        st.session_state.current_views_file_path = str(CURRENT_VIEWS_PATH)
    if "current_views_google_sheets_url" not in st.session_state:
        # Set default Google Sheets URL
        st.session_state.current_views_google_sheets_url = "https://docs.google.com/spreadsheets/d/1QVgiu4hLJSq9m8yRPi40wnXtaYLaqxLiTEifqE3hI_Y/edit?gid=466367795#gid=466367795"
    if "use_google_sheets" not in st.session_state:
        st.session_state.use_google_sheets = True  # Default to Google Sheets
    
    # File upload and configuration section
    with st.expander("📁 Configure Data Source", expanded=False):
        data_source = st.radio(
            "Data Source",
            ["Google Sheets", "Local File"],
            index=0 if st.session_state.use_google_sheets else 1,
            help="Choose between Google Sheets (live data) or local CSV file"
        )
        
        if data_source == "Google Sheets":
            st.session_state.use_google_sheets = True
            google_sheets_url = st.text_input(
                "Google Sheets URL",
                value=st.session_state.current_views_google_sheets_url,
                help="Paste the Google Sheets shareable link"
            )
            sheet_name_input = st.text_input(
                "Sheet name (tab) to read from",
                value=st.session_state.get("current_views_sheet_name", "WorkinOn"),
                help="Optional. If set, links are extracted from this tab."
            )
            if st.button("Save Google Sheets URL"):
                if parse_google_sheets_url(google_sheets_url):
                    st.session_state.current_views_google_sheets_url = google_sheets_url
                    st.session_state.current_views_sheet_name = sheet_name_input or "WorkinOn"
                    st.success("✅ Google Sheets URL saved!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Google Sheets URL. Please check the link.")
            
            # Show current URL status
            if parse_google_sheets_url(st.session_state.current_views_google_sheets_url):
                st.success(f"✅ Connected to Google Sheets")
            else:
                st.warning("⚠️ Please enter a valid Google Sheets URL")
        
        else:
            st.session_state.use_google_sheets = False
            tab1, tab2 = st.tabs(["📤 Upload File", "📂 Set File Path"])
            
            with tab1:
                uploaded_file = st.file_uploader(
                    "Upload CSV file with view counts",
                    type=["csv"],
                    help="Upload a CSV file with date and view count columns"
                )
                if uploaded_file is not None:
                    # Save uploaded file to data directory
                    try:
                        file_path = CURRENT_VIEWS_PATH
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.current_views_file_path = str(file_path)
                        st.success(f"✅ File uploaded successfully to `{file_path.name}`")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save file: {e}")
            
            with tab2:
                file_path_input = st.text_input(
                    "File path for daily updated CSV",
                    value=st.session_state.current_views_file_path,
                    help="Enter the full path where your daily script writes the CSV file, or use the default path"
                )
                if st.button("Save Path"):
                    st.session_state.current_views_file_path = file_path_input
                    st.success("✅ Path saved!")
                    st.rerun()
                
                st.caption(f"💡 Default: `{CURRENT_VIEWS_PATH}`")
                st.caption("💡 Tip: Your daily script should write to this file path")
    
    # Determine data source and load data
    if st.session_state.use_google_sheets:
        # Load from Google Sheets
        current_file_path = None
        data_source_info = "Google Sheets"
    else:
        # Load from local file
        current_file_path = Path(st.session_state.current_views_file_path)
        data_source_info = f"Local file: {current_file_path.name}"
    
    # File status and refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.session_state.use_google_sheets:
            parsed_url = parse_google_sheets_url(st.session_state.current_views_google_sheets_url)
            if parsed_url:
                st.caption(f"📊 Data source: Google Sheets (live data)")
            else:
                st.warning("⚠️ Invalid Google Sheets URL")
        else:
            if current_file_path.exists():
                mod_time = get_file_modification_time(current_file_path)
                if mod_time:
                    st.caption(f"📄 Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.caption(f"📄 File found: `{current_file_path.name}`")
            else:
                st.warning(f"⚠️ File not found at `{current_file_path}`")
                st.info("👆 Use 'Configure Data Source' above to upload a file or set the path")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    # Determine if we're editing an entry
    edit_entry = None
    manual_entries = load_manual_bsr_entries()
    for key in st.session_state.keys():
        if key.startswith("edit_bsr_"):
            idx = int(key.split("_")[-1])
            if idx < len(manual_entries):
                edit_entry = {"date": manual_entries[idx]["date"], "bsr": manual_entries[idx]["bsr"]}
                break
    
    with col3:
        if edit_entry:
            render_bsr_edit_modal(edit_entry["date"], edit_entry["bsr"])
        else:
            render_bsr_edit_modal()

    # Persistence is automatic via SQLite (no manual backup UI)
    
    # Load current data
    if st.session_state.use_google_sheets:
        df = load_current_views_data_from_google_sheets(st.session_state.current_views_google_sheets_url)
    else:
        df = load_current_views_data(current_file_path)
    
    if df.empty:
        st.info(
            f"📊 No current data found. Please:\n"
            f"1. Upload a CSV file or configure the file path in 'Configure Data Source' above\n"
            f"2. Use 'Add/Edit BSR' to manually add BSR values\n"
            f"3. Your daily script should write to: `{current_file_path}`"
        )
        
        # Show manual BSR entries if any
        manual_entries = load_manual_bsr_entries()
        if manual_entries:
            st.markdown("### Manual BSR Entries")
            entries_df = pd.DataFrame(manual_entries)
            entries_df.columns = ["Date", "BSR"]
            st.dataframe(entries_df, use_container_width=True, hide_index=True)
        return
    
    # Show manual BSR entries management
    if manual_entries:
        with st.expander("📝 Manage Manual BSR Entries"):
            entries_df = pd.DataFrame(manual_entries)
            entries_df.columns = ["Date", "BSR"]
            for idx, row in entries_df.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{row['Date']}**: BSR = {row['BSR']}")
                with col2:
                    if st.button("Edit", key=f"edit_{idx}"):
                        st.session_state[f"edit_bsr_{idx}"] = True
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"delete_{idx}"):
                        if delete_manual_bsr_entry(row['Date']):
                            st.success("Deleted!")
                            st.rerun()
    
    daily_views = (
        df.groupby(df["date"].dt.floor("D"))
        .agg({"total_views": "sum", "BSR Amazon": "mean"})
        .rename(columns={"total_views": "Views Sum", "BSR Amazon": "Average BSR"})
        .reset_index()
        .rename(columns={"date": "Date"})
    )
    daily_views["Date"] = daily_views["Date"].dt.strftime("%Y-%m-%d")
    daily_views = daily_views.sort_values("Date", ascending=True)
    
    st.markdown("### Current Performance Dataset")
    st.caption("Daily view counts and manual BSR entries.")
    st.dataframe(
        daily_views,
        use_container_width=True,
        hide_index=True,
    )
    
    if not daily_views.empty:
        latest_row = daily_views.iloc[-1]
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            views_val = int(latest_row['Views Sum']) if pd.notna(latest_row['Views Sum']) else 0
            st.metric("Latest Daily Views", f"{views_val:,}")
        with metric_col2:
            bsr_val = latest_row['Average BSR'] if pd.notna(latest_row['Average BSR']) else None
            if bsr_val:
                st.metric("Latest Average BSR", f"{bsr_val:.0f}")
            else:
                st.metric("Latest Average BSR", "N/A")
    
    # Charts
    if not df.empty and df["total_views"].notna().any():
        line_col1, line_col2 = st.columns(2)
        with line_col1:
            views_line = create_views_line_chart(df)
            if views_line:
                st.plotly_chart(views_line, use_container_width=True)
        
        with line_col2:
            if df["BSR Amazon"].notna().any():
                bsr_line = create_bsr_line_chart(df)
                if bsr_line:
                    st.plotly_chart(bsr_line, use_container_width=True)
            else:
                st.info("Add BSR values to see the BSR chart")
        
        if df["BSR Amazon"].notna().any():
            st.markdown("### Combined Insights")
            combined_chart = create_views_vs_bsr_chart(df)
            if combined_chart:
                st.plotly_chart(combined_chart, use_container_width=True)
            
            # Compute correlation safely (drop NaNs and cast to numeric)
            corr_df = df[["total_views", "BSR Amazon"]].apply(pd.to_numeric, errors="coerce").dropna()
            if not corr_df.empty:
                correlation = corr_df["total_views"].corr(-corr_df["BSR Amazon"])
                st.metric(
                    "Correlation (Views vs BSR improvement)",
                    f"{correlation:.2f}",
                    help="Positive correlation indicates higher TikTok views correspond with better (lower) BSR values.",
                )
            else:
                st.metric(
                    "Correlation (Views vs BSR improvement)",
                    "N/A",
                    help="Add BSR values to compute the correlation.",
                )

            # Highlight potentially impactful TikTok videos (Google Sheets only)
            if st.session_state.use_google_sheets:
                st.markdown("### Potentially Impactful TikTok Videos")
                with st.expander("Filter criteria", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        min_views = st.number_input("Minimum views for consideration", min_value=0, value=5000, step=500)
                    with col_b:
                        min_bsr_improvement = st.number_input(
                            "Minimum BSR improvement (absolute decrease)", min_value=0, value=10, step=5,
                            help="Improvement is yesterday's BSR minus today's BSR; positive values mean better rank."
                        )

                # Build daily aggregates including day-over-day BSR change
                daily = (
                    df.groupby(df["date"].dt.floor("D"))
                    .agg({"total_views": "sum", "BSR Amazon": "mean"})
                    .reset_index()
                    .sort_values("date")
                )
                daily["prev_bsr"] = daily["BSR Amazon"].shift(1)
                daily["bsr_improvement"] = (daily["prev_bsr"] - daily["BSR Amazon"]).fillna(0)

                # Load TikTok URLs per date from Google Sheets (xlsx hyperlinks)
                sheet_name = st.session_state.get("current_views_sheet_name") or None
                date_to_urls = get_date_to_tiktok_urls_from_google_sheets(
                    st.session_state.current_views_google_sheets_url,
                    sheet_name=sheet_name,
                )

                candidates = daily[
                    (daily["total_views"] >= min_views) & (daily["bsr_improvement"] >= min_bsr_improvement)
                ]

                if candidates.empty:
                    st.caption("No dates match the current thresholds. Try lowering the minimums above.")
                    # Diagnostic: show URL counts per day
                    diag = []
                    for _, r in daily.iterrows():
                        d = pd.to_datetime(r["date"]).normalize()
                        diag.append({
                            "Date": pd.to_datetime(d).strftime("%Y-%m-%d"),
                            "Views": int(r["total_views"]) if pd.notna(r["total_views"]) else 0,
                            "BSR": int(r["BSR Amazon"]) if pd.notna(r["BSR Amazon"]) else None,
                            "URLs": len(date_to_urls.get(d, [])),
                        })
                    st.dataframe(pd.DataFrame(diag), use_container_width=True, hide_index=True)
                    # Optional: let user pick any date with URLs to preview embeds
                    dates_with_urls = [k for k, v in date_to_urls.items() if v]
                    if dates_with_urls:
                        pick = st.selectbox("Preview videos for a date (ignores thresholds)", options=[pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates_with_urls])
                        if pick:
                            import streamlit.components.v1 as components
                            urls = date_to_urls.get(pd.to_datetime(pick).normalize(), [])
                            shown = 0
                            for url in urls:
                                vid = parse_tiktok_video_id(url)
                                if vid:
                                    components.iframe(get_tiktok_embed_url(vid), height=640, scrolling=False)
                                else:
                                    embed_html = f'''<blockquote class="tiktok-embed" cite="{url}" data-video-id="" style="max-width: 605px;min-width: 325px;"><section></section></blockquote><script async src="https://www.tiktok.com/embed.js"></script>'''
                                    components.html(embed_html, height=700, scrolling=False)
                                shown += 1
                                if shown >= 3:
                                    break
                else:
                    import streamlit.components.v1 as components
                    for _, row in candidates.iterrows():
                        day = pd.to_datetime(row["date"]).normalize()
                        urls = date_to_urls.get(day, [])
                        if not urls:
                            continue
                        st.markdown(f"**{pd.to_datetime(day).strftime('%Y-%m-%d')}** — Views: {int(row['total_views']):,}, BSR Δ: {int(row['bsr_improvement'])}")
                        # Show up to first 3 embeds for that day
                        shown = 0
                        for url in urls:
                            vid = parse_tiktok_video_id(url)
                            if vid:
                                # Use direct iframe when we have a numeric video id
                                embed_url = get_tiktok_embed_url(vid)
                                components.iframe(embed_url, height=640, scrolling=False)
                            else:
                                # Fallback: use TikTok official embed snippet with the raw URL (works for vm.tiktok.com too)
                                embed_html = f'''
                                <blockquote class="tiktok-embed" cite="{url}" data-video-id="" style="max-width: 605px;min-width: 325px;">
                                  <section></section>
                                </blockquote>
                                <script async src="https://www.tiktok.com/embed.js"></script>
                                '''
                                components.html(embed_html, height=700, scrolling=False)
                            shown += 1
                            if shown >= 3:
                                break

                # Always show recent videos regardless of thresholds (last 7 days with links)
                st.markdown("### Recent TikTok Videos (last 7 days)")
                import streamlit.components.v1 as components
                recent_days = sorted(date_to_urls.keys())[-7:]
                if not recent_days:
                    st.caption("No TikTok links found in the selected sheet.")
                else:
                    for d in recent_days:
                        urls = date_to_urls.get(d, [])[:3]
                        if not urls:
                            continue
                        st.markdown(f"**{pd.to_datetime(d).strftime('%Y-%m-%d')}**")
                        for url in urls:
                            vid = parse_tiktok_video_id(url)
                            if vid:
                                components.iframe(get_tiktok_embed_url(vid), height=640, scrolling=False)
                            else:
                                embed_html = f'''<blockquote class="tiktok-embed" cite="{url}" data-video-id="" style="max-width: 605px;min-width: 325px;"><section></section></blockquote><script async src="https://www.tiktok.com/embed.js"></script>'''
                                components.html(embed_html, height=700, scrolling=False)


def render_historical_dashboard():
    """Render dashboard for historical mode."""
    st.title("True Sea Moss Performance Dashboard - Historical Mode")
    st.caption("90 days historical data analysis.")

    df = load_views_bsr_data(TIKTOK_DATA_PATH)
    if df.empty:
        st.warning(
            f"No TikTok performance data found. Add a CSV to `data/{TIKTOK_FILENAME}` (or update the dataset in `reference_data/`) to enable these charts."
        )
        return

    daily_views = (
        df.groupby(df["date"].dt.floor("D"))
        .agg({"total_views": "sum", "BSR Amazon": "mean"})
        .rename(columns={"total_views": "Views Sum", "BSR Amazon": "Average BSR"})
        .reset_index()
        .rename(columns={"date": "Date"})
    )
    daily_views["Date"] = daily_views["Date"].dt.strftime("%Y-%m-%d")
    daily_views = daily_views.sort_values("Date", ascending=True)

    st.markdown("### TikTok Performance Dataset")
    st.caption("Recent TikTok view activity alongside Amazon Best Seller Rank.")
    st.dataframe(
        daily_views,
        use_container_width=True,
        hide_index=True,
    )

    latest_row = daily_views.iloc[-1]
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Latest Daily Views", f"{int(latest_row['Views Sum']):,}")
    with metric_col2:
        st.metric("Latest Average BSR", f"{latest_row['Average BSR']:.0f}")

    line_col1, line_col2 = st.columns(2)
    with line_col1:
        views_line = create_views_line_chart(df)
        if views_line:
            st.plotly_chart(views_line, use_container_width=True)

    with line_col2:
        bsr_line = create_bsr_line_chart(df)
        if bsr_line:
            st.plotly_chart(bsr_line, use_container_width=True)

    st.markdown("### Combined Insights")
    combined_chart = create_views_vs_bsr_chart(df)
    if combined_chart:
        st.plotly_chart(combined_chart, use_container_width=True)

    # Compute correlation safely (drop NaNs and cast to numeric)
    corr_df = df[["total_views", "BSR Amazon"]].apply(pd.to_numeric, errors="coerce").dropna()
    if not corr_df.empty:
        correlation = corr_df["total_views"].corr(-corr_df["BSR Amazon"])
    st.metric(
        "Correlation (Views vs BSR improvement)",
        f"{correlation:.2f}",
        help="Positive correlation indicates higher TikTok views correspond with better (lower) BSR values.",
    )
    else:
        st.metric(
            "Correlation (Views vs BSR improvement)",
            "N/A",
            help="Correlation requires both Views and BSR values.",
        )


def main():
    # Mode selector in sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio(
            "Select Mode",
            ["Historical", "Current"],
            help="Historical: 90 days historical data\nCurrent: Daily updated data with manual BSR entries"
        )
    
    if mode == "Historical":
        render_historical_dashboard()
    else:
        render_current_mode_dashboard()

if __name__ == "__main__":
    main()
