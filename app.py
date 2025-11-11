import streamlit as st
import pandas as pd

from tiktok_utils import (
    load_views_bsr_data,
    create_views_vs_bsr_chart,
    create_views_line_chart,
    create_bsr_line_chart,
    TIKTOK_DATA_PATH,
    TIKTOK_FILENAME,
)

st.set_page_config(
    page_title="True Sea Moss Performance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_dashboard():
    st.title("True Sea Moss Performance Dashboard")
    st.caption("Real-time correlations between TikTok reach and Amazon Best Seller Rank.")

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

    correlation = df["total_views"].corr(-df["BSR Amazon"])
    st.metric(
        "Correlation (Views vs BSR improvement)",
        f"{correlation:.2f}",
        help="Positive correlation indicates higher TikTok views correspond with better (lower) BSR values.",
    )


def main():
    render_dashboard()

if __name__ == "__main__":
    main()
