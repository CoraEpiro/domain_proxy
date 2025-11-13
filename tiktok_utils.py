import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REFERENCE_DATA_DIR = BASE_DIR / "reference_data"

TIKTOK_FILENAME = "TikTok 90 days history views - Sheet1.csv"
TIKTOK_DATA_PATH = DATA_DIR / TIKTOK_FILENAME
TIKTOK_FALLBACK_PATH = REFERENCE_DATA_DIR / TIKTOK_FILENAME


def _ensure_dataset(csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    if csv_path.exists():
        return csv_path

    if TIKTOK_FALLBACK_PATH.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(TIKTOK_FALLBACK_PATH, csv_path)
        return csv_path

    return csv_path


def _parse_tiktok_dates(raw_dates: pd.Series) -> pd.Series:
    raw = raw_dates.astype(str).str.strip()

    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    def parse_with_format(fmt: str):
        mask = parsed.isna()
        if not mask.any():
            return
        parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")

    # Try known formats explicitly to avoid pandas inference warnings
    parse_with_format("%Y-%m-%d %H:%M:%S")
    parse_with_format("%Y-%m-%d")
    parse_with_format("%m/%d/%Y")

    # Handle month/day values by appending the current year
    mask = parsed.isna()
    if mask.any():
        year_appended = raw.loc[mask] + f"/{datetime.now().year}"
        parsed.loc[mask] = pd.to_datetime(year_appended, format="%m/%d/%Y", errors="coerce")

    # As a last resort, parse bare month/day and set the current year
    mask = parsed.isna()
    if mask.any():
        month_day = pd.to_datetime(raw.loc[mask], format="%m/%d", errors="coerce")
        parsed.loc[mask] = month_day.apply(
            lambda dt: dt.replace(year=datetime.now().year) if pd.notna(dt) else dt
        )

    return parsed


def load_views_bsr_data(csv_path: Path = TIKTOK_DATA_PATH) -> pd.DataFrame:
    csv_path = _ensure_dataset(csv_path)
    if not csv_path.exists():
        return pd.DataFrame()

    df_raw = pd.read_csv(csv_path, header=1)
    df_raw.columns = df_raw.columns.str.strip()

    sum_of_views_col = None
    for col in df_raw.columns:
        if col.strip().lower() == "sum of views":
            sum_of_views_col = col
            break

    if sum_of_views_col:
        total_views = pd.to_numeric(
            df_raw[sum_of_views_col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
            errors="coerce",
        )
    else:
        view_cols = [col for col in df_raw.columns if "view" in col.lower()]
        numeric_cols = df_raw[view_cols].apply(
            lambda col: pd.to_numeric(
                col.astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce"
            )
        )
        total_views = numeric_cols.sum(axis=1, skipna=True)

    df = pd.DataFrame()
    df["date"] = _parse_tiktok_dates(df_raw["date"])
    df["total_views"] = total_views
    df["BSR Amazon"] = pd.to_numeric(
        df_raw.get("BSR Amazon", pd.Series(dtype=float)),
        errors="coerce"
    )

    exclude_days = {"10-03", "10-04", "10-05", "10-06"}
    if not df["date"].isna().all():
        df = df[~df["date"].dt.strftime("%m-%d").isin(exclude_days)]

    df = df.dropna(subset=["date", "BSR Amazon"])
    df = df[df["total_views"].notna()]
    # Exclude days with zero or negative views
    df = df[df["total_views"] > 0]
    df = df.sort_values("date")
    return df[["date", "total_views", "BSR Amazon"]]


def create_views_vs_bsr_chart(df: pd.DataFrame) -> go.Figure | None:
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["total_views"],
            name="TikTok Views",
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Views: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["BSR Amazon"],
            name="Amazon BSR",
            mode="lines+markers",
            line=dict(color="#ff7f0e"),
            yaxis="y2",
            hovertemplate="Date: %{x|%b %d, %Y}<br>BSR: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="TikTok Views vs. Amazon BSR",
        xaxis=dict(title="Date"),
        yaxis=dict(title="TikTok Views"),
        yaxis2=dict(
            title="Amazon BSR",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=450,
        template="plotly_white",
    )
    return fig


def create_views_line_chart(df: pd.DataFrame) -> go.Figure | None:
    if df.empty:
        return None
    fig = px.line(
        df,
        x="date",
        y="total_views",
        title="TikTok Views Over Time",
        markers=True,
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Views", template="plotly_white")
    return fig


def create_bsr_line_chart(df: pd.DataFrame) -> go.Figure | None:
    if df.empty:
        return None
    fig = px.line(
        df,
        x="date",
        y="BSR Amazon",
        title="Amazon BSR Over Time",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Amazon BSR",
        template="plotly_white",
    )
    return fig