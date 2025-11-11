# True Sea Moss Performance Dashboard (Streamlit Only)

This repository contains a standalone Streamlit app that renders the TikTok vs.
Amazon BSR dashboard. Deploy it directly (Railway, Render, Streamlit Community
Cloud, etc.) and map your new domain to this service—no proxy or redirect needed.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will start at `http://localhost:8501`.

## Deploy

1. Push this repo to your hosting provider (Railway, Render, Streamlit Cloud, etc.).
2. Set the working directory to the repo root and run `pip install -r requirements.txt`.
3. Start the service with `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
4. Attach your custom domain through your host’s settings.

The `data/` directory contains the TikTok CSV used for the charts; `reference_data/`
is a baked-in fallback if the runtime volume is empty.

