# Dashboard Proxy

A minimal Next.js app that rewrites every request to the Streamlit dashboard at
`https://www.truseamossanalyzer.one/Performance_Dashboard`. Deploy it on Vercel,
attach your custom domain, and visitors will see the dashboard while the URL stays
on your domain.

## Deploy steps

1. Push this repo to GitHub (or import directly in Vercel).
2. Create a new Vercel project from it. Default build settings are fine.
3. Add your custom domain to the Vercel project. Vercel will ask for a DNS record,
   usually a CNAME pointing to `cname.vercel-dns.com`.
4. Once DNS propagates, the domain will load your dashboard.

Run locally with:

```bash
npm install
npm run dev
```

Visit `http://localhost:3000` and you’ll be proxied to the dashboard.

