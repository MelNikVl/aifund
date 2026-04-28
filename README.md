# AI Pulse

A minimal AI-industry momentum dashboard. Scores OpenAI, Anthropic, DeepSeek,
and Perplexity on a 0–100 scale, surfaces the top 3–5 signals of the day, and
crowns a winner and loser — all auto-updated twice daily via GitHub Actions and
deployed as a static site on Vercel.

## File structure

```
/
├── index.html                  ← single-page frontend (vanilla JS, no build step)
├── data/
│   ├── feed.json               ← current snapshot (written by update script)
│   └── history.json            ← rolling 30-day history of scores
├── scripts/
│   └── update.py               ← fetch RSS → deduplicate → Claude → save JSON
└── .github/
    └── workflows/
        └── update.yml          ← cron at 07:00 & 19:00 UTC + manual trigger
```

## Setup

### 1. Fork / clone the repo

```bash
git clone https://github.com/melnikvl/aifund.git
cd aifund
```

### 2. Add the Anthropic API key secret

In your GitHub repository go to **Settings → Secrets and variables → Actions**
and create a new secret:

| Name | Value |
|------|-------|
| `ANTHROPIC_API_KEY` | your key from [console.anthropic.com](https://console.anthropic.com) |

The workflow will fail silently until this secret is present.

### 3. Connect Vercel

1. Go to [vercel.com/new](https://vercel.com/new) and import this repository.
2. Framework preset: **Other** (it's a static site — no build command needed).
3. Output directory: leave blank (Vercel will serve `index.html` from root).
4. Deploy.

Vercel auto-deploys on every push to `main`, so each data update triggers a
fresh CDN cache. No additional Vercel configuration is required.

### 4. Trigger the first update

Either wait for the next scheduled run (07:00 or 19:00 UTC), or go to
**Actions → Update AI Pulse feed → Run workflow** to trigger it manually.
The `data/feed.json` included in this repo contains sample data so the
frontend works immediately without running the pipeline.

## Local development

Open `index.html` directly in a browser — it reads `data/feed.json` via
`fetch()`. Because of browser CORS restrictions you may need a local server:

```bash
python -m http.server 8000
# then open http://localhost:8000
```

To run the update script locally:

```bash
pip install anthropic feedparser
ANTHROPIC_API_KEY=sk-ant-... python scripts/update.py
```

## Customisation

| What | Where |
|------|-------|
| Companies tracked | `SYSTEM_PROMPT` in `scripts/update.py` |
| RSS sources | `FEEDS` list in `scripts/update.py` |
| Update schedule | `cron` in `.github/workflows/update.yml` |
| Lookback window | `LOOKBACK_HOURS` in `scripts/update.py` |
| History retention | `history[-60:]` slice in `scripts/update.py` |
| Colour theme | CSS variables in `index.html` |
