#!/usr/bin/env python3
"""AI Pulse — data update script.

Sources:
  - RSS feeds (news, Reddit, HN)        → live
  - GitHub API (stars/commits/forks)    → live
  - HuggingFace API (model downloads)   → live
  - PyPI download stats                 → live
  - ArXiv RSS (research papers)         → live
  - Hacker News API (frontpage)         → live
  [planned] Reddit API, YouTube, AppStore, Jobs, Crunchbase
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
import xml.etree.ElementTree as ET

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic feedparser")

# ── Config ────────────────────────────────────────────────────────────────────

LOOKBACK_HOURS = 48
MAX_ITEMS      = 60
SIMILARITY_THRESHOLD = 0.6

REPO_ROOT  = Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / "data"
FEED_FILE  = DATA_DIR / "feed.json"
HISTORY_FILE = DATA_DIR / "history.json"
BETS_FILE  = DATA_DIR / "bets.json"
SIGNALS_FILE = DATA_DIR / "signals_raw.json"  # debug: raw collected signals

# ── RSS feeds ─────────────────────────────────────────────────────────────────

FEEDS = [
    "https://openai.com/news/rss.xml",
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml",
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_research.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://blog.google/technology/ai/rss/",
    "https://engineering.fb.com/feed/",
    "https://raw.githubusercontent.com/0xSMW/rss-feeds/main/feeds/feed_xai_news.xml",
    "https://raw.githubusercontent.com/0xSMW/rss-feeds/main/feeds/feed_mistral_news.xml",
    "https://blog.bittensor.com/feed",
    "https://www.reddit.com/r/bittensor_.rss",
    "https://www.reddit.com/r/GonkaAI/.rss",
    "https://hnrss.org/frontpage?q=Bittensor+TAO+Gonka+Cocoon+decentralized+AI&count=15",
    "https://www.theverge.com/rss/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://wired.com/feed/rss",
    "https://www.reddit.com/r/MachineLearning/.rss",
    "https://www.reddit.com/r/artificial/.rss",
    "https://www.reddit.com/r/OpenAI/.rss",
    "https://www.reddit.com/r/LocalLLaMA/.rss",
    "https://hnrss.org/frontpage?q=AI+LLM+Claude+GPT+DeepSeek+Gemini+Llama+Mistral&count=30",
    # ArXiv — AI/ML papers
    "https://techcrunch.com/category/venture/feed/",
    # Major news outlets
    "https://www.theguardian.com/technology/artificialintelligenceai/rss",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.technologyreview.com/feed/",
    "https://rss.arxiv.org/rss/cs.AI",
    "https://rss.arxiv.org/rss/cs.LG",
    "https://rss.arxiv.org/rss/cs.CL",
    "https://www.producthunt.com/feed",
]

# ── GitHub repos to track ─────────────────────────────────────────────────────

GITHUB_REPOS = {
    "OpenAI":     ["openai/openai-python", "openai/openai-node", "openai/whisper", "openai/chatgpt-retrieval-plugin"],
    "Anthropic":  ["anthropics/anthropic-sdk-python", "anthropics/anthropic-sdk-typescript"],
    "Google":     ["google-gemini/generative-ai-python", "google-deepmind/gemma", "google-deepmind/alphafold"],
    "Meta":       ["meta-llama/llama3", "facebookresearch/llama", "meta-llama/llama-models"],
    "DeepSeek":   ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-Coder-V2"],
    "Mistral":    ["mistralai/mistral-src", "mistralai/mistral-inference"],
    "xAI":        ["xai-org/grok-1"],
    "Perplexity": [],
}

# ── HuggingFace orgs to track ─────────────────────────────────────────────────

HF_ORGS = {
    "OpenAI":     "openai",
    "Anthropic":  "anthropic",
    "Google":     "google",
    "Meta":       "meta-llama",
    "DeepSeek":   "deepseek-ai",
    "Mistral":    "mistralai",
    "xAI":        "xai-org",
    "Perplexity": "perplexity-ai",
}

# ── PyPI packages to track ────────────────────────────────────────────────────

PYPI_PACKAGES = {
    "OpenAI":     "openai",
    "Anthropic":  "anthropic",
    "Google":     "google-generativeai",
    "Mistral":    "mistralai",
    "DeepSeek":   "deepseek",
}

# ── ArXiv affiliation keywords ────────────────────────────────────────────────

ARXIV_KEYWORDS = {
    "OpenAI":     ["openai", "chatgpt", "gpt-4", "gpt-5"],
    "Anthropic":  ["anthropic", "claude"],
    "Google":     ["google deepmind", "google brain", "deepmind", "google research", "gemini"],
    "Meta":       ["meta ai", "fair,", "meta platforms", "llama"],
    "DeepSeek":   ["deepseek"],
    "Mistral":    ["mistral", "mixtral"],
    "xAI":        ["xai", "x.ai", "grok"],
    "Perplexity": ["perplexity"],
    "Bittensor":  ["bittensor"],
    "Gensyn":     ["gensyn"],
}

# ── HN keywords per company ───────────────────────────────────────────────────

HN_KEYWORDS = {
    "OpenAI":     ["openai", "chatgpt", "gpt-4", "gpt-5", "gpt-4o", "dall-e", "sora", "o3", "o4"],
    "Anthropic":  ["anthropic", "claude"],
    "Google":     ["gemini", "google ai", "deepmind", "bard", "google deepmind"],
    "Meta":       ["meta ai", "llama", "meta llm", "llama 3", "llama3"],
    "DeepSeek":   ["deepseek", "deep seek"],
    "Mistral":    ["mistral", "mixtral"],
    "xAI":        ["xai", "grok", "x.ai"],
    "Perplexity": ["perplexity"],
    "Bittensor":  ["bittensor", "tao token", " tao "],
    "Gensyn":     ["gensyn"],
    "Gonka":      ["gonka"],
    "Cocoon":     ["cocoon ai", "ton ai"],
}

# ── HTTP helper ───────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 15, headers: dict = None) -> bytes | None:
    default_headers = {"User-Agent": "AI-Pulse/1.0 (https://github.com/melnikvl/aifund)"}
    if headers:
        default_headers.update(headers)
    try:
        req = Request(url, headers=default_headers)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (URLError, Exception) as exc:
        print(f"  warn: failed {url}: {exc}", file=sys.stderr)
        return None


def fetch_json(url: str, timeout: int = 15, headers: dict = None) -> dict | list | None:
    raw = fetch_url(url, timeout=timeout, headers=headers)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  warn: json parse error {url}: {e}", file=sys.stderr)
        return None

# ── RSS helpers ───────────────────────────────────────────────────────────────

def parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def format_date_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d")


def extract_text(element) -> str:
    if element is None:
        return ""
    text = element.text or ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def parse_feed(raw: bytes) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return items

    tag = root.tag
    if "atom" in tag or "feed" in tag.lower():
        atom_ns = "http://www.w3.org/2005/Atom"
        for entry in root.findall(f"{{{atom_ns}}}entry"):
            title   = extract_text(entry.find(f"{{{atom_ns}}}title"))
            summary = extract_text(entry.find(f"{{{atom_ns}}}summary")) or \
                      extract_text(entry.find(f"{{{atom_ns}}}content"))
            pub = entry.find(f"{{{atom_ns}}}published") or entry.find(f"{{{atom_ns}}}updated")
            items.append({"title": title, "summary": summary, "published": pub.text if pub is not None else None})
    else:
        for item in root.iter("item"):
            title = extract_text(item.find("title"))
            desc  = extract_text(item.find("description")) or extract_text(item.find("summary"))
            pub   = item.find("pubDate") or item.find("published")
            items.append({"title": title, "summary": desc, "published": pub.text if pub is not None else None})

    return items


def fetch_recent_items(lookback_hours: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    all_items: list[dict] = []
    for url in FEEDS:
        print(f"    rss: {url[:80]}", file=sys.stderr)
        raw = fetch_url(url)
        if raw is None:
            continue
        for item in parse_feed(raw):
            pub = parse_date(item.get("published"))
            if pub is not None and pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub is None or pub >= cutoff:
                item["date"] = format_date_iso(pub)
                all_items.append(item)
    return all_items


def title_tokens(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(items: list[dict]) -> list[dict]:
    seen: list[set] = []
    unique: list[dict] = []
    for item in items:
        tokens = title_tokens(item.get("title", ""))
        if any(jaccard(tokens, s) >= SIMILARITY_THRESHOLD for s in seen):
            continue
        seen.append(tokens)
        unique.append(item)
    return unique

# ═══════════════════════════════════════════════════════════════════════════════
# ── STRUCTURED SIGNAL COLLECTORS ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def collect_github(github_token: str | None = None) -> dict:
    """
    Fetch stars, forks, open_issues, pushed_at for tracked repos.
    Returns per-company aggregated metrics.
    """
    print("  [github] fetching repo stats…", file=sys.stderr)
    results = {}
    headers = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    for company, repos in GITHUB_REPOS.items():
        total_stars = 0
        total_forks = 0
        total_issues = 0
        recent_push_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        for repo in repos:
            url = f"https://api.github.com/repos/{repo}"
            data = fetch_json(url, headers=headers)
            if not data or "stargazers_count" not in data:
                continue
            total_stars  += data.get("stargazers_count", 0)
            total_forks  += data.get("forks_count", 0)
            total_issues += data.get("open_issues_count", 0)
            pushed = parse_date(data.get("pushed_at"))
            if pushed and pushed.tzinfo is None:
                pushed = pushed.replace(tzinfo=timezone.utc)
            if pushed and pushed >= cutoff:
                recent_push_count += 1
            time.sleep(0.3)  # respect rate limit

        if total_stars > 0:
            results[company] = {
                "total_stars":      total_stars,
                "total_forks":      total_forks,
                "open_issues":      total_issues,
                "active_repos_7d":  recent_push_count,
                "repos_tracked":    len(repos),
            }
            print(f"    {company}: {total_stars:,} stars, {recent_push_count} active repos", file=sys.stderr)

    return results


def collect_huggingface() -> dict:
    """
    Fetch top models per org — downloads, likes, trending.
    Uses public HF API (no auth needed).
    """
    print("  [huggingface] fetching model stats…", file=sys.stderr)
    results = {}

    for company, org in HF_ORGS.items():
        url = f"https://huggingface.co/api/models?author={org}&sort=downloads&limit=5"
        data = fetch_json(url)
        if not data or not isinstance(data, list):
            continue

        total_downloads = 0
        total_likes     = 0
        top_models      = []

        for model in data[:5]:
            dl    = model.get("downloads", 0) or 0
            likes = model.get("likes", 0) or 0
            total_downloads += dl
            total_likes     += likes
            name = model.get("modelId", "").split("/")[-1]
            top_models.append(f"{name}({dl:,}dl)")

        if total_downloads > 0:
            results[company] = {
                "top5_weekly_downloads": total_downloads,
                "top5_total_likes":      total_likes,
                "top_models":            top_models[:3],
            }
            print(f"    {company}: {total_downloads:,} downloads", file=sys.stderr)
        time.sleep(0.5)

    return results


def collect_pypi() -> dict:
    """
    Fetch last-week download counts from pypistats.org (no auth).
    """
    print("  [pypi] fetching package downloads…", file=sys.stderr)
    results = {}

    for company, package in PYPI_PACKAGES.items():
        url = f"https://pypistats.org/api/packages/{package}/recent?period=week"
        data = fetch_json(url)
        if not data or "data" not in data:
            continue
        weekly = data["data"].get("last_week", 0)
        results[company] = {"weekly_downloads": weekly, "package": package}
        print(f"    {company} ({package}): {weekly:,}/week", file=sys.stderr)
        time.sleep(0.3)

    return results


def collect_arxiv_papers(lookback_days: int = 7) -> dict:
    """
    Count recent ArXiv papers mentioning each company in title/abstract.
    Uses the items already fetched from ArXiv RSS feeds.
    """
    print("  [arxiv] counting papers…", file=sys.stderr)
    results = {company: 0 for company in ARXIV_KEYWORDS}

    # Re-fetch just ArXiv feeds for clean paper data
    arxiv_feeds = [f for f in FEEDS if "arxiv" in f]
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    for url in arxiv_feeds:
        raw = fetch_url(url)
        if raw is None:
            continue
        for item in parse_feed(raw):
            pub = parse_date(item.get("published"))
            if pub is not None and pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub and pub < cutoff:
                continue

            text = (item.get("title", "") + " " + item.get("summary", "")).lower()
            for company, keywords in ARXIV_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    results[company] += 1

    for company, count in results.items():
        if count > 0:
            print(f"    {company}: {count} papers", file=sys.stderr)

    return {k: {"papers_7d": v} for k, v in results.items() if v > 0}


def collect_hackernews() -> dict:
    """
    Count HN frontpage/top stories mentioning each company (last 48h).
    Uses public Firebase API — no auth.
    """
    print("  [hackernews] fetching top stories…", file=sys.stderr)
    results = {company: 0 for company in HN_KEYWORDS}

    # Get top 100 story IDs
    ids_data = fetch_json("https://hacker-news.firebaseio.com/v0/topstories.json")
    if not ids_data:
        return {}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    checked = 0

    for story_id in ids_data[:100]:
        url  = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        item = fetch_json(url, timeout=8)
        if not item:
            continue

        ts   = item.get("time", 0)
        pub  = datetime.fromtimestamp(ts, tz=timezone.utc)
        if pub < cutoff:
            continue

        title = (item.get("title") or "").lower()
        url_  = (item.get("url")   or "").lower()
        text  = title + " " + url_

        for company, keywords in HN_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                results[company] += 1

        checked += 1
        time.sleep(0.1)

    print(f"    checked {checked} stories", file=sys.stderr)
    return {k: {"hn_mentions_48h": v} for k, v in results.items() if v > 0}




def collect_wikipedia() -> dict:
    """
    Track Wikipedia page views for each company (last 7 days).
    Uses Wikimedia REST API — no auth required.
    """
    print("  [wikipedia] fetching page views…", file=sys.stderr)
    
    WIKI_PAGES = {
        "OpenAI":     "OpenAI",
        "Anthropic":  "Anthropic",
        "Google":     "Google_DeepMind",
        "Meta":       "Meta_AI",
        "DeepSeek":   "DeepSeek",
        "Mistral":    "Mistral_AI",
        "xAI":        "xAI_(company)",
        "Perplexity": "Perplexity_AI",
        "Bittensor":  "Bittensor",
        "Gensyn":     "Distributed_computing",
    }
    
    results = {}
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)
    
    for company, page in WIKI_PAGES.items():
        try:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page}/daily/{start_str}/{end_str}"
            data = fetch_json(url)
            if not data or "items" not in data:
                continue
            total = sum(item.get("views", 0) for item in data["items"])
            if total > 0:
                results[company] = {"wiki_views_7d": total}
                print(f"    {company}: {total:,} views", file=sys.stderr)
        except Exception as e:
            pass
    
    return results



# ── CoinGecko token prices (DeAI) ────────────────────────────────────────────

COINGECKO_IDS = {
    "Bittensor": "bittensor",
    "Gensyn":    "gensyn",
    "Gonka":     "gonka-ai",
    "Cocoon":    "the-open-network",  # Cocoon runs on TON network
}

def collect_token_prices() -> dict:
    print("  [coingecko] fetching token prices…", file=sys.stderr)
    ids = ",".join(COINGECKO_IDS.values())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true&include_7d_change=true"
    data = fetch_json(url)
    if not data:
        return {}
    results = {}
    for company, cg_id in COINGECKO_IDS.items():
        if cg_id not in data:
            continue
        d = data[cg_id]
        price = d.get("usd", 0)
        change_24h = d.get("usd_24h_change", 0) or 0
        change_7d  = d.get("usd_7d_change", 0) or 0
        results[company] = {
            "price_usd":    round(price, 4),
            "change_24h":   round(change_24h, 2),
            "change_7d":    round(change_7d, 2),
        }
        print(f"    {company}: ${price:.3f} ({change_24h:+.1f}% 24h)", file=sys.stderr)
    return results


# ── App Store ratings ─────────────────────────────────────────────────────────

APPSTORE_IDS = {
    "OpenAI":     "openai-chatgpt",
    "Anthropic":  "claude-ai-anthropic",
    "Google":     "google-gemini-ai",
    "Perplexity": "perplexity-ask",
}

APPSTORE_SEARCH = {
    "OpenAI":     "chatgpt openai",
    "Anthropic":  "claude anthropic",
    "Google":     "gemini google ai",
    "Perplexity": "perplexity ai",
}

def collect_appstore() -> dict:
    print("  [appstore] fetching ratings…", file=sys.stderr)
    results = {}
    for company, query in APPSTORE_SEARCH.items():
        try:
            from urllib.parse import quote
            url = f"https://itunes.apple.com/search?term={quote(query)}&country=us&entity=software&limit=1"
            data = fetch_json(url)
            if not data or not data.get("results"):
                continue
            app = data["results"][0]
            rating = app.get("averageUserRating", 0)
            count  = app.get("userRatingCount", 0)
            results[company] = {
                "rating":       round(rating, 2),
                "rating_count": count,
            }
            print(f"    {company}: {rating:.1f}★ ({count:,} ratings)", file=sys.stderr)
        except Exception as e:
            print(f"    warn: appstore {company}: {e}", file=sys.stderr)
    return results


# ── Reddit sentiment ──────────────────────────────────────────────────────────

REDDIT_QUERIES = {
    "OpenAI":     ["openai", "chatgpt"],
    "Anthropic":  ["anthropic", "claude"],
    "Google":     ["google gemini", "deepmind"],
    "Meta":       ["meta llama", "llama3"],
    "DeepSeek":   ["deepseek"],
    "Mistral":    ["mistral ai"],
    "xAI":        ["xai grok"],
    "Perplexity": ["perplexity ai"],
    "Bittensor":  ["bittensor tao"],
    "Gensyn":     ["gensyn"],
}

REDDIT_SUBS = ["LocalLLaMA", "MachineLearning", "artificial", "OpenAI", "Singularity"]

def collect_reddit_sentiment() -> dict:
    print("  [reddit] fetching sentiment…", file=sys.stderr)
    results = {}
    for company, queries in REDDIT_QUERIES.items():
        total_score = 0
        post_count  = 0
        for sub in REDDIT_SUBS[:3]:  # limit to 3 subs to avoid rate limits
            for q in queries[:1]:    # one query per company
                try:
                    from urllib.parse import quote
                    url = f"https://www.reddit.com/r/{sub}/search.json?q={quote(q)}&sort=new&limit=10&t=week"
                    data = fetch_json(url, headers={"User-Agent": "AI-Pulse/1.0"})
                    if not data or "data" not in data:
                        continue
                    posts = data["data"].get("children", [])
                    for p in posts:
                        pd = p.get("data", {})
                        total_score += pd.get("score", 0)
                        post_count  += 1
                    time.sleep(0.5)
                except Exception:
                    pass
        if post_count > 0:
            avg_score = round(total_score / post_count, 1)
            results[company] = {
                "reddit_posts_7d":   post_count,
                "reddit_avg_score":  avg_score,
            }
            print(f"    {company}: {post_count} posts, avg score {avg_score}", file=sys.stderr)
    return results




# ── Funding signals ───────────────────────────────────────────────────────────

FUNDING_FEEDS = [
    "https://techcrunch.com/category/venture/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.theinformation.com/feed",
]

FUNDING_KEYWORDS = {
    "OpenAI":     ["openai"],
    "Anthropic":  ["anthropic"],
    "Google":     ["google deepmind", "google ai", "deepmind"],
    "Meta":       ["meta ai", "meta platforms"],
    "DeepSeek":   ["deepseek"],
    "Mistral":    ["mistral"],
    "xAI":        ["xai", "elon musk ai"],
    "Perplexity": ["perplexity"],
    "Bittensor":  ["bittensor", "opentensor"],
    "Gensyn":     ["gensyn"],
    "Gonka":      ["gonka"],
}

import re as _re

def extract_funding_amount(text: str) -> float | None:
    """Extract funding amount in millions from text."""
    patterns = [
        r"\$([\d,\.]+)\s*[Bb]illion",
        r"\$([\d,\.]+)\s*[Mm]illion",
        r"([\d,\.]+)\s*[Bb]illion\s*dollar",
        r"([\d,\.]+)\s*[Mm]illion\s*dollar",
    ]
    for pat in patterns:
        m = _re.search(pat, text, _re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(",", ""))
            if "illion" in pat.lower() and "billion" in pat.lower():
                return val * 1000
            elif "billion" in m.group(0).lower():
                return val * 1000
            return val
    return None


def collect_funding() -> dict:
    """Parse funding news for each company."""
    print("  [funding] scanning for investment news…", file=sys.stderr)
    results = {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    for url in FUNDING_FEEDS:
        raw = fetch_url(url)
        if not raw:
            continue
        for item in parse_feed(raw):
            pub = parse_date(item.get("published"))
            if pub is not None and pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub and pub < cutoff:
                continue
            text = (item.get("title", "") + " " + item.get("summary", "")).lower()
            for company, keywords in FUNDING_KEYWORDS.items():
                if not any(kw in text for kw in keywords):
                    continue
                amount = extract_funding_amount(item.get("title", "") + " " + item.get("summary", ""))
                if company not in results:
                    results[company] = {"funding_events": 0, "funding_total_m": 0}
                results[company]["funding_events"] += 1
                if amount:
                    results[company]["funding_total_m"] += amount

    for company, data in results.items():
        total = data["funding_total_m"]
        events = data["funding_events"]
        if total > 0:
            print(f"    {company}: {events} events, ${total:.0f}M", file=sys.stderr)
        else:
            print(f"    {company}: {events} mentions", file=sys.stderr)
    return results




# ── Crypto AI Sector Index ────────────────────────────────────────────────────

CRYPTO_AI_FILE = DATA_DIR / "crypto_ai.json"

KNOWN_DEAI_PROJECTS = {
    "bittensor":         {"name": "Bittensor",          "type": "training",  "company": "Bittensor"},
    "gensyn":            {"name": "Gensyn",              "type": "compute",   "company": "Gensyn"},
    "the-open-network":  {"name": "TON (Cocoon)",        "type": "inference", "company": "Cocoon"},
    "render-token":      {"name": "Render",              "type": "compute",   "company": None},
    "fetch-ai":          {"name": "Fetch.ai",            "type": "agents",    "company": None},
    "akash-network":     {"name": "Akash",               "type": "compute",   "company": None},
    "virtuals-protocol": {"name": "Virtuals Protocol",   "type": "agents",    "company": None},
    "venice-token":      {"name": "Venice Token",        "type": "inference", "company": None},
    "grass":             {"name": "Grass",               "type": "data",      "company": None},
    "eigenlayer":        {"name": "EigenCloud",          "type": "compute",   "company": None},
    "origintrail":       {"name": "OriginTrail",         "type": "data",      "company": None},
}

def collect_crypto_ai_sector() -> dict:
    """
    Fetch top AI tokens from CoinGecko, analyze new ones with Claude,
    build a DeAI sector index.
    """
    print("  [crypto_ai] fetching AI token sector…", file=sys.stderr)
    
    # Load existing data
    existing = {}
    if CRYPTO_AI_FILE.exists():
        try:
            existing = json.loads(CRYPTO_AI_FILE.read_text())
        except Exception:
            existing = {}
    
    known_ids = set(existing.get("tokens", {}).keys())
    
    # Fetch top 50 AI tokens from CoinGecko
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category=artificial-intelligence&order=market_cap_desc&per_page=20&page=1&price_change_percentage=24h,7d"
    coins = fetch_json(url)
    if not coins or not isinstance(coins, list):
        print("  warn: CoinGecko AI category fetch failed", file=sys.stderr)
        return existing
    
    tokens = existing.get("tokens", {})
    new_tokens = []
    
    for coin in coins:
        cid = coin.get("id", "")
        if not cid:
            continue
        
        price = coin.get("current_price", 0) or 0
        mcap = coin.get("market_cap", 0) or 0
        change_24h = coin.get("price_change_percentage_24h", 0) or 0
        change_7d = coin.get("price_change_percentage_7d_in_currency", 0) or 0
        volume = coin.get("total_volume", 0) or 0
        
        # Update price data for existing tokens
        if cid in tokens:
            tokens[cid].update({
                "price_usd": round(price, 6),
                "mcap_m": round(mcap / 1e6, 1),
                "change_24h": round(change_24h, 2),
                "change_7d": round(change_7d, 2),
                "volume_m": round(volume / 1e6, 1),
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        elif cid in KNOWN_DEAI_PROJECTS:
            # Known project — add directly
            info = KNOWN_DEAI_PROJECTS[cid]
            tokens[cid] = {
                "name": info["name"],
                "symbol": coin.get("symbol", "").upper(),
                "type": info["type"],
                "company": info["company"],
                "price_usd": round(price, 6),
                "mcap_m": round(mcap / 1e6, 1),
                "change_24h": round(change_24h, 2),
                "change_7d": round(change_7d, 2),
                "volume_m": round(volume / 1e6, 1),
                "ai_score": None,
                "ai_type": info["type"],
                "ai_summary": None,
                "verified": True,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            print(f"    + {info['name']} ({coin.get('symbol','').upper()}) ${price:.3f} mcap ${mcap/1e6:.0f}M", file=sys.stderr)
        else:
            # New unknown token — queue for Claude analysis
            new_tokens.append({
                "id": cid,
                "name": coin.get("name", ""),
                "symbol": coin.get("symbol", "").upper(),
                "price_usd": round(price, 6),
                "mcap_m": round(mcap / 1e6, 1),
                "change_24h": round(change_24h, 2),
                "change_7d": round(change_7d, 2),
                "volume_m": round(volume / 1e6, 1),
            })
    
    # Analyze new tokens with Claude — in batches of 8
    if new_tokens:
        print(f"  [crypto_ai] analyzing {len(new_tokens)} new tokens with Claude…", file=sys.stderr)
        try:
            analyzed = []
            for i in range(0, len(new_tokens), 8):
                batch = new_tokens[i:i+8]
                batch_result = analyze_new_tokens_with_claude(batch)
                analyzed.extend(batch_result)
                time.sleep(1)
            for token_data in analyzed:
                cid = token_data.pop("id", None)
                if cid:
                    tokens[cid] = token_data
                    if token_data.get("is_real_ai"):
                        print(f"    + NEW: {token_data.get('name')} ({token_data.get('symbol')}) — {token_data.get('ai_type')}", file=sys.stderr)
        except Exception as e:
            print(f"  warn: Claude token analysis failed: {e}", file=sys.stderr)
    
    # Calculate sector index
    sector_index = calculate_deai_index(tokens)
    
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sector_index": sector_index,
        "tokens": tokens,
    }
    
    CRYPTO_AI_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  wrote {CRYPTO_AI_FILE} ({len(tokens)} tokens)", file=sys.stderr)
    return result


def analyze_new_tokens_with_claude(new_tokens: list) -> list:
    """Use Claude to analyze if new tokens are real AI projects."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    tokens_text = json.dumps(new_tokens, indent=2)
    
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system="""You are a crypto/AI analyst. Given a list of tokens from CoinGecko's AI category, 
analyze each one and determine if it's a real AI infrastructure project.

For each token return JSON:
{
  "id": "<coingecko_id>",
  "name": "<name>",
  "symbol": "<symbol>",
  "is_real_ai": true/false,
  "ai_type": "compute|inference|training|agents|data|defi|other",
  "ai_summary": "one sentence description",
  "ai_score": 0-100,
  "company": null or company name if affiliated,
  "price_usd": <price>,
  "mcap_m": <mcap in millions>,
  "change_24h": <24h change>,
  "change_7d": <7d change>,
  "volume_m": <volume in millions>,
  "verified": false,
  "updated_at": "<ISO datetime>"
}

is_real_ai = true only if the project actually builds AI infrastructure, models, or tooling.
Reject pure memecoins, NFT projects, or projects that just use "AI" in name without substance.
Return only valid JSON array, no markdown.""",
        messages=[{"role": "user", "content": f"Analyze these tokens:\n{tokens_text}"}],
    )
    
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def calculate_deai_index(tokens: dict) -> dict:
    """Calculate weighted DeAI sector index (like S&P but for AI crypto)."""
    real_ai = {k: v for k, v in tokens.items() if v.get("is_real_ai", v.get("verified", False))}
    
    if not real_ai:
        return {"value": 50, "change_24h": 0, "change_7d": 0, "token_count": 0}
    
    # Market-cap weighted index
    total_mcap = sum(t.get("mcap_m", 0) for t in real_ai.values())
    if total_mcap == 0:
        return {"value": 50, "change_24h": 0, "change_7d": 0, "token_count": len(real_ai)}
    
    weighted_change_24h = sum(
        t.get("change_24h", 0) * t.get("mcap_m", 0) / total_mcap
        for t in real_ai.values()
    )
    weighted_change_7d = sum(
        t.get("change_7d", 0) * t.get("mcap_m", 0) / total_mcap
        for t in real_ai.values()
    )
    
    # Index value: start at 100, adjust by 7d performance
    base = 100
    index_value = round(base * (1 + weighted_change_7d / 100), 1)
    
    return {
        "value": index_value,
        "change_24h": round(weighted_change_24h, 2),
        "change_7d": round(weighted_change_7d, 2),
        "total_mcap_b": round(total_mcap / 1000, 2),
        "token_count": len(real_ai),
    }




# ── YouTube collector ─────────────────────────────────────────────────────────

# Official YouTube channels (use channel videos, not search)
YOUTUBE_OFFICIAL_CHANNELS = {
    "OpenAI":    "UCXZCJLdBC09xxGZ6gcdrc6A",
    "Anthropic": "UCrDwWp7EBBv4NwvScIpBDOA",
    "Google":    "UCP7jMXSY2xbc3KCAE0MHQ-A",
}

# Search queries for companies without official channels
YOUTUBE_QUERIES = {
    "Meta":       "Meta Llama AI official",
    "DeepSeek":   "DeepSeek AI model",
    "Mistral":    "Mistral AI model",
    "xAI":        "xAI Grok Elon Musk",
    "Perplexity": "Perplexity AI search",
    "Bittensor":  "Bittensor TAO crypto AI",
    "Gensyn":     "Gensyn AI compute",
    # Gonka — search by topic + Liberman brothers
    "Gonka":      "Gonka AI Liberman brothers gonka protocol",
}

def collect_youtube() -> dict:
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return {}
    
    print("  [youtube] fetching video stats…", file=sys.stderr)
    results = {}
    from urllib.parse import quote
    published_after = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def get_video_stats(video_ids):
        if not video_ids:
            return 0, 0, 0
        stats_url = (
            f"https://www.googleapis.com/youtube/v3/videos"
            f"?part=statistics&id={','.join(video_ids)}&key={api_key}"
        )
        stats_data = fetch_json(stats_url)
        if not stats_data or "items" not in stats_data:
            return 0, 0, 0
        views = sum(int(i.get("statistics", {}).get("viewCount", 0)) for i in stats_data["items"])
        likes = sum(int(i.get("statistics", {}).get("likeCount", 0)) for i in stats_data["items"])
        return len(stats_data["items"]), views, likes

    # Official channels — get their recent videos directly
    for company, channel_id in YOUTUBE_OFFICIAL_CHANNELS.items():
        try:
            url = (
                f"https://www.googleapis.com/youtube/v3/search"
                f"?part=snippet&channelId={channel_id}&type=video"
                f"&publishedAfter={published_after}"
                f"&order=viewCount&maxResults=10&key={api_key}"
            )
            data = fetch_json(url)
            if not data or "items" not in data:
                continue
            video_ids = [i["id"]["videoId"] for i in data["items"] if "videoId" in i.get("id", {})]
            count, views, likes = get_video_stats(video_ids)
            if count > 0:
                results[company] = {"yt_videos_7d": count, "yt_views_7d": views, "yt_likes_7d": likes}
                print(f"    {company} (official): {count} videos, {views:,} views", file=sys.stderr)
            time.sleep(0.3)
        except Exception as e:
            print(f"    warn: youtube channel {company}: {e}", file=sys.stderr)

    # Search-based for others
    for company, query in YOUTUBE_QUERIES.items():
        try:
            url = (
                f"https://www.googleapis.com/youtube/v3/search"
                f"?part=snippet&q={quote(query)}&type=video"
                f"&publishedAfter={published_after}"
                f"&order=viewCount&maxResults=10&key={api_key}"
            )
            data = fetch_json(url)
            if not data or "items" not in data:
                continue
            
            video_ids = [item["id"]["videoId"] for item in data["items"] if "videoId" in item.get("id", {})]
            if not video_ids:
                continue
            
            # Get view counts
            stats_url = (
                f"https://www.googleapis.com/youtube/v3/videos"
                f"?part=statistics&id={','.join(video_ids)}&key={api_key}"
            )
            stats_data = fetch_json(stats_url)
            if not stats_data or "items" not in stats_data:
                continue
            
            total_views = 0
            total_likes = 0
            video_count = len(stats_data["items"])
            
            for item in stats_data["items"]:
                s = item.get("statistics", {})
                total_views += int(s.get("viewCount", 0))
                total_likes += int(s.get("likeCount", 0))
            
            if video_count > 0:
                results[company] = {
                    "yt_videos_7d": video_count,
                    "yt_views_7d": total_views,
                    "yt_likes_7d": total_likes,
                }
                print(f"    {company}: {video_count} videos, {total_views:,} views", file=sys.stderr)
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"    warn: youtube {company}: {e}", file=sys.stderr)
    
    return results


def collect_all_signals(github_token: str | None = None) -> dict:
    """Run all collectors and merge results."""
    signals = {
        "github":      {},
        "huggingface": {},
        "pypi":        {},
        "arxiv":       {},
        "hackernews":  {},
        "wikipedia":   {},
        "tokens":      {},
        "appstore":    {},
        "reddit":      {},
        "funding":     {},
        "crypto_ai":   {},
        "youtube":     {},
    }

    try:
        signals["github"]      = collect_github(github_token)
    except Exception as e:
        print(f"  warn: github collector failed: {e}", file=sys.stderr)

    try:
        signals["huggingface"] = collect_huggingface()
    except Exception as e:
        print(f"  warn: hf collector failed: {e}", file=sys.stderr)

    try:
        signals["pypi"]        = collect_pypi()
    except Exception as e:
        print(f"  warn: pypi collector failed: {e}", file=sys.stderr)

    try:
        signals["arxiv"]       = collect_arxiv_papers()
    except Exception as e:
        print(f"  warn: arxiv collector failed: {e}", file=sys.stderr)

    try:
        signals["hackernews"]  = collect_hackernews()
    except Exception as e:
        print(f"  warn: hn collector failed: {e}", file=sys.stderr)

    try:
        signals["wikipedia"]   = collect_wikipedia()
    except Exception as e:
        print(f"  warn: wikipedia collector failed: {e}", file=sys.stderr)

    try:
        signals["tokens"]      = collect_token_prices()
    except Exception as e:
        print(f"  warn: tokens collector failed: {e}", file=sys.stderr)

    try:
        signals["appstore"]    = collect_appstore()
    except Exception as e:
        print(f"  warn: appstore collector failed: {e}", file=sys.stderr)

    try:
        signals["reddit"]      = collect_reddit_sentiment()
    except Exception as e:
        print(f"  warn: reddit collector failed: {e}", file=sys.stderr)

    try:
        signals["funding"]     = collect_funding()
    except Exception as e:
        print(f"  warn: funding collector failed: {e}", file=sys.stderr)

    try:
        signals["crypto_ai"]   = collect_crypto_ai_sector()
    except Exception as e:
        print(f"  warn: crypto_ai collector failed: {e}", file=sys.stderr)

    try:
        signals["youtube"]     = collect_youtube()
    except Exception as e:
        print(f"  warn: youtube collector failed: {e}", file=sys.stderr)

    # Save raw signals for debugging / sources.html
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS_FILE.write_text(json.dumps({
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }, indent=2))

    return signals


def build_signals_block(signals: dict) -> str:
    """Format structured signals into a readable block for Claude."""
    lines = ["=== STRUCTURED SIGNALS (quantitative, last 7 days) ==="]
    companies = set()
    for src in signals.values():
        companies.update(src.keys())

    for company in sorted(companies):
        parts = []
        gh = signals["github"].get(company)
        if gh:
            parts.append(f"GitHub: {gh['total_stars']:,} total stars, {gh['active_repos_7d']} active repos")

        hf = signals["huggingface"].get(company)
        if hf:
            parts.append(f"HuggingFace: {hf['top5_weekly_downloads']:,} weekly model downloads")

        pypi = signals["pypi"].get(company)
        if pypi:
            parts.append(f"PyPI ({pypi['package']}): {pypi['weekly_downloads']:,}/week")

        arxiv = signals["arxiv"].get(company)
        if arxiv:
            parts.append(f"ArXiv: {arxiv['papers_7d']} papers")

        hn = signals["hackernews"].get(company)
        if hn:
            parts.append(f"HN: {hn['hn_mentions_48h']} mentions")

        wiki = signals.get("wikipedia", {}).get(company)
        if wiki:
            parts.append(f"Wikipedia: {wiki['wiki_views_7d']:,} views/7d")

        token = signals.get("tokens", {}).get(company)
        if token:
            parts.append(f"Token: ${token['price_usd']} ({token['change_24h']:+.1f}% 24h, {token['change_7d']:+.1f}% 7d)")

        app = signals.get("appstore", {}).get(company)
        if app:
            parts.append(f"AppStore: {app['rating']}★ ({app['rating_count']:,} ratings)")

        reddit = signals.get("reddit", {}).get(company)
        if reddit:
            parts.append(f"Reddit: {reddit['reddit_posts_7d']} posts/7d avg score {reddit['reddit_avg_score']}")

        yt = signals.get("youtube", {}).get(company)
        if yt:
            parts.append(f"YouTube: {yt['yt_videos_7d']} videos, {yt['yt_views_7d']:,} views/7d")

        funding = signals.get("funding", {}).get(company)
        if funding:
            total = funding.get("funding_total_m", 0)
            events = funding.get("funding_events", 0)
            if total > 0:
                parts.append(f"Funding: ${total:.0f}M across {events} events this week")
            else:
                parts.append(f"Funding mentions: {events} this week")

        if parts:
            lines.append(f"\n{company}:")
            lines.extend(f"  - {p}" for p in parts)

    return "\n".join(lines)

# ── Claude analysis ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI news analyst. Given recent AI news headlines AND structured quantitative signals \
(GitHub activity, HuggingFace downloads, PyPI installs, ArXiv papers, HN mentions), \
return a JSON object with this exact structure:
{
  "updated_at": "<ISO datetime>",
  "scores": [
    { "name": "OpenAI",     "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Anthropic",  "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Google",     "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Meta",       "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "DeepSeek",   "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Mistral",    "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "xAI",        "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Perplexity", "score": 0-100, "delta": -10 to 10, "badge": "release|agents|regulatory|open model|controversy|quiet" },
    { "name": "Bittensor",  "score": 0-100, "delta": -10 to 10, "badge": "release|network|tokenomics|controversy|quiet" },
    { "name": "Gonka",      "score": 0-100, "delta": -10 to 10, "badge": "release|network|tokenomics|controversy|quiet" },
    { "name": "Cocoon",     "score": 0-100, "delta": -10 to 10, "badge": "release|network|tokenomics|controversy|quiet" }
  ],
  "signals": [
    { "company": "...", "text": "...one line max...", "tag": "...", "date": "YYYY-MM-DD" },
    ... 5-8 items
  ],
  "winner": { "name": "...", "text": "...one sentence, slightly ironic..." },
  "loser":  { "name": "...", "text": "...one sentence, slightly ironic..." }
}

Score rules:
- Higher = more positive momentum. Neutral baseline = 50.
- Weight: news 40% + GitHub/HF/PyPI activity 30% + community signals (Reddit/HN) 20% + ArXiv research 10%
- Delta = change from prior score (use previous scores if provided, else from 50 baseline)
- For Bittensor/Gonka/Cocoon: score based on network activity, token news, technical milestones.
- Signals: use actual publication dates where available.
Return ONLY valid JSON, no markdown, no explanation."""


def build_news_block(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items[:MAX_ITEMS], 1):
        title   = item.get("title", "").strip()
        summary = item.get("summary", "").strip()
        date    = item.get("date", "")
        if title:
            line = f"{i}. [{date}] {title}" if date else f"{i}. {title}"
            if summary:
                line += f" — {summary[:200]}"
            lines.append(line)
    return "\n".join(lines)


def analyse_with_claude(news_block: str, signals_block: str) -> dict:
    client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Load previous scores for delta calculation
    prev_scores = ""
    if FEED_FILE.exists():
        try:
            prev = json.loads(FEED_FILE.read_text())
            prev_scores = "\n\nPREVIOUS SCORES (for delta calculation):\n" + \
                "\n".join(f"  {s['name']}: {s['score']}" for s in prev.get("scores", []))
        except Exception:
            pass

    user_msg = (
        f"Current UTC time: {now_iso}\n"
        f"{prev_scores}\n\n"
        f"{signals_block}\n\n"
        f"=== NEWS ITEMS (last 48h) ===\n{news_block}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)

# ── Prediction Markets ─────────────────────────────────────────────────────────

BETS_GENERATE_PROMPT = """You are an AI industry analyst. Given recent AI news, generate 3 fresh prediction market questions.
Rules:
- Questions must be YES/NO answerable
- Resolvable within 7-14 days based on observable facts
- About specific AI companies, product launches, regulatory events, or market moves
- Must be directly connected to the news provided
- No duplicates of existing open questions listed below

Existing open questions (do not duplicate):
{existing}

Return ONLY valid JSON, no markdown:
[
  {{
    "id": "q_<6 random hex chars>",
    "text": "Will X do Y by <specific date>?",
    "created_at": "<ISO datetime now>",
    "resolves_at": "<ISO datetime 7-14 days from now>",
    "status": "open",
    "yes_votes": 0,
    "no_votes": 0,
    "resolution": null,
    "resolution_text": null
  }}
]"""

BETS_RESOLVE_PROMPT = """You are an AI industry analyst. Given recent AI news and open prediction questions,
resolve any questions whose resolution date has passed OR whose outcome is clearly determinable.

News summary:
{news}

Questions to evaluate:
{questions}

Return ONLY valid JSON:
[{{"id": "q_xxx", "resolution": "yes"|"no"|null, "resolution_text": "..."}}]"""


def load_bets() -> dict:
    if BETS_FILE.exists():
        try:
            return json.loads(BETS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"questions": []}


def save_bets(bets: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BETS_FILE.write_text(json.dumps(bets, indent=2, ensure_ascii=False))
    print(f"  wrote {BETS_FILE}", file=sys.stderr)


def generate_bets(news_block: str, existing_questions: list) -> list:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    existing_texts = "\n".join(f"- {q['text']}" for q in existing_questions) or "none"
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prompt = BETS_GENERATE_PROMPT.format(existing=existing_texts)
    message = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=1024,
        system=prompt,
        messages=[{"role": "user", "content": f"Current UTC time: {now_iso}\n\nRecent news:\n{news_block[:3000]}"}],
    )
    raw = re.sub(r"^```[a-z]*\n?", "", message.content[0].text.strip())
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def resolve_bets(news_block: str, open_questions: list) -> list:
    if not open_questions:
        return []
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    questions_text = json.dumps(
        [{"id": q["id"], "text": q["text"], "resolves_at": q["resolves_at"]} for q in open_questions], indent=2
    )
    prompt = BETS_RESOLVE_PROMPT.format(news=news_block[:3000], questions=questions_text)
    message = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=1024,
        system="You are an AI industry analyst. Return only valid JSON arrays.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = re.sub(r"^```[a-z]*\n?", "", message.content[0].text.strip())
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def update_bets(news_block: str) -> None:
    bets = load_bets()
    now  = datetime.now(timezone.utc)

    open_qs = [q for q in bets["questions"] if q["status"] == "open"]
    if open_qs:
        print("  resolving open bets…", file=sys.stderr)
        try:
            for r in resolve_bets(news_block, open_qs):
                for q in bets["questions"]:
                    if q["id"] == r["id"] and r.get("resolution") in ("yes", "no"):
                        q["resolution"]      = r["resolution"]
                        q["resolution_text"] = r.get("resolution_text", "")
                        q["status"]          = "resolved"
        except Exception as e:
            print(f"  warn: resolution failed: {e}", file=sys.stderr)

    for q in bets["questions"]:
        if q["status"] == "open":
            try:
                ra = datetime.fromisoformat(q["resolves_at"].replace("Z", "+00:00"))
                if ra < now - timedelta(days=2):
                    q["status"] = "expired"
            except Exception:
                pass

    open_count = sum(1 for q in bets["questions"] if q["status"] == "open")
    if open_count < 5:
        print("  generating new bets…", file=sys.stderr)
        try:
            new_qs = generate_bets(news_block, [q for q in bets["questions"] if q["status"] == "open"])
            bets["questions"] = (bets["questions"] + new_qs)[-50:]
        except Exception as e:
            print(f"  warn: bet generation failed: {e}", file=sys.stderr)

    save_bets(bets)

# ── Persistence ────────────────────────────────────────────────────────────────

def save_feed(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEED_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    generate_static_html(data)
    print(f"  wrote {FEED_FILE}", file=sys.stderr)


def save_history(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            history = []
    snapshot = {
        "updated_at": data.get("updated_at"),
        "scores": [
            {"name": s["name"], "score": s["score"], "delta": s["delta"], "badge": s["badge"]}
            for s in data.get("scores", [])
        ],
    }
    history.append(snapshot)
    history = history[-60:]
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    print(f"  wrote {HISTORY_FILE}", file=sys.stderr)


def generate_static_html(data: dict):
    signals_html = ""
    for s in data.get("signals", []):
        date_str = f" ({s.get('date', '')})" if s.get("date") else ""
        signals_html += f'<li><strong>{s.get("company","")}</strong>{date_str}: {s.get("text","")}</li>\n'

    scores_html = ""
    for info in data.get("scores", []):
        name  = info.get("name", "")
        delta = info.get("delta", 0)
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        scores_html += f'<li>{name}: {info.get("score","?")} ({delta_str}) — {info.get("badge","")}</li>\n'

    winner  = data.get("winner", {})
    loser   = data.get("loser", {})
    updated = data.get("updated_at", "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Pulse — Daily AI Industry Momentum</title>
<meta name="description" content="Daily momentum scores for OpenAI, Anthropic, Google, Meta, DeepSeek, Mistral, xAI and Perplexity.">
<link rel="canonical" href="https://app.ai-groundtruth.com/">
</head>
<body>
<h1>AI Pulse — AI Industry Momentum</h1>
<p>Updated: {updated}</p>
<h2>Scores</h2><ul>{scores_html}</ul>
<h2>Top Signals</h2><ul>{signals_html}</ul>
<h2>Winner: {winner.get("name","")}</h2><p>{winner.get("text","")}</p>
<h2>Loser: {loser.get("name","")}</h2><p>{loser.get("text","")}</p>
<p><a href="/">View live dashboard</a> | <a href="/sources.html">Data Sources</a></p>
</body></html>"""

    (DATA_DIR / "seo.html").write_text(html)
    print("  static seo.html generated", file=sys.stderr)

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY not set.")

    github_token = os.environ.get("GITHUB_TOKEN")  # optional, increases rate limit

    print("1/5  collecting structured signals (GitHub, HF, PyPI, ArXiv, HN)…", file=sys.stderr)
    signals = collect_all_signals(github_token)

    print("2/5  fetching RSS feeds…", file=sys.stderr)
    items = fetch_recent_items(LOOKBACK_HOURS)
    print(f"     {len(items)} raw items", file=sys.stderr)

    print("3/5  deduplicating…", file=sys.stderr)
    items = deduplicate(items)
    print(f"     {len(items)} unique items", file=sys.stderr)

    if not items:
        sys.exit("No items found — check feed URLs.")

    print("4/5  analysing with Claude…", file=sys.stderr)
    news_block    = build_news_block(items)
    signals_block = build_signals_block(signals)
    data = analyse_with_claude(news_block, signals_block)

    print("5/5  saving output…", file=sys.stderr)
    save_feed(data)
    save_history(data)
    update_bets(news_block)

    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
