"""
Step 1 of the scraping pipeline: HARVEST raw candidate images.

Pulls candidates from MULTIPLE search engines per query into
data/scraped_raw/<target>/. This step is deliberately "dumb" — it grabs anything
matching the query; the vision verifier (verify.py) does the real filtering.

Sources (all best-effort — a missing package or a failing engine is skipped, not
fatal, so you always get whatever the available engines return):
  1. DuckDuckGo images  (`ddgs`)        — keyless; aggregates the Bing index. Gives
                                          exact source URLs (best provenance).
  2. Google images      (`icrawler`)    — keyless crawler.
  3. Bing images        (`icrawler`)    — keyless crawler.
  4. Baidu images       (`icrawler`)    — keyless crawler; surfaces different/Asian
                                          results the Western engines miss.
  5. Reddit             (`praw`, opt.)  — r/3Dprinting, r/FixMyPrint, ... The single
                                          best source for in-situ "webcam-above-bed"
                                          failure photos, which is exactly the view
                                          our deployment needs. Needs REDDIT_CLIENT_ID
                                          / REDDIT_CLIENT_SECRET; no-op without them.

Using several engines widens recall AND colour/scene diversity, and means we don't
depend on any one engine's quirks or rate limits. All sources funnel through
_save_image(), which validates, content-hash de-dups across engines, and records
provenance in a per-target sources.jsonl manifest.

Legal/ethical note: small-scale academic research collection (thesis). Provenance
is logged per image; the dataset is not redistributed. Prefer official APIs/datasets
(Roboflow Universe, Kaggle) where licences are explicit — see README.md.

Install: pip install ddgs icrawler requests pillow
Optional Reddit: pip install praw  (+ REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET)
"""

import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

try:
    from ddgs import DDGS               # current package name
except ImportError:                     # pragma: no cover
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

from config import (CANDIDATES_PER_QUERY, ENABLED_ENGINES, MAX_RAW_PER_CLASS,
                    RAW_DIR, SAFE_SEARCH)

ACCEPTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_DIMENSION = 200          # drop thumbnails / icons
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (research thesis dataset collector; contact via repo)"
REDDIT_SUBS = ["3Dprinting", "FixMyPrint", "3Dprinting_help", "prusa3d", "BambuLab"]


def _save_image(content: bytes, dest_dir: Path, source: str) -> Path | None:
    """Validate, content-hash de-dup (across all engines), and write. Path or None."""
    try:
        Image.open(io.BytesIO(content)).verify()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return None
    if min(img.size) < MIN_DIMENSION:
        return None

    digest = hashlib.sha1(content).hexdigest()[:16]
    dest = dest_dir / f"{digest}.jpg"
    if dest.exists():
        return None                      # exact-byte dup already saved by another engine
    img.save(dest, "JPEG", quality=90)

    with (dest_dir / "sources.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"file": dest.name, "source": source}) + "\n")
    return dest


# --- Source 1: DuckDuckGo (exact URLs) -------------------------------------
def _from_ddgs(query: str, dest_dir: Path, n: int) -> int:
    if DDGS is None:
        return 0
    saved = 0
    try:
        with DDGS() as ddgs:
            ss = "on" if SAFE_SEARCH else "off"
            for r in ddgs.images(query, safesearch=ss, max_results=n * 3):
                if saved >= n:
                    break
                url = r.get("image")
                if not url or Path(url.split("?")[0]).suffix.lower() not in ACCEPTED_EXTS:
                    continue
                try:
                    resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                                        headers={"User-Agent": USER_AGENT})
                    if resp.status_code == 200 and resp.content and \
                       _save_image(resp.content, dest_dir, url):
                        saved += 1
                except requests.RequestException:
                    continue
                time.sleep(0.15)
    except Exception as e:
        print(f"        ! ddg '{query}': {e}")
    return saved


# --- Sources 2-4: icrawler engines (Google / Bing / Baidu) -----------------
def _from_icrawler(engine: str, query: str, dest_dir: Path, n: int) -> int:
    try:
        if engine == "google":
            from icrawler.builtin import GoogleImageCrawler as Crawler
        elif engine == "bing":
            from icrawler.builtin import BingImageCrawler as Crawler
        elif engine == "baidu":
            from icrawler.builtin import BaiduImageCrawler as Crawler
        else:
            return 0
    except ImportError:
        return 0

    saved = 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            crawler = Crawler(
                storage={"root_dir": tmp},
                downloader_threads=4,
                log_level=50,            # quiet (logging.CRITICAL)
            )
            crawler.crawl(keyword=query, max_num=n)
        except Exception as e:
            print(f"        ! {engine} '{query}': {e}")
            return 0
        # Ingest whatever it downloaded through our validate+dedup+manifest path.
        for p in Path(tmp).iterdir():
            if p.suffix.lower() not in ACCEPTED_EXTS:
                continue
            try:
                if _save_image(p.read_bytes(), dest_dir, f"icrawler:{engine}:{query}"):
                    saved += 1
            except Exception:
                continue
    return saved


# --- Source 5: Reddit (optional) -------------------------------------------
def _from_reddit(query: str, dest_dir: Path, n: int) -> int:
    cid = os.environ.get("REDDIT_CLIENT_ID")
    csec = os.environ.get("REDDIT_CLIENT_SECRET")
    if not (cid and csec):
        return 0
    try:
        import praw
    except ImportError:
        return 0
    try:
        reddit = praw.Reddit(client_id=cid, client_secret=csec,
                             user_agent="3d-defect-thesis-scraper/0.1")
    except Exception:
        return 0

    saved = 0
    for sub in REDDIT_SUBS:
        if saved >= n:
            break
        try:
            for post in reddit.subreddit(sub).search(query, limit=n):
                if saved >= n:
                    break
                url = getattr(post, "url", "")
                if Path(url.split("?")[0]).suffix.lower() not in ACCEPTED_EXTS:
                    continue
                try:
                    resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                                        headers={"User-Agent": USER_AGENT})
                    if resp.status_code == 200 and \
                       _save_image(resp.content, dest_dir, f"reddit:r/{sub}:{url}"):
                        saved += 1
                except requests.RequestException:
                    continue
        except Exception as e:
            print(f"        ! reddit r/{sub} '{query}': {e}")
    return saved


# Engines run per query. Each gets the full per-query budget; cross-engine
# duplicates collapse via the content hash, so more engines = more unique images.
SOURCES = [
    ("ddg",    _from_ddgs),
    ("google", lambda q, d, n: _from_icrawler("google", q, d, n)),
    ("bing",   lambda q, d, n: _from_icrawler("bing", q, d, n)),
    ("baidu",  lambda q, d, n: _from_icrawler("baidu", q, d, n)),
    ("reddit", _from_reddit),
]


def _count_images(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.suffix.lower() in ACCEPTED_EXTS)


def harvest_target(name: str, spec: dict) -> int:
    """Harvest queries for one target across the ENABLED engines, stopping at the
    MAX_RAW_PER_CLASS cap so we never pull a thousand images for one class."""
    dest_dir = RAW_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    engines = [(lbl, fn) for lbl, fn in SOURCES if lbl in ENABLED_ENGINES]
    queries = spec["search_queries"]
    print(f"  [{name}] up to {MAX_RAW_PER_CLASS} candidates | "
          f"engines={[e[0] for e in engines]} | {len(queries)} queries -> {dest_dir}")

    start = _count_images(dest_dir)
    for q in queries:
        if _count_images(dest_dir) >= start + MAX_RAW_PER_CLASS:
            print(f"  [{name}] hit cap of {MAX_RAW_PER_CLASS} — stopping harvest")
            break
        tally = []
        for label, fn in engines:
            got = fn(q, dest_dir, CANDIDATES_PER_QUERY)
            if got:
                tally.append(f"{label}+{got}")
        print(f"      '{q}': {', '.join(tally) if tally else 'nothing'}")

    total = _count_images(dest_dir) - start
    print(f"  [{name}] harvested {total} unique raw candidates")
    return total


if __name__ == "__main__":
    from config import all_targets
    for tname, tspec, _ in all_targets():
        harvest_target(tname, tspec)
