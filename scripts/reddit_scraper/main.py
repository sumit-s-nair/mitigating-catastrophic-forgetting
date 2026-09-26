"""
Reddit scraper — RedScrapsLib backend
=====================================
Scrapes every subreddit in SUBREDDITS in a single run and writes the same
per-subreddit artifacts the GNN/HGNN pipelines expect:

    data/{sub}_gnn_comments_raw.csv
    data/{sub}_gnn_nodes.csv
    data/{sub}_gnn_edges.csv
    data/{sub}_gnn_metadata.json
    data/{sub}_gnn.graphml
    data/{sub}_gnn_checked_posts.csv

Progress is flushed periodically, so interrupting with Ctrl-C keeps everything
collected so far and the next run resumes where this one stopped.

Requires: .NET 10 runtime, `pip install redscrapslib`, and Reddit session
cookies (see load_cookies below).
"""

import json
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import RedScrapsLib as rs


# ─── Configuration ───────────────────────────────────────────────────────────
# Must stay in sync with SUBREDDITS in scripts/hgnn/config.py.
SUBREDDITS = [
    "python",
    "learnpython",
    "django",
    "javascript",
    "node",
    "webdev",
    "devops",
    "sysadmin",
    "linux",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
COOKIES_FILE = PROJECT_ROOT / "cookies.txt"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

MAX_COMMENT_NODES = 8000    # per subreddit
POST_PAGE_SIZE = 100        # get_home hard max
COMMENT_FETCH_LIMIT = 500   # per post; the library returns fewer if it caps lower
FLUSH_EVERY_N_POSTS = 25
INTER_CALL_DELAY = 0.5      # a small gap avoids Reddit's ~480s sustained-hammering penalty

# get_home reads Reddit's own listing API, which caps at ~1000 items per sort.
# Sweeping several sorts and time windows pushes the reachable post count past
# that ceiling, since each listing exposes a different slice of the subreddit.
DISCOVERY_PASSES = [
    ("new", None),
    ("hot", None),
    ("rising", None),
    ("top", "all"),
    ("top", "year"),
    ("top", "month"),
    ("top", "week"),
]

CHECKED_COLUMNS = ["post_id", "title", "had_new_comments", "checked_at_utc"]
COMMENT_COLUMNS = [
    "id", "author", "body", "score", "created_utc",
    "subreddit", "link_id", "parent_id",
]


# ─── Cookies / session ───────────────────────────────────────────────────────
def parse_netscape_cookies(path: Path, domain: str | None = None) -> dict:
    cookies = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 7:
                continue
            c_domain, _, _, _, _, name, value = parts
            if domain is None or domain in c_domain:
                cookies[name] = value
    return cookies


# Reddit's logged-in session cookie. Without it Reddit returns nothing for these
# endpoints, and the C# layer surfaces that as a None result rather than an error.
AUTH_COOKIE_NAMES = {"reddit_session", "token_v2"}


def load_cookies():
    """Return (cookies, source_description). Prefers cookies.txt, falls back to the browser."""
    if COOKIES_FILE.exists():
        try:
            cookies = parse_netscape_cookies(COOKIES_FILE, domain="reddit.com")
            if cookies:
                return cookies, f"{COOKIES_FILE.name} ({len(cookies)} cookies)"
            print(f"  {COOKIES_FILE.name} contained no reddit.com cookies.")
        except Exception as exc:
            print(f"  Failed to parse {COOKIES_FILE.name}: {exc}")

    try:
        import browser_cookie3
    except ImportError:
        print("  browser_cookie3 not installed, so no browser fallback.")
        return None, None

    for browser_name in ("firefox", "chrome", "edge", "brave"):
        loader = getattr(browser_cookie3, browser_name, None)
        if loader is None:
            continue
        try:
            jar = list(loader(domain_name=".reddit.com"))
        except Exception as exc:
            # Chrome/Edge on Windows encrypt their cookie store app-bound, which
            # browser_cookie3 cannot read without elevation. Use cookies.txt instead.
            print(f"  {browser_name}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if jar:
            return jar, f"browser_cookie3 ({browser_name}, {len(jar)} cookies)"
        print(f"  {browser_name}: no reddit.com cookies (not logged in there?)")

    return None, None


def init_scraper() -> None:
    print("Loading Reddit cookies...")
    cookies, source = load_cookies()

    if cookies is None:
        print(
            "\nERROR: no Reddit cookies found, and Reddit returns nothing without them.\n"
            "  Fix with either:\n"
            f"   (a) export cookies.txt (Netscape format) for reddit.com to {COOKIES_FILE}\n"
            "   (b) log in to Reddit in Firefox, which browser_cookie3 can read unelevated\n"
        )
        raise SystemExit(1)

    names = set(cookies) if isinstance(cookies, dict) else {c.name for c in cookies}
    print(f"Loaded Reddit cookies from {source}.")
    if not (AUTH_COOKIE_NAMES & names):
        print(f"WARNING: none of {sorted(AUTH_COOKIE_NAMES)} present — these look like\n"
              "         logged-out cookies. Requests will likely return nothing.")

    rs.init(user_agent=USER_AGENT, cookies=cookies)


# ─── ID normalisation ────────────────────────────────────────────────────────
def to_post_fullname(raw: str) -> str:
    raw = str(raw or "").strip()
    if not raw:
        return ""
    return raw if raw.startswith("t3_") else f"t3_{raw}"


def to_base36(raw: str) -> str:
    raw = str(raw or "").strip()
    for prefix in ("t1_", "t3_"):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def normalize_parent_id(raw: str, post_base36: str) -> str:
    """
    RedScrapsLib's ParentID format is not documented, so accept either a Reddit
    fullname (t1_/t3_ prefixed) or a bare base36 id. A bare id equal to the post
    means a top-level comment; anything else is a reply to a sibling comment.
    """
    raw = str(raw or "").strip()
    if not raw:
        return ""
    if raw.startswith(("t1_", "t3_")):
        return raw
    return f"t3_{raw}" if raw == post_base36 else f"t1_{raw}"


# ─── Resume state ────────────────────────────────────────────────────────────
def load_existing_comments(raw_comments_file: Path, subreddit: str) -> pd.DataFrame:
    if not raw_comments_file.exists():
        return pd.DataFrame()

    try:
        existing_df = pd.read_csv(raw_comments_file)
    except Exception as exc:
        print(f"Failed to read existing comments file {raw_comments_file}: {exc}")
        return pd.DataFrame()

    if existing_df.empty:
        return existing_df

    if "id" not in existing_df.columns:
        print("Existing comments file does not contain 'id'. Starting fresh.")
        return pd.DataFrame()

    if "subreddit" not in existing_df.columns:
        existing_df["subreddit"] = subreddit.lower()
    else:
        existing_df["subreddit"] = existing_df["subreddit"].astype(str).str.lower()

    existing_df["id"] = existing_df["id"].astype(str)
    return existing_df.drop_duplicates(subset=["id"])


def load_checked_posts(checked_posts_file: Path) -> pd.DataFrame:
    empty = pd.DataFrame(columns=CHECKED_COLUMNS)
    if not checked_posts_file.exists():
        return empty

    try:
        checked_df = pd.read_csv(checked_posts_file)
    except Exception as exc:
        print(f"Failed to read checked posts file {checked_posts_file}: {exc}")
        return empty

    if checked_df.empty or "post_id" not in checked_df.columns:
        if not checked_df.empty:
            print("Checked posts file missing 'post_id'. Starting checked-posts tracking fresh.")
        return empty

    checked_df["post_id"] = checked_df["post_id"].astype(str)
    return checked_df.drop_duplicates(subset=["post_id"], keep="last")


# ─── Fetching ────────────────────────────────────────────────────────────────
class LibraryBugError(RuntimeError):
    """Raised when RedScrapsLib returns structurally unusable data. Aborts the whole run."""



def iter_unprocessed_posts(subreddit: str, processed_post_ids: set[str]):
    """Yield (post_fullname, title) for posts not yet processed, across every discovery pass."""
    seen_this_run: set[str] = set()

    for sort, time_filter in DISCOVERY_PASSES:
        after = None
        page_count = 0

        while True:
            try:
                page = rs.get_home(
                    subreddit,
                    sort=sort,
                    limit=POST_PAGE_SIZE,
                    time=time_filter,
                    after=after,
                )
            except Exception as exc:
                print(f"  get_home({subreddit}, sort={sort}, time={time_filter}) raised: {exc}")
                break

            # The library returns None on any non-429 failure rather than raising.
            if page is None:
                print(f"  get_home(sort={sort}, time={time_filter}) returned None "
                      "— request failed or cookies are not valid.")
                break

            posts = list(getattr(page, "Posts", None) or [])
            if not posts:
                break

            page_count += 1
            for post in posts:
                post_id = to_post_fullname(getattr(post, "PostID", ""))
                if not post_id or post_id in seen_this_run:
                    continue
                seen_this_run.add(post_id)
                if post_id in processed_post_ids:
                    continue
                yield post_id, str(getattr(post, "Title", "") or "")

            # Fewer than a full page means this listing is exhausted.
            if len(posts) < POST_PAGE_SIZE:
                break

            next_after = str(getattr(page, "LastID", "") or "")
            if not next_after or next_after == after:
                break
            after = next_after
            time.sleep(INTER_CALL_DELAY)

        print(f"  pass sort={sort} time={time_filter}: {page_count} pages, "
              f"{len(seen_this_run)} unique posts seen so far")


def fetch_comments_for_post(
    subreddit: str, post_id: str, known_comment_ids: set[str]
) -> tuple[pd.DataFrame, bool]:
    """
    Returns (comments, request_succeeded). The success flag matters: a failed
    request must not mark the post as checked, or resume logic would skip it
    permanently and its comments would never be collected.
    """
    base36 = to_base36(post_id)

    try:
        result = rs.get_comments(subreddit, post_id=base36, limit=COMMENT_FETCH_LIMIT)
    except Exception as exc:
        print(f"  get_comments raised for {post_id}: {exc}")
        return pd.DataFrame(), False

    if result is None:
        print(f"  get_comments returned None for {post_id} — leaving it unchecked to retry later.")
        return pd.DataFrame(), False

    comments = list(getattr(result, "Comments", None) or [])
    if not comments:
        return pd.DataFrame(), True

    # RedScrapsLib 0.1.7 set every CommentID (and ParentID) to the post's own id,
    # which silently collapses a whole thread into one comment and makes reply_to
    # edges impossible. Abort loudly rather than spend a night collecting garbage.
    raw_ids = [to_base36(getattr(c, "CommentID", "")) for c in comments]
    if len(comments) >= 3 and len(set(raw_ids)) == 1:
        raise LibraryBugError(
            f"get_comments returned {len(comments)} comments for {post_id} but only "
            f"1 distinct CommentID ({raw_ids[0]!r}"
            f"{', the post id' if raw_ids[0] == base36 else ''}).\n"
            "  The library is not mapping comment-level ids, so every thread would "
            "collapse to a single node.\n"
            "  Nothing was written. This needs a fix in RedScrap's C# Map layer."
        )

    result_subreddit = str(getattr(result, "Subreddit", "") or subreddit).lower()
    rows: list[dict] = []
    local_seen: set[str] = set()

    for comment in comments:
        comment_id = to_base36(getattr(comment, "CommentID", ""))
        if not comment_id or comment_id in known_comment_ids or comment_id in local_seen:
            continue
        local_seen.add(comment_id)

        rows.append(
            {
                "id": comment_id,
                "author": getattr(comment, "Author", None),
                "body": getattr(comment, "Body", None),
                # RedScrapsLib's get_comments does not expose score or created_utc.
                # Neither is consumed by the GNN/HGNN pipelines; the columns are kept
                # so the CSV schema stays stable.
                "score": None,
                "created_utc": None,
                "subreddit": result_subreddit,
                "link_id": f"t3_{base36}",
                "parent_id": normalize_parent_id(getattr(comment, "ParentID", ""), base36),
            }
        )

    return (pd.DataFrame(rows, columns=COMMENT_COLUMNS) if rows else pd.DataFrame()), True


# ─── Graph construction ──────────────────────────────────────────────────────
def build_gnn_tables(comments_df: pd.DataFrame, subreddit: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    comments_df = comments_df.copy()
    if "id" not in comments_df.columns:
        raise ValueError("Comment frame does not contain an 'id' column.")

    comments_df = comments_df.dropna(subset=["id", "subreddit"])
    comments_df = comments_df.drop_duplicates(subset=["id"])

    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    node_id_by_key: dict[str, int] = {}

    def add_node(node_key: str, node_type: str, text: str = "", score=None,
                 node_subreddit: str = "", label: int = -1) -> int:
        if node_key in node_id_by_key:
            return node_id_by_key[node_key]
        node_id = len(node_id_by_key)
        node_id_by_key[node_key] = node_id
        node_rows.append(
            {
                "node_id": node_id,
                "node_key": node_key,
                "node_type": node_type,
                "text": text,
                "score": score,
                "subreddit": node_subreddit,
                "label": label,
            }
        )
        return node_id

    def add_edge(src_key: str, dst_key: str, relation: str):
        edge_rows.append(
            {
                "src": node_id_by_key[src_key],
                "dst": node_id_by_key[dst_key],
                "relation": relation,
            }
        )

    # Pass 1: every comment node must exist before reply_to edges are wired up,
    # otherwise replies that arrive before their parent silently lose the edge.
    for row in comments_df.itertuples(index=False):
        comment_id = str(getattr(row, "id", ""))
        add_node(
            f"comment:{comment_id}",
            node_type="comment",
            text=str(getattr(row, "body", "") or ""),
            score=getattr(row, "score", None),
            node_subreddit=str(getattr(row, "subreddit", "")).lower(),
            label=0,
        )

    # Pass 2: author / post / reply edges.
    for row in comments_df.itertuples(index=False):
        comment_key = f"comment:{str(getattr(row, 'id', ''))}"

        author = str(getattr(row, "author", "") or "")
        if author and author.lower() not in {"[deleted]", "[removed]", "nan"}:
            author_key = f"user:{author.lower()}"
            add_node(author_key, node_type="user", text=author)
            add_edge(author_key, comment_key, relation="authored")

        link_id = str(getattr(row, "link_id", "") or "")
        if link_id and link_id.lower() != "nan":
            post_key = f"post:{link_id}"
            add_node(post_key, node_type="post", text=link_id)
            add_edge(comment_key, post_key, relation="on_post")

        parent_id = str(getattr(row, "parent_id", "") or "")
        if parent_id.startswith("t1_"):
            parent_comment_key = f"comment:{parent_id[3:]}"
            if parent_comment_key in node_id_by_key:
                add_edge(comment_key, parent_comment_key, relation="reply_to")

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)
    relation_counts = (
        edges_df["relation"].value_counts().to_dict() if not edges_df.empty else {}
    )
    metadata = {
        "subreddit": subreddit.lower(),
        "num_nodes": int(len(nodes_df)),
        "num_edges": int(len(edges_df)),
        "num_comment_nodes": int((nodes_df["node_type"] == "comment").sum()) if not nodes_df.empty else 0,
        "num_subreddits": 1,
        "label_map": {subreddit.lower(): 0},
        "edge_relations": {str(k): int(v) for k, v in relation_counts.items()},
        "source": "RedScrapsLib",
    }
    return nodes_df, edges_df, metadata


def save_graphml(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, out_file: Path) -> None:
    graph = nx.DiGraph()

    for row in nodes_df.itertuples(index=False):
        graph.add_node(
            int(row.node_id),
            node_key=str(row.node_key),
            node_type=str(row.node_type),
            text=str(row.text or ""),
            score="" if pd.isna(row.score) else float(row.score),
            subreddit=str(row.subreddit or ""),
            label=int(row.label),
        )

    for row in edges_df.itertuples(index=False):
        graph.add_edge(int(row.src), int(row.dst), relation=str(row.relation))

    nx.write_graphml(graph, out_file)


# ─── Per-subreddit driver ────────────────────────────────────────────────────
def scrape_subreddit(subreddit: str, summary: list[dict]) -> None:
    """
    Scrape one subreddit. Appends a summary row in a finally block so a Ctrl-C
    still records progress and writes artifacts before propagating.
    """
    raw_comments_file = DATA_DIR / f"{subreddit}_gnn_comments_raw.csv"
    nodes_file = DATA_DIR / f"{subreddit}_gnn_nodes.csv"
    edges_file = DATA_DIR / f"{subreddit}_gnn_edges.csv"
    meta_file = DATA_DIR / f"{subreddit}_gnn_metadata.json"
    graphml_file = DATA_DIR / f"{subreddit}_gnn.graphml"
    checked_posts_file = DATA_DIR / f"{subreddit}_gnn_checked_posts.csv"

    existing_comments_df = load_existing_comments(raw_comments_file, subreddit)
    bootstrap_full_scrape = existing_comments_df.empty

    if bootstrap_full_scrape:
        # No stored dataset, so old checked-post state would skip posts we never saved.
        checked_posts_df = pd.DataFrame(columns=CHECKED_COLUMNS)
        checked_posts_file.unlink(missing_ok=True)
    else:
        checked_posts_df = load_checked_posts(checked_posts_file)

    known_comment_ids = (
        set(existing_comments_df["id"].astype(str)) if not bootstrap_full_scrape else set()
    )
    processed_post_ids: set[str] = set()
    if not bootstrap_full_scrape:
        if "link_id" in existing_comments_df.columns:
            processed_post_ids.update(
                str(link_id) for link_id in existing_comments_df["link_id"].dropna().astype(str) if link_id
            )
        if not checked_posts_df.empty:
            processed_post_ids.update(checked_posts_df["post_id"].astype(str))

    mode = "full scrape" if bootstrap_full_scrape else "incremental update"
    print(f"  mode: {mode} | existing comments: {len(known_comment_ids)} | "
          f"posts already processed: {len(processed_post_ids)}")

    comments_df = existing_comments_df
    pending_comment_frames: list[pd.DataFrame] = []
    pending_checked_rows: list[dict] = []
    posts_checked = 0
    posts_with_new_comments = 0
    new_comments_total = 0
    failed_posts = 0

    def flush_progress() -> None:
        """
        Persist comments before checked-posts, so a crash in between only causes a
        harmless re-fetch. The reverse order would mark a post done whose comments
        were never saved, losing them permanently.
        """
        nonlocal comments_df, checked_posts_df, pending_comment_frames, pending_checked_rows

        if pending_comment_frames:
            frames = [df for df in ([comments_df] + pending_comment_frames) if not df.empty]
            comments_df = pd.concat(frames, ignore_index=True)
            comments_df["id"] = comments_df["id"].astype(str)
            comments_df = comments_df.drop_duplicates(subset=["id"])
            comments_df["subreddit"] = (
                comments_df["subreddit"].fillna(subreddit).astype(str).str.lower()
            )
            comments_df.to_csv(raw_comments_file, index=False, encoding="utf-8")
            pending_comment_frames = []

        if pending_checked_rows:
            run_df = pd.DataFrame(pending_checked_rows, columns=CHECKED_COLUMNS)
            merged = pd.concat([checked_posts_df, run_df], ignore_index=True)
            checked_posts_df = merged.drop_duplicates(subset=["post_id"], keep="last")
            checked_posts_df.to_csv(checked_posts_file, index=False, encoding="utf-8")
            pending_checked_rows = []

    try:
        if len(known_comment_ids) >= MAX_COMMENT_NODES:
            print(f"  already at the {MAX_COMMENT_NODES}-comment cap "
                  f"({len(known_comment_ids)}). Skipping fetch.")
            return

        for post_id, title in iter_unprocessed_posts(subreddit, processed_post_ids):
            new_comments, ok = fetch_comments_for_post(subreddit, post_id, known_comment_ids)
            had_new = not new_comments.empty

            posts_checked += 1
            if not ok:
                failed_posts += 1
                time.sleep(INTER_CALL_DELAY)
                continue

            processed_post_ids.add(post_id)
            pending_checked_rows.append(
                {
                    "post_id": post_id,
                    "title": title,
                    "had_new_comments": bool(had_new),
                    "checked_at_utc": int(time.time()),
                }
            )

            if had_new:
                posts_with_new_comments += 1
                new_comments_total += len(new_comments)
                known_comment_ids.update(new_comments["id"].astype(str))
                pending_comment_frames.append(new_comments)
                print(f"  [{len(known_comment_ids):>5}/{MAX_COMMENT_NODES}] "
                      f"+{len(new_comments):>3} from {post_id} | {title[:60]}")

            if posts_checked % FLUSH_EVERY_N_POSTS == 0:
                flush_progress()

            if len(known_comment_ids) >= MAX_COMMENT_NODES:
                print(f"  reached comment cap of {MAX_COMMENT_NODES}. Stopping this subreddit.")
                break

            time.sleep(INTER_CALL_DELAY)
        else:
            print(f"  discovery exhausted for r/{subreddit} "
                  f"({len(known_comment_ids)} comments, below the {MAX_COMMENT_NODES} cap)")

    finally:
        flush_progress()

        row = {
            "subreddit": subreddit,
            "comments": int(len(comments_df)),
            "new_this_run": int(new_comments_total),
            "posts_checked": int(posts_checked),
            "posts_with_comments": int(posts_with_new_comments),
            "failed": int(failed_posts),
            "reply_edges": 0,
        }
        if failed_posts:
            print(f"  {failed_posts} post(s) failed and stayed unchecked; re-run to retry them.")

        if not comments_df.empty:
            nodes_df, edges_df, metadata = build_gnn_tables(comments_df, subreddit)
            metadata.update(
                {
                    "new_comments_added": int(new_comments_total),
                    "checked_posts_this_run": int(posts_checked),
                    "posts_with_new_comments_this_run": int(posts_with_new_comments),
                    "checked_posts_total": int(len(checked_posts_df)),
                    "processed_post_count": int(
                        comments_df["link_id"].dropna().astype(str).nunique()
                    ) if "link_id" in comments_df.columns else 0,
                }
            )

            nodes_df.to_csv(nodes_file, index=False, encoding="utf-8")
            edges_df.to_csv(edges_file, index=False, encoding="utf-8")
            meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            save_graphml(nodes_df, edges_df, graphml_file)

            row["reply_edges"] = metadata["edge_relations"].get("reply_to", 0)
            print(f"  wrote {metadata['num_nodes']} nodes / {metadata['num_edges']} edges "
                  f"({metadata['num_comment_nodes']} comments) → data/{subreddit}_gnn_*")

            if row["reply_edges"] == 0 and metadata["num_comment_nodes"] > 0:
                print("  WARNING: zero reply_to edges. RedScrapsLib may be returning only\n"
                      "           top-level comments, or ParentID in an unexpected format.\n"
                      "           Run probe_redscraps.py to inspect the raw fields.")
        else:
            print(f"  no comments collected for r/{subreddit}; nothing written.")

        summary.append(row)


# ─── Entry point ─────────────────────────────────────────────────────────────
def main() -> None:
    # When piped (e.g. into Tee-Object), Windows Python writes cp1252, which cannot
    # encode emoji or arrows in post titles. Degrade those to '?' instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_scraper()

    summary: list[dict] = []
    interrupted = False

    for index, subreddit in enumerate(SUBREDDITS, start=1):
        print(f"\n{'=' * 78}\n[{index}/{len(SUBREDDITS)}] r/{subreddit}\n{'=' * 78}")
        try:
            scrape_subreddit(subreddit, summary)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\nInterrupted during r/{subreddit}. Progress saved.")
            break
        except LibraryBugError as exc:
            print(f"\nABORTING THE ENTIRE RUN — RedScrapsLib is returning unusable data:\n  {exc}")
            raise SystemExit(2)
        except Exception as exc:
            print(f"r/{subreddit} failed: {exc}")

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    header = (f"{'subreddit':<14}{'comments':>10}{'new':>8}{'posts':>8}"
              f"{'with cmts':>11}{'failed':>8}{'replies':>9}")
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['subreddit']:<14}{row['comments']:>10}{row['new_this_run']:>8}"
              f"{row['posts_checked']:>8}{row['posts_with_comments']:>11}"
              f"{row['failed']:>8}{row['reply_edges']:>9}")

    total_comments = sum(row["comments"] for row in summary)
    print("-" * len(header))
    print(f"{'total':<14}{total_comments:>10}")

    try:
        print(f"\nRedScrapsLib session stats: {rs.get_stats()}")
    except Exception:
        pass

    if interrupted:
        print("\nRun was interrupted — re-run this script to resume from where it stopped.")
    else:
        remaining = [row["subreddit"] for row in summary if row["comments"] < MAX_COMMENT_NODES]
        if remaining:
            print(f"\nBelow the {MAX_COMMENT_NODES}-comment cap: {', '.join(remaining)}")
            print("Re-running will pick up newly posted threads; the listing ceiling "
                  "(~1000 posts per sort) limits how much further these can grow.")


if __name__ == "__main__":
    main()
