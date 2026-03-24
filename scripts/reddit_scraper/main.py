import json
import time
from pathlib import Path
from urllib.parse import urlparse

import networkx as nx
import pandas as pd
import requests


# Be conservative with request pacing, especially for reddit.com fallback calls.
HOST_MIN_INTERVAL_SECONDS = {
    "www.reddit.com": 2.0,
    "reddit.com": 2.0,
    "api.pullpush.io": 0.5,
}
_LAST_REQUEST_TS_BY_HOST: dict[str, float] = {}


def throttle_request(url: str) -> None:
    host = urlparse(url).netloc.lower()
    min_interval = HOST_MIN_INTERVAL_SECONDS.get(host, 0.0)
    if min_interval <= 0:
        return

    now = time.time()
    last_ts = _LAST_REQUEST_TS_BY_HOST.get(host)
    if last_ts is not None:
        elapsed = now - last_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    _LAST_REQUEST_TS_BY_HOST[host] = time.time()


def fetch_json_with_retry(url: str, params: dict, retries: int = 20, timeout: int = 30) -> dict:
    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            throttle_request(url)
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            last_error = str(exc)
            # 4xx errors (except rate limit) are usually request-shape issues and should not be retried.
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                raise RuntimeError(f"Non-retriable HTTP {status_code}: {last_error}")

            retry_after_seconds = None
            if exc.response is not None:
                retry_after_header = exc.response.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        retry_after_seconds = max(1, int(float(retry_after_header)))
                    except ValueError:
                        retry_after_seconds = None

            wait_seconds = min(2 ** attempt, 20)
            if retry_after_seconds is not None:
                wait_seconds = max(wait_seconds, retry_after_seconds)

            print(f"Request attempt {attempt}/{retries} failed: {last_error}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
            wait_seconds = min(2 ** attempt, 20)
            print(f"Request attempt {attempt}/{retries} failed: {last_error}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


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
    existing_df = existing_df.drop_duplicates(subset=["id"])
    return existing_df


def load_checked_posts(checked_posts_file: Path) -> pd.DataFrame:
    if not checked_posts_file.exists():
        return pd.DataFrame(columns=["post_id", "title", "had_new_comments", "checked_at_utc"])

    try:
        checked_df = pd.read_csv(checked_posts_file)
    except Exception as exc:
        print(f"Failed to read checked posts file {checked_posts_file}: {exc}")
        return pd.DataFrame(columns=["post_id", "title", "had_new_comments", "checked_at_utc"])

    if checked_df.empty:
        return pd.DataFrame(columns=["post_id", "title", "had_new_comments", "checked_at_utc"])

    if "post_id" not in checked_df.columns:
        print("Checked posts file missing 'post_id'. Starting checked-posts tracking fresh.")
        return pd.DataFrame(columns=["post_id", "title", "had_new_comments", "checked_at_utc"])

    checked_df["post_id"] = checked_df["post_id"].astype(str)
    checked_df = checked_df.drop_duplicates(subset=["post_id"], keep="last")
    return checked_df


def append_checked_posts(
    checked_posts_file: Path,
    existing_checked_df: pd.DataFrame,
    run_checked_rows: list[dict],
) -> pd.DataFrame:
    if run_checked_rows:
        run_df = pd.DataFrame(run_checked_rows)
        run_df["post_id"] = run_df["post_id"].astype(str)
        merged_df = pd.concat([existing_checked_df, run_df], ignore_index=True)
    else:
        merged_df = existing_checked_df.copy()

    if merged_df.empty:
        merged_df = pd.DataFrame(columns=["post_id", "title", "had_new_comments", "checked_at_utc"])
    else:
        merged_df = merged_df.drop_duplicates(subset=["post_id"], keep="last")

    merged_df.to_csv(checked_posts_file, index=False, encoding="utf-8")
    return merged_df


def mark_post_checked(
    checked_posts_file: Path,
    checked_posts_df: pd.DataFrame,
    post_id: str,
    title: str,
    had_new_comments: bool,
) -> pd.DataFrame:
    row = {
        "post_id": str(post_id),
        "title": str(title or ""),
        "had_new_comments": bool(had_new_comments),
        "checked_at_utc": int(time.time()),
    }
    return append_checked_posts(
        checked_posts_file=checked_posts_file,
        existing_checked_df=checked_posts_df,
        run_checked_rows=[row],
    )


def iter_unprocessed_posts(subreddit: str, processed_post_ids: set[str], page_size: int = 100):
    url = "https://api.pullpush.io/reddit/search/submission/"
    params = {
        "subreddit": subreddit,
        "sort_type": "created_utc",
        "sort": "desc",
        "size": min(max(1, page_size), 100),
    }
    before_cursor: int | None = None

    while True:
        request_params = dict(params)
        if before_cursor is not None:
            request_params["before"] = before_cursor

        try:
            payload = fetch_json_with_retry(url, request_params)
        except RuntimeError as exc:
            print(f"Request failed while searching posts for r/{subreddit}: {exc}")
            return

        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            return

        min_created = None
        for item in data:
            post_id = str(item.get("id", ""))
            if not post_id:
                continue

            normalized_post_id = post_id if post_id.startswith("t3_") else f"t3_{post_id}"
            created = item.get("created_utc")
            if isinstance(created, (int, float)):
                created_int = int(created)
                min_created = created_int if min_created is None else min(min_created, created_int)

            if normalized_post_id in processed_post_ids:
                continue

            yield item

        if min_created is None:
            return
        before_cursor = min_created - 1


def fetch_comments_from_reddit_json(post_id: str, subreddit: str, known_comment_ids: set[str]) -> pd.DataFrame:
    base36_post_id = post_id.replace("t3_", "", 1) if post_id.startswith("t3_") else post_id
    url = f"https://www.reddit.com/comments/{base36_post_id}.json"
    headers = {"User-Agent": "mitigating-catastrophic-forgetting-scraper/1.0"}
    params = {"limit": 500, "sort": "new", "raw_json": 1}

    try:
        throttle_request(url)
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Reddit JSON fallback failed for post {post_id}: {exc}")
        return pd.DataFrame()

    if not isinstance(payload, list) or len(payload) < 2:
        return pd.DataFrame()

    comments_listing = payload[1].get("data", {}).get("children", []) if isinstance(payload[1], dict) else []
    rows: list[dict] = []
    local_seen: set[str] = set()

    def walk(children):
        if not isinstance(children, list):
            return

        for child in children:
            if not isinstance(child, dict) or child.get("kind") != "t1":
                continue

            data = child.get("data", {})
            comment_id = str(data.get("id", ""))
            if comment_id and comment_id not in known_comment_ids and comment_id not in local_seen:
                local_seen.add(comment_id)
                parent_id = str(data.get("parent_id", "") or "")
                rows.append(
                    {
                        "id": comment_id,
                        "author": data.get("author"),
                        "body": data.get("body"),
                        "score": data.get("score"),
                        "created_utc": data.get("created_utc"),
                        "subreddit": str(data.get("subreddit", subreddit)).lower(),
                        "link_id": f"t3_{base36_post_id}",
                        "parent_id": parent_id,
                    }
                )

            replies = data.get("replies")
            if isinstance(replies, dict):
                nested = replies.get("data", {}).get("children", [])
                walk(nested)

    walk(comments_listing)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        print(f"Fallback collected {len(frame)} comments from Reddit JSON for post t3_{base36_post_id}.")
    return frame


def fetch_comments_for_post(
    post_id: str,
    subreddit: str,
    known_comment_ids: set[str],
    expected_comment_count: int = 0,
    page_size: int = 100,
) -> pd.DataFrame:
    url = "https://api.pullpush.io/reddit/search/comment/"
    normalized_post_id = post_id if post_id.startswith("t3_") else f"t3_{post_id}"
    base36_post_id = normalized_post_id.replace("t3_", "", 1)
    base_params = {
        "sort_type": "created_utc",
        "sort": "desc",
        "size": min(max(1, page_size), 100),
    }

    # PullPush docs and behavior can vary between base36 and fullname formats.
    param_variants = [
        {**base_params, "link_id": base36_post_id},
        {**base_params, "link_id": normalized_post_id},
        {**base_params, "subreddit": subreddit, "link_id": base36_post_id},
        {**base_params, "subreddit": subreddit, "link_id": normalized_post_id},
    ]

    working_params = None
    for variant in param_variants:
        try:
            test_payload = fetch_json_with_retry(url, variant)
        except RuntimeError as exc:
            print(f"Skipping unsupported param variant {variant}: {exc}")
            continue

        if isinstance(test_payload.get("data", []), list):
            working_params = variant
            break

    if working_params is None:
        print(f"Could not find a valid query format for comments on post {normalized_post_id}.")
        return pd.DataFrame()

    print(f"Using comment query params: {working_params}")

    all_comments: list[dict] = []
    before_cursor: int | None = None
    local_seen: set[str] = set()

    while True:
        request_params = dict(working_params)
        if before_cursor is not None:
            request_params["before"] = before_cursor

        try:
            payload = fetch_json_with_retry(url, request_params)
        except RuntimeError as exc:
            print(f"Request failed for comments on post {normalized_post_id}: {exc}")
            break

        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            break

        page_new_rows = 0
        min_created = None
        for item in data:
            comment_id = str(item.get("id", ""))
            if not comment_id or comment_id in known_comment_ids or comment_id in local_seen:
                continue

            local_seen.add(comment_id)
            all_comments.append(item)
            page_new_rows += 1

            created = item.get("created_utc")
            if isinstance(created, (int, float)):
                created_int = int(created)
                min_created = created_int if min_created is None else min(min_created, created_int)

        print(f"Collected {len(all_comments)} new comments for post {normalized_post_id}...")

        if min_created is None:
            break

        if page_new_rows == 0:
            break

        before_cursor = min_created - 1

    frame = pd.DataFrame(all_comments)
    if frame.empty:
        if expected_comment_count > 0:
            print(
                f"PullPush returned 0 comments for {normalized_post_id} despite num_comments={expected_comment_count}. "
                "Trying Reddit JSON fallback..."
            )
            return fetch_comments_from_reddit_json(
                post_id=normalized_post_id,
                subreddit=subreddit,
                known_comment_ids=known_comment_ids,
            )
        return frame

    frame["subreddit"] = frame.get("subreddit", subreddit).fillna(subreddit).astype(str).str.lower()
    return frame


def build_gnn_tables(comments_df: pd.DataFrame, subreddit: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    comments_df = comments_df.copy()
    if "id" not in comments_df.columns:
        raise ValueError("API response does not contain comment 'id' field.")

    comments_df = comments_df.dropna(subset=["id", "subreddit"])
    comments_df = comments_df.drop_duplicates(subset=["id"])

    if "score" not in comments_df.columns and "ups" in comments_df.columns:
        comments_df = comments_df.rename(columns={"ups": "score"})

    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    node_id_by_key: dict[str, int] = {}

    def add_node(node_key: str, node_type: str, text: str = "", score=None, subreddit: str = "", label: int = -1) -> int:
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
                "subreddit": subreddit,
                "label": label,
            }
        )
        return node_id

    def add_edge(src_key: str, dst_key: str, relation: str):
        src_id = node_id_by_key[src_key]
        dst_id = node_id_by_key[dst_key]
        edge_rows.append({"src": src_id, "dst": dst_id, "relation": relation})

    for row in comments_df.itertuples(index=False):
        comment_id = str(getattr(row, "id", ""))
        subreddit = str(getattr(row, "subreddit", "")).lower()
        body = str(getattr(row, "body", "") or "")
        score = getattr(row, "score", None)

        comment_key = f"comment:{comment_id}"
        add_node(
            comment_key,
            node_type="comment",
            text=body,
            score=score,
            subreddit=subreddit,
            label=0,
        )

        author = str(getattr(row, "author", "") or "")
        if author and author.lower() not in {"[deleted]", "[removed]"}:
            author_key = f"user:{author.lower()}"
            add_node(author_key, node_type="user", text=author)
            add_edge(author_key, comment_key, relation="authored")

        link_id = str(getattr(row, "link_id", "") or "")
        if link_id:
            post_key = f"post:{link_id}"
            add_node(post_key, node_type="post", text=link_id)
            add_edge(comment_key, post_key, relation="on_post")

        parent_id = str(getattr(row, "parent_id", "") or "")
        if parent_id.startswith("t1_"):
            parent_comment_key = f"comment:{parent_id.replace('t1_', '', 1)}"
            if parent_comment_key in node_id_by_key:
                add_edge(comment_key, parent_comment_key, relation="reply_to")

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)
    metadata = {
        "subreddit": subreddit.lower(),
        "num_nodes": int(len(nodes_df)),
        "num_edges": int(len(edges_df)),
        "num_comment_nodes": int((nodes_df["node_type"] == "comment").sum()),
        "num_subreddits": 1,
        "label_map": {subreddit.lower(): 0},
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


def main() -> None:
    target_subreddit = "javascript"
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_comments_file = out_dir / "javascript_gnn_comments_raw.csv"
    nodes_file = out_dir / "javascript_gnn_nodes.csv"
    edges_file = out_dir / "javascript_gnn_edges.csv"
    meta_file = out_dir / "javascript_gnn_metadata.json"
    graphml_file = out_dir / "javascript_gnn.graphml"
    checked_posts_file = out_dir / "javascript_gnn_checked_posts.csv"

    existing_comments_df = load_existing_comments(raw_comments_file=raw_comments_file, subreddit=target_subreddit)
    bootstrap_full_scrape = existing_comments_df.empty

    if bootstrap_full_scrape:
        # If there is no stored dataset, start from scratch and do not reuse old checked-post state.
        checked_posts_df = pd.DataFrame(columns=["post_id", "title", "had_new_comments", "checked_at_utc"])
        if checked_posts_file.exists():
            checked_posts_file.unlink(missing_ok=True)
    else:
        checked_posts_df = load_checked_posts(checked_posts_file=checked_posts_file)

    known_comment_ids = set(existing_comments_df["id"].astype(str).tolist()) if not existing_comments_df.empty else set()
    processed_post_ids: set[str] = set()

    if not bootstrap_full_scrape and "link_id" in existing_comments_df.columns:
        processed_post_ids = {
            str(link_id)
            for link_id in existing_comments_df["link_id"].dropna().astype(str).tolist()
            if str(link_id)
        }
    if not bootstrap_full_scrape and not checked_posts_df.empty:
        processed_post_ids.update(checked_posts_df["post_id"].astype(str).tolist())

    print(f"Existing comments: {len(known_comment_ids)}")
    print(f"Already checked/processed posts: {len(processed_post_ids)}")
    if bootstrap_full_scrape:
        print("Mode: full scrape from start (no existing dataset found).")
    else:
        print("Mode: incremental update (existing dataset found).")

    new_comment_frames: list[pd.DataFrame] = []
    selected_posts_this_run = 0
    posts_with_new_comments = 0

    try:
        for post in iter_unprocessed_posts(subreddit=target_subreddit, processed_post_ids=processed_post_ids):
            selected_post_id = str(post.get("id", ""))
            selected_post_id = selected_post_id if selected_post_id.startswith("t3_") else f"t3_{selected_post_id}"
            selected_title = str(post.get("title", "") or "")
            post_num_comments = post.get("num_comments", 0)
            if not isinstance(post_num_comments, (int, float)):
                post_num_comments = 0
            selected_posts_this_run += 1
            print(f"Selected post: {selected_post_id} | {selected_title}")

            new_comments_df = fetch_comments_for_post(
                post_id=selected_post_id,
                subreddit=target_subreddit,
                known_comment_ids=known_comment_ids,
                expected_comment_count=int(post_num_comments),
            )

            had_new_comments = not new_comments_df.empty
            processed_post_ids.add(selected_post_id)
            checked_posts_df = mark_post_checked(
                checked_posts_file=checked_posts_file,
                checked_posts_df=checked_posts_df,
                post_id=selected_post_id,
                title=selected_title,
                had_new_comments=had_new_comments,
            )

            if not had_new_comments:
                if bootstrap_full_scrape:
                    print(f"No comments found for post {selected_post_id}. Moving to next post...")
                else:
                    print(f"No new comments found for post {selected_post_id}. Moving to next post...")
                continue

            posts_with_new_comments += 1
            known_comment_ids.update(new_comments_df["id"].astype(str).tolist())
            new_comment_frames.append(new_comments_df)

        print(f"No more unprocessed posts found in r/{target_subreddit}.")
    except KeyboardInterrupt:
        print("Interrupted by user. Checked post progress has been saved.")

    if new_comment_frames:
        all_new_comments_df = pd.concat(new_comment_frames, ignore_index=True)
    else:
        all_new_comments_df = pd.DataFrame()

    if all_new_comments_df.empty:
        if bootstrap_full_scrape:
            print("No comments were found across scanned posts.")
        else:
            print("No new comments were found across remaining posts.")
        print(f"Checked posts this run: {selected_posts_this_run}")
        print(f"Saved checked posts file: {checked_posts_file}")
        return

    comments_df = pd.concat([existing_comments_df, all_new_comments_df], ignore_index=True)
    comments_df["id"] = comments_df["id"].astype(str)
    comments_df = comments_df.drop_duplicates(subset=["id"])
    comments_df["subreddit"] = comments_df.get("subreddit", target_subreddit).fillna(target_subreddit).astype(str).str.lower()

    print(f"Checked posts this run: {selected_posts_this_run}")
    print(f"Posts with new comments this run: {posts_with_new_comments}")
    print(f"New comments added this run: {len(all_new_comments_df)}")
    print(f"Total comments after merge: {len(comments_df)}")

    nodes_df, edges_df, metadata = build_gnn_tables(comments_df, subreddit=target_subreddit)

    metadata["new_comments_added"] = int(len(all_new_comments_df))
    metadata["checked_posts_this_run"] = int(selected_posts_this_run)
    metadata["posts_with_new_comments_this_run"] = int(posts_with_new_comments)
    metadata["checked_posts_total"] = int(len(checked_posts_df))
    metadata["processed_post_count"] = int(comments_df["link_id"].dropna().astype(str).nunique()) if "link_id" in comments_df.columns else 0

    comments_df.to_csv(raw_comments_file, index=False, encoding="utf-8")
    nodes_df.to_csv(nodes_file, index=False, encoding="utf-8")
    edges_df.to_csv(edges_file, index=False, encoding="utf-8")
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_graphml(nodes_df, edges_df, graphml_file)

    print(f"Saved raw comments: {raw_comments_file}")
    print(f"Saved nodes: {nodes_file}")
    print(f"Saved edges: {edges_file}")
    print(f"Saved metadata: {meta_file}")
    print(f"Saved graphml: {graphml_file}")
    print(f"Saved checked posts: {checked_posts_file}")
    print("This graph contains nodes and edges only from r/javascript, incrementally updated.")


if __name__ == "__main__":
    main()