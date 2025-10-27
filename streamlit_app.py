import os, json, time
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PR Clustering", layout="wide")
st.title("🔎 PR Clustering")

# ---- Data source selector
mode = st.sidebar.radio("Data source", ["Upload file", "Fetch from GitHub"])

# ---- Common controls
threshold = st.sidebar.slider("Clustering threshold", 0.70, 0.98, 0.92, 0.01)
model_name = st.sidebar.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")

def build_text_block(row: dict) -> str:
    """Make a single text string from common PR fields."""
    title = row.get("title") or ""
    body = row.get("body") or ""
    commits = row.get("commits_messages") or row.get("commits") or ""
    if isinstance(commits, list): commits = "\n".join(commits)
    files = row.get("files") or ""
    if isinstance(files, list):
        # accept list of dicts (from GitHub API) or list of strings
        files = " ".join([f.get("filename","") if isinstance(f, dict) else str(f) for f in files])
    return f"TITLE: {title}\nBODY: {body}\nCOMMITS:\n{commits}\nFILES: {files}".strip()

@st.cache_resource(show_spinner=True)
def load_model(name): return SentenceTransformer(name)

def cluster_by_threshold(emb: np.ndarray, thr: float):
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(emb); np.fill_diagonal(sim, 1.0)
    n = sim.shape[0]
    parent = list(range(n)); rank = [0]*n
    def find(x):
        if parent[x]!=x: parent[x]=find(parent[x]); return parent[x]
        return x
    def union(a,b):
        ra,rb = find(a),find(b)
        if ra==rb: return
        if rank[ra]<rank[rb]: parent[ra]=rb
        elif rank[ra]>rank[rb]: parent[rb]=ra
        else: parent[rb]=ra; rank[ra]+=1
    for i in range(n):
        js = np.where(sim[i, i+1:]>=thr)[0] + (i+1)
        for j in js: union(i,j)
    roots, next_id = {}, 0
    cids = np.full(n, -1)
    for i in range(n):
        r = find(i)
        if r not in roots: roots[r] = next_id; next_id += 1
        cids[i] = roots[r]
    return cids

# ------------------ MODE A: UPLOAD ------------------
if mode == "Upload file":
    up = st.file_uploader("📂 Upload .csv or .json of PRs", type=["csv","json"])
    if up is not None:
        if up.name.endswith(".csv"):
            df = pd.read_csv(up)
        else:
            df = pd.read_json(up)

        st.write("Preview:", df.head())

        # Accept either a ready-made 'text' column OR build one from typical PR fields
        if "text" not in df.columns:
            st.info("No 'text' column found. Building one from title/body/commits/files if present.")
            records = df.to_dict(orient="records")
            texts = [build_text_block(r) for r in records]
        else:
            texts = df["text"].astype(str).tolist()

        if st.button("🚀 Embed + Cluster"):
            model = load_model(model_name)
            emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            cids = cluster_by_threshold(emb, threshold)
            out = df.copy()
            out["cluster_id"] = cids

            st.subheader("Cluster sizes (top 20)")
            sizes = out.groupby("cluster_id").size().sort_values(ascending=False)
            st.write(sizes.head(20))

            st.subheader("Sample (first 50 rows)")
            show_cols = [c for c in ["number","title","labels","url","text","cluster_id"] if c in out.columns or c=="cluster_id"]
            st.dataframe(out[show_cols].head(50) if show_cols else out.head(50))

            st.download_button(
                "💾 Download clustered CSV",
                out.to_csv(index=False).encode("utf-8"),
                "clustered_results.csv",
                "text/csv",
            )

# ------------------ MODE B: FETCH ------------------
else:
    # Minimal fetch UI (optional; keep if you want to fetch directly from GitHub)
    GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
    owner = st.sidebar.text_input("Owner", "wesnoth")
    repo = st.sidebar.text_input("Repo", "wesnoth")
    target = st.sidebar.slider("Merged PRs to fetch", 50, 500, 200, 50)

    import requests
    def gh_get(url, token, params=None):
        h = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(url, headers=h, params=params, timeout=60); r.raise_for_status()
        return r.json()

    if st.button("⬇️ Fetch PRs"):
        if not GITHUB_TOKEN:
            st.error("Set GITHUB_TOKEN in .streamlit/secrets.toml or Streamlit Cloud Secrets.")
        else:
            # fetch closed & merged PRs (simplified)
            items, page = [], 1
            while len(items) < target:
                batch = gh_get(f"https://api.github.com/repos/{owner}/{repo}/pulls",
                               GITHUB_TOKEN, {"state":"closed","per_page":100,"page":page})
                if not batch: break
                merged = [pr for pr in batch if pr.get("merged_at")]
                items.extend(merged)
                page += 1
                if len(batch) < 100: break
            items = items[:target]

            # minimal record + text build
            rows = []
            for pr in items:
                rows.append({
                    "number": pr["number"],
                    "title": pr.get("title"),
                    "body": pr.get("body"),
                    "labels": ",".join([x["name"] for x in pr.get("labels",[])]),
                    "url": pr.get("html_url"),
                    "text": build_text_block({"title": pr.get("title"), "body": pr.get("body"),
                                              "commits_messages":"", "files":""})
                })
            df = pd.DataFrame(rows)
            st.success(f"Fetched {len(df)} merged PRs.")
            st.dataframe(df.head())

            if st.button("🚀 Embed + Cluster (fetched)"):
                model = load_model(model_name)
                emb = model.encode(df["text"].astype(str).tolist(), convert_to_numpy=True, normalize_embeddings=True)
                cids = cluster_by_threshold(emb, threshold)
                df["cluster_id"] = cids
                sizes = df.groupby("cluster_id").size().sort_values(ascending=False)
                st.write("Top clusters:", sizes.head(20))
                st.dataframe(df.head(50))
