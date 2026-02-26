"""
human_ui/app.py
---------------
Streamlit pairwise ranking UI for the text-to-image benchmark.

Run from the project root:
    streamlit run human_ui/app.py

Features
--------
- Browse prompts by category, navigate with prev/next or select from list
- Shows Gemini (left) vs ChatGPT (right) images side-by-side for a given sample
- Optional blind mode: hides model labels and randomizes left/right order
- Displays VLM faithfulness score per image (if results/faithfulness_scores.csv exists)
- Voting: Gemini better | ChatGPT better | Tie | Both fail  (+ optional notes)
- Auto-advances to next unranked prompt after a vote
- Sidebar shows session progress and a live win-rate bar chart
- Saves every vote to results/human_rankings.csv (append mode, idempotent headers)
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image
import streamlit as st

# ---------------------------------------------------------------------------
# Paths  (all relative to project root; run from there)
# ---------------------------------------------------------------------------

PROMPTS_FILE = Path("prompts/core40.jsonl")
RANKINGS_CSV = Path("results/human_rankings.csv")
FAITH_CSV = Path("results/faithfulness_scores.csv")
QUALITY_CSV = Path("results/quality_scores.csv")

PROVIDERS = ("gemini", "chatgpt")
PROVIDER_LABELS = {"gemini": "Gemini (Imagen)", "chatgpt": "ChatGPT (GPT Image 1 Mini)"}

RANKINGS_FIELDS = [
    "timestamp", "run_dir", "prompt_id", "category", "sample",
    "winner",       # "gemini" | "chatgpt" | "tie" | "neither"
    "left_model",   # which model was shown on the left (for blind mode)
    "right_model",
    "blind",        # "1" if blind mode was active
    "notes",
    "annotator",
]

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@st.cache_data
def load_prompts() -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    if not PROMPTS_FILE.exists():
        st.error(f"Prompts file not found: {PROMPTS_FILE}")
        return prompts
    with PROMPTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


@st.cache_data
def load_faith_df() -> Optional[pd.DataFrame]:
    if FAITH_CSV.exists():
        return pd.read_csv(FAITH_CSV)
    return None


@st.cache_data
def load_quality_df() -> Optional[pd.DataFrame]:
    if QUALITY_CSV.exists():
        return pd.read_csv(QUALITY_CSV)
    return None


# ---------------------------------------------------------------------------
# Rankings helpers
# ---------------------------------------------------------------------------


def load_rankings() -> pd.DataFrame:
    if not RANKINGS_CSV.exists():
        return pd.DataFrame(columns=RANKINGS_FIELDS)
    return pd.read_csv(RANKINGS_CSV)


def save_vote(
    run_dir: str,
    prompt_id: str,
    category: str,
    sample: int,
    winner: str,
    left_model: str,
    right_model: str,
    blind: bool,
    notes: str,
    annotator: str,
) -> None:
    RANKINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "prompt_id": prompt_id,
        "category": category,
        "sample": sample,
        "winner": winner,
        "left_model": left_model,
        "right_model": right_model,
        "blind": int(blind),
        "notes": notes,
        "annotator": annotator,
    }
    if RANKINGS_CSV.exists():
        df = pd.read_csv(RANKINGS_CSV)
        mask = (df["prompt_id"] == prompt_id) & (df["sample"].astype(str) == str(sample))
        if mask.any():
            df.loc[mask, list(new_row.keys())] = list(new_row.values())
            df.to_csv(RANKINGS_CSV, index=False)
            return
    else:
        df = pd.DataFrame(columns=RANKINGS_FIELDS)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(RANKINGS_CSV, index=False)


def done_key(rankings: pd.DataFrame, pid: str, sample: int) -> bool:
    if rankings.empty:
        return False
    mask = (rankings["prompt_id"] == pid) & (rankings["sample"].astype(str) == str(sample))
    return bool(mask.any())


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def find_image(run_dir: str, provider: str, prompt_id: str, sample: int) -> Optional[Path]:
    p = Path(run_dir) / "images" / provider / f"{prompt_id}__s{sample}.png"
    return p if p.exists() else None


def score_label(df: Optional[pd.DataFrame], provider: str, pid: str, sample: int) -> str:
    if df is None:
        return ""
    row = df[
        (df["provider"] == provider)
        & (df["prompt_id"] == pid)
        & (df["sample"] == sample)
    ]
    if row.empty:
        return ""
    r = row.iloc[0]
    score = r.get("faithfulness_score", None)
    yes = r.get("checks_yes", "?")
    total = r.get("checks_total", "?")
    if score is not None:
        return f"Faithfulness {score:.0%}  ({yes}/{total} checks)"
    return ""


def quality_label(df: Optional[pd.DataFrame], provider: str, pid: str, sample: int) -> str:
    if df is None:
        return ""
    row = df[
        (df["provider"] == provider)
        & (df["prompt_id"] == pid)
        & (df["sample"] == sample)
    ]
    if row.empty:
        return ""
    r = row.iloc[0]
    q = r.get("quality_score", None)
    if q is not None:
        return f"Quality {q:.0%}"
    return ""


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Image Benchmark — Pairwise Ranking",
    page_icon="🖼️",
    layout="wide",
)

prompts = load_prompts()
if not prompts:
    st.stop()

prompt_by_id: Dict[str, Dict[str, Any]] = {p["prompt_id"]: p for p in prompts}
categories = sorted({p["category"] for p in prompts})

faith_df = load_faith_df()
quality_df = load_quality_df()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    run_dir = st.text_input("Run directory", value="runs/2026-02-12_core40_k3_1024")
    annotator = st.text_input("Your name", value="anonymous")
    sample_k = st.number_input("Sample #", min_value=1, max_value=3, value=1, step=1)

    blind_mode = st.toggle(
        "Blind mode",
        value=False,
        help="Hide model names and randomize left/right order to reduce bias",
    )

    st.divider()
    cat_filter = st.multiselect("Categories", options=categories, default=categories)
    filtered = [p for p in prompts if p["category"] in cat_filter]
    filtered_ids = [p["prompt_id"] for p in filtered]

    st.divider()

    # Progress
    rankings = load_rankings()
    ranked_in_filter = sum(
        1 for pid in filtered_ids if done_key(rankings, pid, sample_k)
    )
    remaining_ids = [pid for pid in filtered_ids if not done_key(rankings, pid, sample_k)]

    st.metric("Total prompts", len(filtered_ids))
    st.metric("Ranked (sample %d)" % sample_k, ranked_in_filter)
    st.metric("Remaining", len(remaining_ids))
    st.progress(ranked_in_filter / max(len(filtered_ids), 1))

    if remaining_ids and st.button("→ Jump to next unranked", use_container_width=True):
        st.session_state["current_pid"] = remaining_ids[0]

    st.divider()

    # Win-rate summary
    st.subheader("Results so far")
    if not rankings.empty:
        counts = rankings["winner"].value_counts().rename_axis("winner").reset_index(name="votes")
        st.bar_chart(counts.set_index("winner")["votes"])
    else:
        st.caption("No votes recorded yet.")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "current_pid" not in st.session_state or st.session_state["current_pid"] not in filtered_ids:
    st.session_state["current_pid"] = filtered_ids[0] if filtered_ids else None

# Blind-mode assignment persists per (pid, sample) so the order doesn't flip
# between reruns while the user is looking at the same pair.
blind_key = f"blind_{st.session_state['current_pid']}_{sample_k}"
if blind_key not in st.session_state:
    order = list(PROVIDERS)
    random.shuffle(order)
    st.session_state[blind_key] = order  # [left_model, right_model]

# ---------------------------------------------------------------------------
# Prompt selector
# ---------------------------------------------------------------------------

if not filtered_ids:
    st.warning("No prompts match the selected categories.")
    st.stop()

pid = st.session_state["current_pid"]
item = prompt_by_id[pid]

# Reset blind order when prompt changes
blind_key = f"blind_{pid}_{sample_k}"
if blind_key not in st.session_state:
    order = list(PROVIDERS)
    random.shuffle(order)
    st.session_state[blind_key] = order

left_model, right_model = st.session_state[blind_key]

# ---------------------------------------------------------------------------
# Prompt info
# ---------------------------------------------------------------------------

cols = st.columns([2, 1])
with cols[0]:
    st.markdown(f"**Prompt:** {item['prompt']}")
    if item.get("checks"):
        st.markdown("**Checks:**")
        for c in item["checks"]:
            st.markdown(f"  - {c}")
with cols[1]:
    st.markdown(f"**Category:** `{item['category']}`")
    st.markdown(f"**Prompt ID:** `{pid}`")
    st.markdown(f"**Sample:** {sample_k}")
    already = done_key(rankings, pid, sample_k)
    if already:
        row = rankings[
            (rankings["prompt_id"] == pid)
            & (rankings["sample"].astype(str) == str(sample_k))
        ].iloc[-1]
        st.success(f"Already voted: **{row['winner']}** wins")

# ---------------------------------------------------------------------------
# Image columns
# ---------------------------------------------------------------------------

left_label = "Model A" if blind_mode else PROVIDER_LABELS[left_model]
right_label = "Model B" if blind_mode else PROVIDER_LABELS[right_model]

col_l, col_r = st.columns(2)

with col_l:
    st.subheader(left_label)
    left_path = find_image(run_dir, left_model, pid, sample_k)
    if left_path:
        st.image(str(left_path), use_container_width=True)
        sl = score_label(faith_df, left_model, pid, sample_k)
        ql = quality_label(quality_df, left_model, pid, sample_k)
        if sl:
            st.caption(sl)
        if ql:
            st.caption(ql)
    else:
        st.warning(f"Image not found:\n`{run_dir}/images/{left_model}/{pid}__s{sample_k}.png`")

with col_r:
    st.subheader(right_label)
    right_path = find_image(run_dir, right_model, pid, sample_k)
    if right_path:
        st.image(str(right_path), use_container_width=True)
        sl = score_label(faith_df, right_model, pid, sample_k)
        ql = quality_label(quality_df, right_model, pid, sample_k)
        if sl:
            st.caption(sl)
        if ql:
            st.caption(ql)
    else:
        st.warning(f"Image not found:\n`{run_dir}/images/{right_model}/{pid}__s{sample_k}.png`")

# ---------------------------------------------------------------------------
# Voting UI
# ---------------------------------------------------------------------------

st.divider()
notes = st.text_input("Notes (optional)", key=f"notes_{pid}_{sample_k}", placeholder="e.g. left has better color, right has extra object...")

btn_cols = st.columns(4)

def cast_vote(winner_model: str) -> None:
    """winner_model is "gemini", "chatgpt", "tie", or "neither"."""
    save_vote(
        run_dir=run_dir,
        prompt_id=pid,
        category=item["category"],
        sample=sample_k,
        winner=winner_model,
        left_model=left_model,
        right_model=right_model,
        blind=blind_mode,
        notes=notes,
        annotator=annotator,
    )
    # Refresh rankings cache and advance
    load_faith_df.clear()
    load_quality_df.clear()
    # Auto-advance to next unranked
    fresh_rankings = load_rankings()
    next_unranked = [
        p for p in filtered_ids
        if not done_key(fresh_rankings, p, sample_k)
    ]
    if next_unranked:
        st.session_state["current_pid"] = next_unranked[0]
    st.rerun()


# Button labels depend on blind mode
left_btn_label  = f"👈 {left_label} is better"
right_btn_label = f"{right_label} is better 👉"

with btn_cols[0]:
    if st.button(left_btn_label, use_container_width=True, type="primary"):
        cast_vote(left_model)

with btn_cols[1]:
    if st.button(right_btn_label, use_container_width=True, type="primary"):
        cast_vote(right_model)

with btn_cols[2]:
    if st.button("🤝 Tie", use_container_width=True):
        cast_vote("tie")

with btn_cols[3]:
    if st.button("❌ Both fail", use_container_width=True):
        cast_vote("neither")

# ---------------------------------------------------------------------------
# Prev / Next navigation
# ---------------------------------------------------------------------------

nav_cols = st.columns([1, 8, 1])
cur_idx = filtered_ids.index(pid)

with nav_cols[0]:
    if cur_idx > 0 and st.button("← Prev", use_container_width=True):
        st.session_state["current_pid"] = filtered_ids[cur_idx - 1]
        st.rerun()

with nav_cols[2]:
    if cur_idx < len(filtered_ids) - 1 and st.button("Next →", use_container_width=True):
        st.session_state["current_pid"] = filtered_ids[cur_idx + 1]
        st.rerun()

# ---------------------------------------------------------------------------
# Rankings table (collapsible)
# ---------------------------------------------------------------------------

if not rankings.empty:
    with st.expander("📊 All rankings", expanded=False):
        display_cols = ["timestamp", "prompt_id", "category", "sample", "winner", "annotator", "notes"]
        st.dataframe(
            rankings[display_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
