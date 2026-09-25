"""
Confession Wall — Graduating Student Farewell Messages
CS 315 - Application Development and Emerging Technologies | Activity 3

A GenAI-powered app where graduating students leave anonymous (or signed)
farewell messages addressed to a teacher, official, or location (e.g. canteen,
library). GenAI analyzes each message for sentiment, theme, and suggestions,
and an insights dashboard summarizes everything for the school.

UI: paper-note wall, free-text recipient, note-color picker, floating
purple iOS-style AI chat MODAL (not a sidebar) with a blurred backdrop.

NOTE: the modal/backdrop trick below relies on Streamlit's
`st.container(key=...)` feature (Streamlit 1.31+), which puts a stable
CSS class (`st-key-<key>`) on the container so we can position it with
`position: fixed`. If you're on an older Streamlit, upgrade it
(`pip install -U streamlit`) or the chat will fall back to plain inline
layout instead of a floating modal.
"""

import html
import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import InferenceClient

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DATA_PATH = "data/messages.csv"
COLUMNS = [
    "id", "timestamp", "target_type", "target_name", "message",
    "sender_name", "sentiment", "emoji_tag", "keywords", "suggestion", "views",
    "note_color",
    # Per-note "Confession Analyzer" fields (cached so we don't re-call the
    # API every time the same note is opened) — additive, older rows without
    # these just show as "not yet analyzed" until someone opens them.
    "meaning", "emotion", "secondary_emotion", "intensity", "analyzed_at",
]

# Controlled emotion vocabulary — used both by the per-note analyzer AND the
# Insights "Emotions on the Wall" chart, so they always speak the same language.
EMOTION_CATEGORIES = {
    "Happiness": "😊", "Sadness": "😢", "Love": "❤️", "Anger": "😡",
    "Loneliness": "😔", "Fear": "😨", "Gratitude": "😌", "Excitement": "🎉",
    "Confusion": "😕", "Heartbreak": "💔", "Neutral": "😐",
    "Support": "🤝", "Surprise": "😮",
}

# name -> (picker emoji, paper color, tape color, ink color, soft ink color)
NOTE_COLORS = {
    "White":  ("⚪", "#fffdf6", "#e2d9c6", "#3b2f38", "#7a6a75"),
    "Yellow": ("🟡", "#fff3b0", "#ffd54a", "#3b2f38", "#7a6a75"),
    "Orange": ("🟠", "#ffd9b0", "#ffa94d", "#3b2a1a", "#7a5a3a"),
    "Red":    ("🔴", "#ffc9c9", "#ff8787", "#3b1f1f", "#7a4a4a"),
    "Pink":   ("🩷", "#ffd6e3", "#ff9ebd", "#3b2f38", "#7a6a75"),
    "Purple": ("🟣", "#e4d7ff", "#b79cff", "#2f2a3b", "#6a6480"),
    "Blue":   ("🔵", "#cfe6ff", "#8ec5ff", "#1f2f3b", "#4a6072"),
    "Teal":   ("🩵", "#c6f0ea", "#7fd8cc", "#1f3b37", "#4a726c"),
    "Green":  ("🟢", "#d5f5df", "#8fdcaa", "#1f3b2a", "#4a725a"),
    "Gray":   ("🩶", "#e3e5e8", "#aeb4bb", "#2b2f33", "#666c72"),
    "Brown":  ("🟤", "#c9a27a", "#8b5e3c", "#2d1f14", "#5a4030"),
    "Black":  ("⚫", "#2b2b2f", "#8a8a92", "#f5f5f7", "#b5b5bd"),
}
COLOR_NAMES = list(NOTE_COLORS.keys())
# Older rows with no saved color cycle through the light colors only
FALLBACK_COLORS = [c for c in COLOR_NAMES if c not in ("Black", "Brown")]
TILTS = [-2.0, 1.5, -1.0, 2.0, -1.5, 1.0]

# Any instruct-tuned chat model available on HF Inference Providers works.
# This one is free-tier friendly and good at following JSON instructions.
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

st.set_page_config(page_title="Confession Wall", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------
# GENAI CLIENT
# --------------------------------------------------------------------------
# Put your token in .streamlit/secrets.toml as:
# HF_TOKEN = "hf_..."
client = InferenceClient(token=st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", "")))


def _chat(prompt: str, temperature: float = 0.4) -> str:
    """Send a single-turn prompt to the Hugging Face chat model and return text."""
    response = client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def analyze_message(message: str, target_type: str, target_name: str) -> dict:
    """Send a message to the GenAI API and get back structured analysis."""
    prompt = f"""
You are analyzing a farewell message written by a graduating student.
The message is addressed to: {target_type} - {target_name}

Message: "{message}"

Return ONLY a valid JSON object (no markdown, no extra text, no explanation) with
these exact keys:
- "sentiment": one word, one of [Grateful, Nostalgic, Critical, Hopeful, Mixed, Neutral]
- "emoji_tag": a short emoji + 1-2 word label that fits the sentiment (e.g. "🌿 Grateful")
- "keywords": a list of up to 3 short theme keywords (e.g. ["teaching style", "patience"])
- "suggestion": if the message implies a suggestion or complaint for the school,
  summarize it in ONE short sentence. If there is no suggestion, return an empty string.
"""
    try:
        text = _chat(prompt, temperature=0.4)
        text = text.replace("```json", "").replace("```", "").strip()
        # Some instruct models add a stray sentence before/after the JSON —
        # grab just the {...} block to be safe.
        start, end = text.find("{"), text.rfind("}")
        text = text[start:end + 1]
        return json.loads(text)
    except Exception as e:
        st.error(f"GenAI analysis failed: {e}")
        return {
            "sentiment": "Neutral", "emoji_tag": "💭 Unanalyzed",
            "keywords": [], "suggestion": "",
        }


def chatbot_answer(question: str, df: pd.DataFrame, history: list | None = None) -> str:
    """A friendly general-purpose assistant that also knows about this wall's messages."""
    sample = df[["target_type", "target_name", "message", "sender_name", "sentiment"]].to_dict(orient="records")
    convo = ""
    if history:
        for turn in history[-6:]:  # keep last few turns for context
            role = "Student" if turn["role"] == "user" else "You"
            convo += f"{role}: {turn['content']}\n"
    prompt = f"""
You are a warm, friendly, knowledgeable chat assistant living inside a "Confession Wall"
app, where graduating students post farewell messages. You can chat about ANYTHING the
student asks — general knowledge, casual conversation, advice — using your own knowledge,
AND you also have access to EVERY message currently posted on this wall, shown below
(this list always reflects the latest state of the wall, updated every time someone posts
a new message). Use the wall's data whenever the question is about it; otherwise just
answer naturally like a smart, personable human would.

All messages currently on the wall (as JSON records):
{json.dumps(sample)[:6000]}

Conversation so far:
{convo}
Student: {question}

Reply naturally and conversationally, like a helpful human would in a chat — keep it
brief (2-4 sentences) unless more detail is clearly needed. If you're not fully sure
about a fast-changing real-world fact (like a current officeholder or recent news),
say so honestly instead of guessing.
"""
    try:
        return _chat(prompt, temperature=0.5)
    except Exception as e:
        return f"Sorry, I couldn't process that: {e}"


def analyze_emotion(message: str) -> dict:
    """Confession Analyzer: interpret ONE confession's meaning + emotion.

    Separate from analyze_message() (which handles the sentiment/keywords/
    suggestion shown when a note is first posted) — this powers the
    click-to-view 'THE MEANING' + 'EMOTION REVIEW' panel instead.
    """
    categories = ", ".join(EMOTION_CATEGORIES.keys())
    prompt = f"""
You are reading one short message posted on a farewell/message wall app. Messages
range from deep, heartfelt notes to simple casual greetings — read each one for
what it actually is, don't over-interpret. Interpret ONLY what is reasonably
supported by the text — do not invent events, relationships, names, or
psychological diagnoses that weren't stated. If the message is ambiguous, use
wording like "The message appears to express...". If the message is clearly
neutral or just a casual greeting, say so plainly and use "Neutral" as the emotion.

IMPORTANT: Never use the word "confession" in your response — just call it "this
message" or "this note". Many messages here are simple, casual, or friendly
(like "hello, ma'am!") and calling them a "confession" would be a mismatched,
overly dramatic label. Match your tone to the message: casual message → casual,
brief read; heartfelt message → warmer, more thoughtful read.

Message: "{message}"

Return ONLY a valid JSON object (no markdown, no extra text) with these exact keys:
- "meaning": ONE to TWO short sentences on what the confession appears to express.
  Go directly to the meaning — no filler like "I understand..." or "Let me analyze...".
- "emotion": the single best-fitting category from this exact list: [{categories}]
- "secondary_emotion": a second-best fitting category from the same list, or "" if none fits
- "intensity": one word, one of [Mild, Moderate, Strong]
"""
    try:
        text = _chat(prompt, temperature=0.3)
        text = text.replace("```json", "").replace("```", "").strip()
        start, end = text.find("{"), text.rfind("}")
        text = text[start:end + 1]
        result = json.loads(text)
        if result.get("emotion") not in EMOTION_CATEGORIES:
            result["emotion"] = "Neutral"
        if result.get("secondary_emotion") not in EMOTION_CATEGORIES:
            result["secondary_emotion"] = ""
        return result
    except Exception as e:
        return {
            "meaning": f"(Analysis unavailable: {e})",
            "emotion": "Neutral", "secondary_emotion": "", "intensity": "Mild",
        }


# --------------------------------------------------------------------------
# DATA HELPERS
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["message"] = df["message"].fillna("").astype(str).str.strip()
    df = df[df["message"] != ""]
    df["sender_name"] = df["sender_name"].fillna("Anonymous")
    df["sender_name"] = df["sender_name"].replace("", "Anonymous")
    return df


def save_message(row: dict):
    df = load_data()
    new_id = int(df["id"].max()) + 1 if len(df) else 1
    row["id"] = new_id
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)


def update_message_analysis(note_id: int, analysis: dict):
    """Persist a note's emotion analysis back into the CSV so it's cached —
    the same stored values feed both the note viewer AND the Insights chart."""
    df = load_data()
    mask = df["id"] == note_id
    if not mask.any():
        return
    df.loc[mask, "meaning"] = analysis.get("meaning", "")
    df.loc[mask, "emotion"] = analysis.get("emotion", "Neutral")
    df.loc[mask, "secondary_emotion"] = analysis.get("secondary_emotion", "")
    df.loc[mask, "intensity"] = analysis.get("intensity", "Mild")
    df.loc[mask, "analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_csv(DATA_PATH, index=False)


def get_or_analyze_emotion(row: pd.Series) -> dict:
    """Return the cached emotion analysis for a note, or run it once and save it.

    Per the spec: don't re-call the API every time the same note is opened —
    only analyze if there's no saved result yet.
    """
    if str(row.get("emotion", "")).strip():
        return {
            "meaning": row.get("meaning", ""),
            "emotion": row.get("emotion", "Neutral"),
            "secondary_emotion": row.get("secondary_emotion", ""),
            "intensity": row.get("intensity", "Mild"),
        }
    result = analyze_emotion(row["message"])
    update_message_analysis(int(row["id"]), result)
    return result


# --------------------------------------------------------------------------
# STYLING (paper-note wall + purple floating chat MODAL with blurred backdrop)
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&family=Quicksand:wght@400;600;700&display=swap');

/* ---------- Paper wall ---------- */
.wall { column-count: 3; column-gap: 26px; padding: 12px 6px 30px; }
@media (max-width: 900px) { .wall { column-count: 2; } }
@media (max-width: 600px) { .wall { column-count: 1; } }

.note {
    --tilt: 0deg; --tape: #ddd; --ink: #333; --sub: #777;
    position: relative;
    display: inline-block; width: 100%;
    break-inside: avoid; box-sizing: border-box;
    margin: 0 0 30px; padding: 30px 24px 18px;
    border-radius: 6px 6px 18px 6px;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.20);
    transform: rotate(var(--tilt));
    transition: transform .25s ease, box-shadow .25s ease;
    text-align: left; overflow-wrap: anywhere;
    background-image: repeating-linear-gradient(transparent 0 27px, rgba(128,128,128,0.16) 27px 28px);
}
.note:hover { transform: rotate(0deg) translateY(-6px) scale(1.02); box-shadow: 0 14px 28px rgba(0,0,0,.28); }
.note::before {
    content: ''; position: absolute; top: -12px; left: 50%;
    width: 84px; height: 24px; transform: translateX(-50%) rotate(-3deg);
    background: var(--tape); opacity: .8; border-radius: 3px;
}
.note-to { font-family: 'Quicksand', sans-serif; font-weight: 700; font-size: 13px;
           letter-spacing: 1px; text-transform: uppercase; color: var(--sub); }
.note-msg { font-family: 'Caveat', cursive; font-size: 25px; line-height: 28px; color: var(--ink); margin: 14px 0; }
.note-tag { display: inline-block; background: rgba(128,128,128,.22); color: var(--ink);
            padding: 3px 12px; border-radius: 999px; font-size: 12px; font-family: 'Quicksand', sans-serif;
            margin-top: 10px; }
.note-from { font-family: 'Caveat', cursive; font-size: 21px; color: var(--sub); margin-top: 10px; }
.note-date { font-family: 'Quicksand', sans-serif; font-size: 11px; color: var(--sub); margin-top: 4px; }

.single { max-width: 420px; margin: 24px auto; }

/* ---------- Floating purple AI chat: trigger button ---------- */
/* Position/size/drag is now fully controlled by _CHAT_HEAD_JS (inline styles),
   so no CSS positioning rules here — avoids fighting with the JS-driven drag. */

/* ---------- Full-screen dim + blur backdrop (the whole site sits behind this) ---------- */
.st-key-chat_overlay {
    position: fixed !important; inset: 0 !important; z-index: 9998 !important;
    background: rgba(20, 10, 30, .45) !important;
    backdrop-filter: blur(6px) !important; -webkit-backdrop-filter: blur(6px) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    padding: 20px !important;
}

/* ---------- The chat modal: NO white box — just the blurred backdrop, messages float on it ---------- */
.st-key-chat_modal {
    background: transparent !important; border-radius: 0 !important;
    width: 460px !important; max-width: 92vw !important; max-height: 82vh !important;
    box-shadow: none !important;
    display: flex !important; flex-direction: column !important; overflow: hidden !important;
    animation: cwOpen .18s ease-out;
}
@keyframes cwOpen {
    from { opacity: 0; transform: translateY(10px) scale(.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.st-key-chat_header {
    background: transparent !important;
    padding: 6px 4px 6px 18px !important;
}
.st-key-chat_header button {
    background: rgba(0,0,0,.25) !important; border: none !important; box-shadow: none !important;
    color: #fff !important; font-size: 16px !important; width: 34px !important; height: 34px !important;
    margin-top: 2px !important; border-radius: 50% !important;
}
.cw-header-title { font-family: 'Quicksand', sans-serif; font-weight: 700; font-size: 15px; color: #fff; padding-top: 10px;
                    text-shadow: 0 1px 4px rgba(0,0,0,.5); }
.cw-header-sub { font-family: 'Quicksand', sans-serif; font-size: 12px; color: rgba(255,255,255,.85); margin-top: 2px; padding-bottom: 10px;
                 text-shadow: 0 1px 4px rgba(0,0,0,.5); }

.cw-body { padding: 12px 14px 4px; max-height: 46vh; overflow-y: auto; scroll-behavior: smooth; }
.cw-row { display: flex; margin: 7px 0; animation: cwIn .2s ease; }
.cw-row.ai { justify-content: flex-start; }
.cw-row.user { justify-content: flex-end; }
@keyframes cwIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

.cw-label { font-family: 'Quicksand', sans-serif; font-size: 10px; color: rgba(255,255,255,.8); margin: 0 4px 2px; text-align: right; }
.cw-row.ai .cw-label { text-align: left; }

.cw-bubble {
    max-width: 78%; padding: 9px 14px; border-radius: 16px;
    font-family: 'Quicksand', sans-serif; font-size: 14px; line-height: 1.4;
    box-shadow: 0 2px 6px rgba(0,0,0,.25);
}
/* Default AI bubble = grey. When the AI mentions a note's author, its accent
   color (from that note) is applied inline, overriding this grey. */
.cw-bubble.ai { background: rgba(120,120,132,.92); color: #fff; border-bottom-left-radius: 4px; }
.cw-bubble.user { background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }

.cw-typing {
    display: inline-flex; gap: 4px; padding: 12px 14px;
    background: rgba(120,120,132,.92); border-radius: 16px; border-bottom-left-radius: 4px;
}
.cw-typing span {
    width: 6px; height: 6px; border-radius: 50%; background: #fff;
    animation: cwBlink 1.2s infinite ease-in-out;
}
.cw-typing span:nth-child(2) { animation-delay: .2s; }
.cw-typing span:nth-child(3) { animation-delay: .4s; }
@keyframes cwBlink { 0%, 80%, 100% { opacity: .3; } 40% { opacity: 1; } }

.st-key-chat_inputbar { border-top: none !important; padding: 10px 12px !important; background: transparent !important; }
.st-key-chat_inputbar input {
    border-radius: 999px !important; border: none !important;
    padding: 8px 14px !important; font-family: 'Quicksand', sans-serif !important;
    background: rgba(255,255,255,.92) !important; color: #111 !important;
}
.st-key-chat_inputbar button[kind="formSubmit"], .st-key-chat_inputbar button {
    background: #2563eb !important; color: #fff !important; border: none !important;
    border-radius: 50% !important; width: 36px !important; height: 36px !important;
    padding: 0 !important;
}

/* ---------- Clickable note cards on the wall ---------- */
/* A real (guaranteed-clickable) button, styled to blend into the bottom of
   the note itself — same rounded corners, no border, no gap — so it reads
   as part of the note rather than a separate UI control. Its background
   color is set per-note via an inline style (matches that note's paper color). */
[class*="st-key-note_wrap_"] { margin-bottom: 30px; }
[class*="st-key-note_wrap_"] .note { margin-bottom: 0; border-radius: 6px 6px 0 0; box-shadow: none; }
[class*="st-key-note_wrap_"] [data-testid="stButton"] { margin-top: 0; }
[class*="st-key-note_wrap_"] [data-testid="stButton"] button {
    width: 100% !important; border: none !important; border-radius: 0 0 12px 12px !important;
    box-shadow: 0 6px 16px rgba(0,0,0,.2) !important;
    font-family: 'Quicksand', sans-serif !important; font-size: 12px !important;
    padding: 8px !important; opacity: .85;
}
[class*="st-key-note_wrap_"] [data-testid="stButton"] button:hover { opacity: 1; }

/* ---------- Confession Analyzer modal: same blurred backdrop as the AI chat ---------- */
.st-key-note_overlay {
    position: fixed !important; inset: 0 !important; z-index: 9998 !important;
    background: rgba(20, 10, 30, .5) !important;
    backdrop-filter: blur(6px) !important; -webkit-backdrop-filter: blur(6px) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    padding: 20px !important; overflow-y: auto !important;
}
.st-key-note_modal {
    background: #1c1626 !important; border-radius: 20px !important;
    width: 760px !important; max-width: 94vw !important; max-height: 88vh !important;
    box-shadow: 0 25px 60px rgba(0,0,0,.5) !important;
    overflow: hidden !important; animation: cwOpen .18s ease-out;
    display: flex !important; flex-direction: column !important;
}
.st-key-note_close button {
    background: rgba(255,255,255,.1) !important; border: none !important;
    color: #fff !important; border-radius: 50% !important;
    width: 32px !important; height: 32px !important; padding: 0 !important;
}
.cw-note-grid { display: flex; gap: 0; overflow-y: auto; }
@media (max-width: 720px) { .cw-note-grid { flex-direction: column; } }
.cw-note-left { flex: 1; padding: 22px; display: flex; align-items: flex-start; justify-content: center; }
.cw-note-right { flex: 1; padding: 22px; border-left: 1px solid rgba(255,255,255,.08); }
@media (max-width: 720px) { .cw-note-right { border-left: none; border-top: 1px solid rgba(255,255,255,.08); } }

.cw-ai-badge { font-family: 'Quicksand', sans-serif; font-weight: 700; font-size: 13px;
               color: #b79cff; letter-spacing: .5px; margin-bottom: 14px; }
.cw-meaning-label, .cw-emotion-label { font-family: 'Quicksand', sans-serif; font-weight: 700;
    font-size: 12px; letter-spacing: 1px; text-transform: uppercase; color: #9a8fb0; margin: 14px 0 6px; }
.cw-meaning-text { font-family: 'Quicksand', sans-serif; font-size: 15px; line-height: 1.5; color: #f0eaf7; }
.cw-emotion-badge { font-family: 'Quicksand', sans-serif; font-size: 16px; font-weight: 700; color: #f0eaf7; }
.cw-emotion-secondary { font-family: 'Quicksand', sans-serif; font-size: 13px; color: #b0a4c4; margin-top: 4px; }
.cw-intensity { font-family: 'Quicksand', sans-serif; font-size: 12px; color: #9a8fb0; margin-top: 8px; }

/* ---------- Custom nav bar (replaces st.tabs so the selected tab persists) ---------- */
.cw-nav [data-testid="stRadio"] > div { gap: 22px; border-bottom: 1px solid rgba(255,255,255,.12); padding-bottom: 0; }
.cw-nav [data-testid="stRadio"] label { padding: 6px 2px 10px !important; }
.cw-nav [data-testid="stRadio"] label > div:first-child { display: none; }  /* hide the radio circle */
.cw-nav [data-testid="stRadio"] label p {
    font-family: 'Quicksand', sans-serif; font-weight: 700; font-size: 14px; color: #9a8fb0;
}
.cw-nav [data-testid="stRadio"] label[data-checked="true"] p,
.cw-nav [data-testid="stRadio"] label:has(input:checked) p { color: #ff9ebd; }
.cw-nav [data-testid="stRadio"] label:has(input:checked) { border-bottom: 2px solid #ff9ebd; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Draggable floating chat head — real-time follows the cursor while held,
# still a normal click when not dragged. Pure UI behavior; does not touch
# any AI/analysis logic above.
# --------------------------------------------------------------------------
_CHAT_HEAD_JS = r"""
(function () {
    var win = window.parent;
    var doc = win.document;
    var SIZE = 56;

    function findButtonByText(text) {
        var btns = doc.querySelectorAll('[data-testid="stButton"] button');
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].textContent.trim() === text) return btns[i];
        }
        return null;
    }

    function place(wrap, left, top) {
        var maxL = win.innerWidth - SIZE - 4;
        var maxT = win.innerHeight - SIZE - 4;
        wrap.style.left = Math.max(4, Math.min(left, maxL)) + 'px';
        wrap.style.top = Math.max(4, Math.min(top, maxT)) + 'px';
        wrap.style.right = 'auto';
        wrap.style.bottom = 'auto';
    }

    // Global listeners: bound only once, even though this script re-runs on every rerun
    function bindOnce() {
        if (win.__cwHeadBound) return;
        win.__cwHeadBound = true;

        doc.addEventListener('pointermove', function (e) {
            var s = win.__cwHeadState;
            if (!s || !s.down) return;
            var dx = e.clientX - s.x0, dy = e.clientY - s.y0;
            if (!s.moved && Math.abs(dx) + Math.abs(dy) > 4) s.moved = true;
            if (!s.moved) return;
            s.nextL = s.left0 + dx;
            s.nextT = s.top0 + dy;
            if (!s.raf) {
                s.raf = win.requestAnimationFrame(function () {
                    s.raf = null;
                    place(s.wrap, s.nextL, s.nextT);
                });
            }
        });

        doc.addEventListener('pointerup', function () {
            var s = win.__cwHeadState;
            if (!s || !s.down) return;
            s.down = false;
            clearTimeout(s.timer);
            s.wrap.style.transform = '';
            if (s.moved) {
                s.swallow = true;  // don't let this drop count as a click
                setTimeout(function () { s.swallow = false; }, 400);
                var r = s.wrap.getBoundingClientRect();
                try {
                    win.sessionStorage.setItem('cwHeadPos', JSON.stringify({ left: r.left, top: r.top }));
                } catch (err) {}
            }
        });

        // Capture-phase click filter: swallows the click only right after a drag
        doc.addEventListener('click', function (e) {
            var s = win.__cwHeadState;
            if (s && s.swallow) {
                s.swallow = false;
                e.stopPropagation();
                e.preventDefault();
            }
        }, true);
    }

    function onDown(e) {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        var wrap = e.currentTarget;
        var r = wrap.getBoundingClientRect();
        var s = {
            wrap: wrap, down: true, moved: false, swallow: false,
            x0: e.clientX, y0: e.clientY, left0: r.left, top0: r.top,
            timer: null, raf: null
        };
        // Long press (~180ms) = "lift" visual, so the user knows it's grabbed
        s.timer = setTimeout(function () {
            wrap.style.transform = 'scale(1.1)';
        }, 180);
        win.__cwHeadState = s;
    }

    function styleFloatingHead() {
        bindOnce();
        var btn = findButtonByText('\ud83d\udcac');  // 💬
        if (!btn) return;
        var wrap = btn.closest('[data-testid="stButton"]');
        if (!wrap || wrap.dataset.cwStyled) return;
        wrap.dataset.cwStyled = '1';

        wrap.style.position = 'fixed';
        wrap.style.zIndex = '999997';
        wrap.style.touchAction = 'none';
        wrap.style.transition = 'transform .15s ease';

        btn.style.background = 'linear-gradient(135deg, #7c3aed, #9333ea)';
        btn.style.color = '#fff';
        btn.style.border = 'none';
        btn.style.borderRadius = '50%';
        btn.style.width = SIZE + 'px';
        btn.style.height = SIZE + 'px';
        btn.style.padding = '0';
        btn.style.fontSize = '24px';
        btn.style.boxShadow = '0 6px 18px rgba(124,58,237,.5)';
        btn.style.cursor = 'grab';
        btn.style.touchAction = 'none';

        // Restore last position (survives reruns and open/close)
        var saved = null;
        try { saved = JSON.parse(win.sessionStorage.getItem('cwHeadPos')); } catch (e) {}
        if (saved && typeof saved.left === 'number') {
            place(wrap, saved.left, saved.top);
        } else {
            place(wrap, win.innerWidth - SIZE - 24, win.innerHeight - SIZE - 24);
        }

        wrap.addEventListener('pointerdown', onDown);
    }

    styleFloatingHead();
    setTimeout(styleFloatingHead, 200);
    setTimeout(styleFloatingHead, 600);
})();
"""


def _clean(v) -> str:
    """NaN-safe, HTML-safe text."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    return html.escape(str(v))


def _note_html(row) -> str:
    """Build the HTML of one paper note (color chosen by the sender)."""
    color = row.get("note_color", "")
    try:
        rid = int(row.get("id", 0))
    except Exception:
        rid = 0
    if color not in NOTE_COLORS:  # older rows: stable color from their id
        color = FALLBACK_COLORS[rid % len(FALLBACK_COLORS)]
    _, paper, tape, ink, sub = NOTE_COLORS[color]
    tilt = TILTS[rid % len(TILTS)]
    try:
        date = pd.to_datetime(row.get("timestamp")).strftime("%B %d, %Y")
    except Exception:
        date = ""
    msg = _clean(row["message"]).replace("\n", "<br>")
    tag = _clean(row.get("emoji_tag", ""))
    tag_html = f'<span class="note-tag">{tag}</span>' if tag else ""
    return (
        f'<div class="note" style="background-color:{paper};--tape:{tape};--ink:{ink};--sub:{sub};--tilt:{tilt}deg;">'
        f'<div class="note-to">To: {_clean(row["target_name"])}</div>'
        f'<div class="note-msg">{msg}</div>'
        f'<div class="note-from">— {_clean(row["sender_name"])}</div>'
        f'<div class="note-date">{date}</div>'
        f'{tag_html}'
        f'</div>'
    )


def render_card(row):
    """Show a single note (used for the preview after posting)."""
    st.markdown(f'<div class="single">{_note_html(row)}</div>', unsafe_allow_html=True)


def render_wall(df: pd.DataFrame):
    """Show many notes as a responsive paper wall."""
    notes = "".join(_note_html(r) for _, r in df.iterrows())
    st.markdown(f'<div class="wall">{notes}</div>', unsafe_allow_html=True)


def render_note_modal():
    """Confession Analyzer: focused modal for one note — original note on the
    left, AI 'meaning' + 'emotion review' on the right (stacks on mobile)."""
    df = load_data()
    match = df[df["id"] == st.session_state.selected_note_id]
    if match.empty:
        st.session_state.note_modal_open = False
        return
    row = match.iloc[0]

    with st.container(key="note_overlay"):
        with st.container(key="note_modal"):
            top1, top2 = st.columns([8, 1])
            with top1:
                st.markdown(
                    '<div style="padding:14px 18px 0; font-family:Quicksand,sans-serif; '
                    'font-weight:700; color:#b79cff; font-size:14px;">🔍 Confession Analyzer</div>',
                    unsafe_allow_html=True,
                )
            with top2:
                with st.container(key="note_close"):
                    if st.button("✕", key="note_close_btn"):
                        st.session_state.note_modal_open = False
                        st.rerun()

            st.markdown('<div class="cw-note-grid">', unsafe_allow_html=True)

            left, right = st.columns(2)
            with left:
                st.markdown(f'<div class="cw-note-left">{_note_html(row)}</div>', unsafe_allow_html=True)

            with right:
                already_analyzed = str(row.get("emotion", "")).strip() != ""
                if already_analyzed:
                    result = get_or_analyze_emotion(row)
                else:
                    with st.spinner("✨ AI is reading this confession..."):
                        result = get_or_analyze_emotion(row)

                # subtle accent from the note's own color
                color_name = row.get("note_color", "")
                if color_name not in NOTE_COLORS:
                    color_name = FALLBACK_COLORS[int(row["id"]) % len(FALLBACK_COLORS)]
                accent = NOTE_COLORS[color_name][2]

                emo = result.get("emotion", "Neutral")
                sec = result.get("secondary_emotion", "")
                emo_icon = EMOTION_CATEGORIES.get(emo, "😐")
                sec_icon = EMOTION_CATEGORIES.get(sec, "")

                right_html = f'<div class="cw-note-right" style="border-top:3px solid {accent};">'
                right_html += '<div class="cw-ai-badge">✨ AI Interpretation</div>'
                right_html += '<div class="cw-meaning-label">The Meaning</div>'
                right_html += f'<div class="cw-meaning-text">{_clean(result.get("meaning",""))}</div>'
                right_html += '<div class="cw-emotion-label">Emotion Review</div>'
                right_html += f'<div class="cw-emotion-badge">{emo_icon} {_clean(emo)}</div>'
                if sec:
                    right_html += f'<div class="cw-emotion-secondary">Secondary: {sec_icon} {_clean(sec)}</div>'
                right_html += f'<div class="cw-intensity">Intensity: {_clean(result.get("intensity","Mild"))}</div>'
                right_html += '</div>'
                st.markdown(right_html, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


def _guess_note_accent(answer: str, df: pd.DataFrame):
    """If the AI's reply mentions a sender/author from the wall, return that
    note's accent color (hex) so the AI chat bubble can match it. Returns
    None if no author is mentioned (bubble stays default grey)."""
    lower_answer = answer.lower()
    for _, row in df.iterrows():
        sender = str(row.get("sender_name", "")).strip()
        if not sender or sender.lower() == "anonymous":
            continue
        if sender.lower() in lower_answer:
            color = row.get("note_color", "")
            if color in NOTE_COLORS:
                return NOTE_COLORS[color][2]  # the vivid "tape" color as accent
    return None


# --------------------------------------------------------------------------
# APP LAYOUT
# --------------------------------------------------------------------------
st.title("🎓 Confession Wall")
st.caption("Say what's on your mind about teachers, rooms, staff, or fellow students.")

# ---- Floating purple AI chat: MODAL + full-screen blurred backdrop ----
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_pending" not in st.session_state:
    st.session_state.chat_pending = None
if "note_modal_open" not in st.session_state:
    st.session_state.note_modal_open = False
if "selected_note_id" not in st.session_state:
    st.session_state.selected_note_id = None

# Only one overlay active at a time: opening the Confession Analyzer closes the chat.
if st.session_state.note_modal_open:
    st.session_state.chat_open = False
    render_note_modal()
elif not st.session_state.chat_open:
    # CLOSED: only the small floating purple icon is visible.
    with st.container(key="chat_toggle_btn"):
        if st.button("💬", key="chat_toggle"):
            st.session_state.chat_open = True
            st.rerun()
    # Make that button draggable (long-press + drag = move; plain click = open).
    components.html(f"<script>{_CHAT_HEAD_JS}</script>", height=0)
else:
    # OPEN: full-screen blurred/dimmed backdrop with a sharp floating card on top.
    with st.container(key="chat_overlay"):
        with st.container(key="chat_modal"):
            with st.container(key="chat_header"):
                hcol1, hcol2 = st.columns([6, 1])
                with hcol1:
                    st.markdown(
                        '<div class="cw-header-title">AI Assistant</div>'
                        '<div class="cw-header-sub">Ask anything about the confessions posted on the wall.</div>',
                        unsafe_allow_html=True,
                    )
                with hcol2:
                    if st.button("✕", key="chat_close"):
                        st.session_state.chat_open = False
                        st.rerun()

            # message area (clean initial state — no pre-filled/leftover content)
            body_html = '<div class="cw-body" id="cw-body">'
            if not st.session_state.chat_history:
                body_html += (
                    '<div class="cw-row ai"><div>'
                    '<div class="cw-label">AI</div>'
                    '<div class="cw-bubble ai">Ask me anything about the confessions posted on the wall!</div>'
                    '</div></div>'
                )
            for turn in st.session_state.chat_history:
                is_ai = turn["role"] != "user"
                side = "ai" if is_ai else "user"
                label = "AI" if is_ai else "You"
                accent = turn.get("accent") if is_ai else None
                style_attr = f' style="background:{accent};color:#1a1a1a;"' if accent else ""
                body_html += (
                    f'<div class="cw-row {side}"><div>'
                    f'<div class="cw-label">{label}</div>'
                    f'<div class="cw-bubble {side}"{style_attr}>{html.escape(turn["content"])}</div>'
                    f'</div></div>'
                )
            if st.session_state.chat_pending:
                body_html += (
                    '<div class="cw-row ai"><div>'
                    '<div class="cw-label">AI</div>'
                    '<div class="cw-typing"><span></span><span></span><span></span></div>'
                    '</div></div>'
                )
            body_html += "</div>"
            st.markdown(body_html, unsafe_allow_html=True)

            # auto-follow the latest message — no manual scrolling needed
            components.html(
                """
                <script>
                  var d = window.parent.document.getElementById('cw-body');
                  if (d) { d.scrollTop = d.scrollHeight; }
                </script>
                """,
                height=0,
            )

            # if a question was just sent, generate the reply now (the dots
            # above show for this render), then rerun with the real answer.
            # NOTE: we pass the FULL wall (load_data()), not a sentiment-filtered
            # subset — otherwise the AI only "sees" messages that already went
            # through GenAI analysis and misses everything else on the wall.
            if st.session_state.chat_pending:
                question = st.session_state.chat_pending
                _full_df = load_data()
                answer = chatbot_answer(question, _full_df, st.session_state.chat_history)
                accent = _guess_note_accent(answer, _full_df)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "accent": accent}
                )
                st.session_state.chat_pending = None
                st.rerun()

            with st.container(key="chat_inputbar"):
                with st.form("chat_form", clear_on_submit=True):
                    colA, colB = st.columns([5, 1])
                    with colA:
                        user_q = st.text_input(
                            "msg", placeholder="Type a message...", label_visibility="collapsed"
                        )
                    with colB:
                        send = st.form_submit_button("➤")
                if send and user_q.strip():
                    st.session_state.chat_history.append({"role": "user", "content": user_q})
                    st.session_state.chat_pending = user_q
                    st.rerun()

# Custom nav (instead of st.tabs) so the current tab survives reruns —
# e.g. closing the Confession Analyzer modal keeps you on Browse Wall
# instead of snapping back to the first tab.
TAB_OPTIONS = ["✏️ Leave a Message", "📝 Browse Wall", "📊 Insights"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = TAB_OPTIONS[0]

st.markdown('<div class="cw-nav">', unsafe_allow_html=True)
st.session_state.active_tab = st.radio(
    "nav", TAB_OPTIONS,
    index=TAB_OPTIONS.index(st.session_state.active_tab),
    horizontal=True, label_visibility="collapsed", key="nav_radio",
)
st.markdown('</div>', unsafe_allow_html=True)

# ---- TAB 1: Submit a message ----
if st.session_state.active_tab == TAB_OPTIONS[0]:
    st.subheader("Write your message")
    with st.form("new_message_form", clear_on_submit=True):
        target_name = st.text_input(
            "Recipient",
            placeholder="🔍 Search recipient... (e.g. Sir John, Room 204, Canteen staff)",
        )
        message = st.text_area("Message", height=150, placeholder="Write what's on your mind...")
        sender_name = st.text_input("From (optional — leave blank to stay Anonymous)")
        with st.expander("🎨 Add Color"):
            note_color = st.radio(
                "Choose your note color",
                COLOR_NAMES,
                format_func=lambda n: f"{NOTE_COLORS[n][0]} {n}",
                horizontal=True,
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("Post to the Wall")

    if submitted:
        if not target_name.strip():
            st.warning("Please type who or what this message is for.")
        elif not message.strip():
            st.warning("Please write your message.")
        else:
            with st.spinner("Analyzing your message..."):
                analysis = analyze_message(message, "Recipient", target_name.strip())
            new_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "target_type": "Recipient",
                "target_name": target_name.strip(),
                "message": message.strip(),
                "sender_name": sender_name.strip() if sender_name.strip() else "Anonymous",
                "sentiment": analysis.get("sentiment", "Neutral"),
                "emoji_tag": analysis.get("emoji_tag", ""),
                "keywords": ", ".join(analysis.get("keywords", [])),
                "suggestion": analysis.get("suggestion", ""),
                "views": 0,
                "note_color": note_color,
            }
            save_message(new_row)
            st.success("Your message has been posted to the wall. 🎓")
            render_card(new_row)

# ---- TAB 2: Browse the wall ----
elif st.session_state.active_tab == TAB_OPTIONS[1]:
    st.subheader("The Wall")
    df = load_data()

    name_filter = st.text_input("Search by recipient name", placeholder="🔍 Search recipient...")

    filtered = df.copy()
    if name_filter.strip():
        filtered = filtered[
            filtered["target_name"].astype(str).str.contains(name_filter.strip(), case=False, na=False, regex=False)
        ]

    if filtered.empty:
        st.info("No messages match your search yet.")
    else:
        st.caption("Tap any note to see the AI's interpretation.")
        rows = filtered.sort_values("id", ascending=False).to_dict(orient="records")
        n_cols = 3
        cols = st.columns(n_cols)
        for i, note_row in enumerate(rows):
            with cols[i % n_cols]:
                with st.container(key=f"note_wrap_{note_row['id']}"):
                    st.markdown(_note_html(note_row), unsafe_allow_html=True)
                    if st.button("open", key=f"view_note_{note_row['id']}"):
                        st.session_state.selected_note_id = note_row["id"]
                        st.session_state.note_modal_open = True
                        st.rerun()

# ---- TAB 3: Insights dashboard ----
elif st.session_state.active_tab == TAB_OPTIONS[2]:
    st.subheader("🧠 Confession Wall Insights")
    st.caption("A quick look at what people are sharing on the wall.")
    df = load_data()
    analyzed = df[df["sentiment"].notna() & (df["sentiment"] != "")]
    emotion_analyzed = df[df["emotion"].notna() & (df["emotion"] != "")]

    if analyzed.empty and emotion_analyzed.empty:
        st.info("No analyzed messages yet. Post a message, or open one on the Browse Wall, to see insights.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if not analyzed.empty:
                sentiment_counts = analyzed["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["sentiment", "count"]
                fig = px.pie(sentiment_counts, names="sentiment", values="count", title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data yet.")

        with c2:
            all_keywords = analyzed["keywords"].dropna().str.split(", ").explode()
            all_keywords = all_keywords[all_keywords != ""]
            if not all_keywords.empty:
                kw_counts = all_keywords.value_counts().head(10).reset_index()
                kw_counts.columns = ["keyword", "count"]
                fig2 = px.bar(kw_counts, x="keyword", y="count", title="Most Common Themes")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Not enough keyword data yet.")

        # Emotions on the Wall — fed by the same per-note analysis shown in the
        # Confession Analyzer modal (Browse Wall → 🔍 View). Same data, same chart.
        st.markdown("### 😊 Emotions on the Wall")
        if emotion_analyzed.empty:
            st.info("No confessions analyzed yet — open a note on the Browse Wall and tap 🔍 View.")
        else:
            emo_counts = emotion_analyzed["emotion"].value_counts().reset_index()
            emo_counts.columns = ["emotion", "count"]
            emo_counts["label"] = emo_counts["emotion"].apply(
                lambda e: f"{EMOTION_CATEGORIES.get(e, '')} {e}"
            )
            fig3 = px.bar(emo_counts, x="label", y="count", title="Most Common Emotions (from analyzed confessions)")
            st.plotly_chart(fig3, use_container_width=True)
            top = emo_counts.iloc[0]
            st.caption(f"Most common emotion so far: **{top['label']}**")

        st.markdown("### 💡 Wall Insights")
        st.caption("AI-generated notes surfaced from what people are posting — not a school evaluation.")
        suggestions = analyzed[analyzed["suggestion"].notna() & (analyzed["suggestion"] != "")]
        if suggestions.empty:
            st.info("No wall insights extracted yet.")
        else:
            for _, row in suggestions.iterrows():
                st.markdown(f"- **[{row['target_name']}]** {row['suggestion']}")

        st.markdown("### 💾 Backup your data")
        st.caption("Download the current notes before editing the app code, so nothing gets lost on the next deploy.")
        st.download_button(
            "Download messages.csv",
            data=df.to_csv(index=False),
            file_name="messages_backup.csv",
            mime="text/csv",
        )

        st.info("💬 Tap the chat icon (top-right, any tab) to ask about the confessions on the wall.")
