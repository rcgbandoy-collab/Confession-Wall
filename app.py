"""
Confession Wall — Graduating Student Farewell Messages
CS 315 - Application Development and Emerging Technologies | Activity 3

A GenAI-powered app where graduating students leave anonymous (or signed)
farewell messages addressed to a teacher, official, or location (e.g. canteen,
library). GenAI analyzes each message for sentiment, theme, and suggestions,
and an insights dashboard summarizes everything for the school.

UI: paper-note wall, free-text recipient, note-color picker, floating
purple iOS-style AI chat MODAL (not a sidebar) with a blurred backdrop.
The chat now reasons over STRUCTURED note records (author/message/color/
timestamp/id) instead of one text blob, so "latest" questions are answered
from real timestamps and authors are never confused with each other.

NOTE: the modal/backdrop trick below relies on Streamlit's
`st.container(key=...)` feature (Streamlit 1.31+), which puts a stable
CSS class (`st-key-<key>`) on the container so we can position it with
`position: fixed`. If you're on an older Streamlit, upgrade it.
"""

import html
import json
import os
import re
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
    "note_color",  # NEW (optional): older rows without it still work
]

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

TEMPORAL_KEYWORDS = [
    "latest", "newest", "most recent", "just posted", "just now",
    "last note", "last message", "last post", "recently posted",
    "who posted last", "who posted most recently", "newest post",
]

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


# --------------------------------------------------------------------------
# CHATBOT: structured record formatting + temporal ("latest") grounding
# --------------------------------------------------------------------------
def _sorted_by_time(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    d = df.copy()
    d["_dt"] = pd.to_datetime(d["timestamp"], errors="coerce")
    return d.sort_values("_dt", ascending=ascending)


def _format_notes_block(df: pd.DataFrame, limit: int = 60) -> str:
    """Each note as its OWN structured record — never a merged text blob, so
    the AI can't confuse one author's words/color/date with another's."""
    d = _sorted_by_time(df, ascending=False).head(limit)
    blocks = []
    for _, r in d.iterrows():
        posted = r["_dt"].strftime("%Y-%m-%d %I:%M %p") if pd.notna(r["_dt"]) else str(r.get("timestamp", ""))
        blocks.append(
            f'NOTE #{r.get("id", "?")}\n'
            f'Author: {r.get("sender_name") or "Anonymous"}\n'
            f'Recipient: {r.get("target_name", "")}\n'
            f'Color: {r.get("note_color") or "White"}\n'
            f'Posted: {posted}\n'
            f'Message: "{r.get("message", "")}"'
        )
    return "\n\n".join(blocks)


def _wants_latest(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in TEMPORAL_KEYWORDS)


def _latest_note_ground_truth(df: pd.DataFrame) -> str:
    """Computed in Python (not guessed by the LLM) so 'latest' questions are
    always answered from the real timestamp, never retrieval order."""
    d = _sorted_by_time(df, ascending=False)
    if d.empty:
        return "GROUND TRUTH: there are no notes on the wall yet."
    r = d.iloc[0]
    posted = r["_dt"].strftime("%Y-%m-%d %I:%M %p") if pd.notna(r["_dt"]) else str(r.get("timestamp", ""))
    return (
        f'GROUND TRUTH (computed by timestamp, trust this over anything else): the most '
        f'recently posted note right now is NOTE #{r.get("id", "?")}, by '
        f'{r.get("sender_name") or "Anonymous"}, posted {posted}, addressed to '
        f'"{r.get("target_name", "")}": "{r.get("message", "")}"'
    )


def chatbot_answer(question: str, df: pd.DataFrame, history: list | None = None) -> str:
    """A friendly assistant that reasons over the wall's STRUCTURED notes
    (author + message + color + timestamp + id kept together per note) and
    can also chat about anything else using its own knowledge."""
    notes_block = _format_notes_block(df)[:6000]
    latest_hint = _latest_note_ground_truth(df) if _wants_latest(question) else ""
    convo = ""
    if history:
        for turn in history[-6:]:
            role = "Student" if turn["role"] == "user" else "You"
            convo += f"{role}: {turn['content']}\n"

    prompt = f"""
You are a precise, friendly assistant embedded in a school "Confession Wall" app.
You are ONLY the assistant — never a student, never a wall author, never a wall
recipient. You can chat about anything using your own knowledge, but for
questions about the wall you MUST rely only on the structured notes below.

HARD ANTI-HALLUCINATION RULES — these override everything else:
- Every name you use MUST come from an "Author:" or "Recipient:" field in the
  NOTES block below, or from something the Student themselves typed in this
  chat. NEVER introduce, invent, or mention any other name. If no notes are
  relevant to the question, say plainly that there's nothing about that on the
  wall — do not make up a person to talk about instead.
- "Author" = who WROTE that note. "Recipient" = who that note is ADDRESSED TO.
  These are different people. Never say a Recipient "said" or "wrote" or
  "asked" something — only an Author can be quoted or described as saying
  something, and only for their own note.
- Never address the Student as if they were a wall Author or Recipient, and
  never claim the Student asked something they did not actually type. Quote
  the Student's own words only when they actually said them.
- If you're unsure who wrote or is referenced in a note, say so plainly
  instead of guessing — accuracy matters more than sounding confident.

OTHER RULES:
- Every note below is its OWN separate record. Never merge two notes together,
  and never attribute one author's words to a different author.
- For questions about "latest / newest / most recent / last / just posted",
  use ONLY the actual "Posted" timestamps (or the GROUND TRUTH line below if
  present) — never guess from which note appears first, similarity, or order.
- When discussing a specific note, name its actual Author from the record
  (never a placeholder or invented name, and never "the person"/"someone").
- Answer directly. Do NOT start with filler like "I understand...", "That's a
  straightforward message, isn't it?", or "Let me analyze this...".
- Keep the explanation proportional to the message — don't overanalyze a
  simple "hi", and give real detail for a message that has more to it.
- If asked about someone's texting/writing style, base it only on patterns you
  can actually see across THEIR OWN messages below — don't invent traits.
- Wrap the key point, names, or dates worth emphasizing in **double
  asterisks** — sparingly, not every word.
- Stay on the note/author the student is currently asking about; switch only
  when they clearly change topic.
- Give exactly ONE answer in your own voice as the assistant. Never write it
  as a back-and-forth or quote a conversation that didn't happen.

{latest_hint}

ALL NOTES CURRENTLY ON THE WALL (newest first). If this list is empty, there
is nothing posted yet — say so instead of inventing a note:
{notes_block if notes_block.strip() else "(no notes have been posted yet)"}

Conversation so far:
{convo}
Student: {question}

Reply in 2-4 sentences unless the question clearly needs more.
"""
    try:
        return _chat(prompt, temperature=0.2)
    except Exception as e:
        return f"Sorry, I couldn't process that: {e}"


def _guess_note_accent(answer: str, df: pd.DataFrame):
    """If the AI's reply names one author, tint that AI bubble with that
    author's most recent note color (subtle, per-message — not the whole
    chatbot theme, which stays grey by default)."""
    names = [n for n in df["sender_name"].dropna().unique() if n and n.lower() != "anonymous"]
    names.sort(key=len, reverse=True)  # longer names first avoids partial-name collisions
    low = answer.lower()
    for name in names:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", low):
            sub = _sorted_by_time(df[df["sender_name"] == name], ascending=False)
            if sub.empty:
                continue
            color = sub.iloc[0].get("note_color", "")
            if color in NOTE_COLORS:
                return NOTE_COLORS[color][2]  # the more saturated "tape" hex
    return None


def _bold(escaped_text: str) -> str:
    """Turn **bold** markers (from the AI's own reply) into <strong> tags.
    Safe to run AFTER html.escape — no raw user HTML can reach this point."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped_text)


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
            padding: 3px 12px; border-radius: 999px; font-size: 12px; font-family: 'Quicksand', sans-serif; }
.note-from { font-family: 'Caveat', cursive; font-size: 21px; color: var(--sub); margin-top: 10px; }
.note-date { font-family: 'Quicksand', sans-serif; font-size: 11px; color: var(--sub); margin-top: 4px; }

.single { max-width: 420px; margin: 24px auto; }

/* ---------- Floating purple AI chat ----------
   NOTE: positioning is done in JS (see below), not CSS-only, because
   Streamlit's container "key" CSS classes are only stable on newer
   versions. These rules only cover things that DON'T depend on that. */
@keyframes cwOpen {
    from { opacity: 0; transform: translateY(10px) scale(.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.cw-header-title { font-family: 'Quicksand', sans-serif; font-weight: 700; font-size: 15px; color: #fff; padding-top: 10px; }
.cw-header-sub { font-family: 'Quicksand', sans-serif; font-size: 12px; color: rgba(255,255,255,.85); margin-top: 2px; padding-bottom: 10px; }

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
    box-shadow: 0 1px 2px rgba(0,0,0,.08);
}
.cw-bubble.ai { background: #e5e7eb; color: #1f2937; border-bottom-left-radius: 4px; }
.cw-bubble.user { background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
.cw-bubble strong { text-decoration: underline; text-underline-offset: 2px; }

.cw-typing {
    display: inline-flex; gap: 4px; padding: 12px 14px;
    background: #e5e7eb; border-radius: 16px; border-bottom-left-radius: 4px;
}
.cw-typing span {
    width: 6px; height: 6px; border-radius: 50%; background: #6b7280;
    animation: cwBlink 1.2s infinite ease-in-out;
}
.cw-typing span:nth-child(2) { animation-delay: .2s; }
.cw-typing span:nth-child(3) { animation-delay: .4s; }
@keyframes cwBlink { 0%, 80%, 100% { opacity: .3; } 40% { opacity: 1; } }

</style>
""", unsafe_allow_html=True)


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
        f'{tag_html}'
        f'<div class="note-from">— {_clean(row["sender_name"])}</div>'
        f'<div class="note-date">{date}</div>'
        f'</div>'
    )


def render_card(row):
    """Show a single note (used for the preview after posting)."""
    st.markdown(f'<div class="single">{_note_html(row)}</div>', unsafe_allow_html=True)


def render_wall(df: pd.DataFrame):
    """Show many notes as a responsive paper wall."""
    notes = "".join(_note_html(r) for _, r in df.iterrows())
    st.markdown(f'<div class="wall">{notes}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# APP LAYOUT
# --------------------------------------------------------------------------
st.title("🎓 Confession Wall")
st.caption("Say what's on your mind about teachers, rooms, staff, or fellow students.")

# ---- Floating purple AI chat: MODAL + full-screen blurred backdrop ----
# NOTE: instead of relying on Streamlit's container "key" CSS classes
# (only stable on newer Streamlit versions, and were silently failing —
# that's why the button was invisible before), everything below is
# positioned with plain JavaScript that locates elements by their stable
# data-testid attributes / marker <div id> anchors. This works on old and
# new Streamlit alike.
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_pending" not in st.session_state:
    st.session_state.chat_pending = None

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

_CHAT_MODAL_JS = r"""
(function () {
    var doc = window.parent.document;
    function blockFor(id) {
        var a = doc.getElementById(id);
        return a ? a.closest('[data-testid="stVerticalBlock"]') : null;
    }
    function findButtonByText(root, text) {
        if (!root) return null;
        var btns = root.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            if (btns[i].textContent.trim() === text) return btns[i];
        }
        return null;
    }
    function styleModal() {
        var overlay = blockFor('cw-overlay-anchor');
        var modal = blockFor('cw-modal-anchor');
        var header = blockFor('cw-header-anchor');
        var inputbar = blockFor('cw-inputbar-anchor');
        if (!overlay || !modal || overlay.dataset.cwStyled) return;
        overlay.dataset.cwStyled = '1';

        overlay.style.position = 'fixed';
        overlay.style.inset = '0';
        overlay.style.zIndex = '999998';
        overlay.style.background = 'rgba(20,10,30,.45)';
        overlay.style.backdropFilter = 'blur(6px)';
        overlay.style.webkitBackdropFilter = 'blur(6px)';
        overlay.style.display = 'flex';
        overlay.style.alignItems = 'center';
        overlay.style.justifyContent = 'center';
        overlay.style.padding = '20px';

        modal.style.background = 'transparent';
        modal.style.borderRadius = '0';
        modal.style.width = '460px';
        modal.style.maxWidth = '92vw';
        modal.style.maxHeight = '82vh';
        modal.style.boxShadow = 'none';
        modal.style.display = 'flex';
        modal.style.flexDirection = 'column';
        modal.style.overflow = 'hidden';
        modal.style.animation = 'cwOpen .18s ease-out';

        if (header) {
            header.style.background = 'transparent';
            header.style.padding = '6px 4px 6px 18px';
            var newBtn = findButtonByText(header, '\ud83d\uddd1\ufe0f');
            var closeBtn = findButtonByText(header, '\u2715');
            [newBtn, closeBtn].forEach(function (b) {
                if (!b) return;
                b.style.background = 'transparent';
                b.style.border = 'none';
                b.style.boxShadow = 'none';
                b.style.color = '#fff';
                b.style.fontSize = '15px';
                b.style.width = '30px';
                b.style.height = '30px';
                b.style.marginTop = '2px';
            });
        }

        if (inputbar) {
            inputbar.style.borderTop = 'none';
            inputbar.style.padding = '10px 12px';
            inputbar.style.background = 'transparent';
            var inp = inputbar.querySelector('input[type="text"]');
            if (inp) {
                inp.style.borderRadius = '999px';
                inp.style.border = 'none';
                inp.style.padding = '8px 14px';
                inp.style.background = 'rgba(255,255,255,.92)';
                inp.style.color = '#111';
            }
            var sendBtn = findButtonByText(inputbar, '\u27a4');
            if (sendBtn) {
                sendBtn.style.background = '#2563eb';
                sendBtn.style.color = '#fff';
                sendBtn.style.border = 'none';
                sendBtn.style.borderRadius = '50%';
                sendBtn.style.width = '36px';
                sendBtn.style.height = '36px';
                sendBtn.style.padding = '0';
            }
        }
    }
    function autoScroll() {
        var d = doc.getElementById('cw-body');
        if (d) d.scrollTop = d.scrollHeight;
    }
    styleModal(); autoScroll();
    setTimeout(function () { styleModal(); autoScroll(); }, 150);
    setTimeout(function () { styleModal(); autoScroll(); }, 400);
})();
"""

if not st.session_state.chat_open:
    # CLOSED: only the small floating purple "chat head" is visible (bottom-right, draggable).
    if st.button("💬", key="chat_toggle"):
        st.session_state.chat_open = True
        st.rerun()
    components.html(f"<script>{_CHAT_HEAD_JS}</script>", height=0)
else:
    # OPEN: full-screen blurred/dimmed backdrop with a sharp floating card on top.
    with st.container():
        st.markdown('<div id="cw-overlay-anchor"></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div id="cw-modal-anchor"></div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div id="cw-header-anchor"></div>', unsafe_allow_html=True)
                hcol1, hcol2, hcol3 = st.columns([5, 1, 1])
                with hcol1:
                    st.markdown(
                        '<div class="cw-header-title">AI Assistant</div>'
                        '<div class="cw-header-sub">Ask anything about the messages posted here.</div>',
                        unsafe_allow_html=True,
                    )
                with hcol2:
                    if st.button("🗑️", key="chat_new", help="New chat (clears this conversation only — wall posts stay)"):
                        st.session_state.chat_history = []
                        st.session_state.chat_pending = None
                        st.rerun()
                with hcol3:
                    if st.button("✕", key="chat_close"):
                        st.session_state.chat_open = False
                        st.rerun()

            # message area (clean initial state — no pre-filled/leftover content)
            _wall_df = load_data()
            body_html = '<div class="cw-body" id="cw-body">'
            if not st.session_state.chat_history:
                body_html += (
                    '<div class="cw-row ai"><div>'
                    '<div class="cw-label">AI</div>'
                    '<div class="cw-bubble ai">Ask anything about the messages posted here.</div>'
                    '</div></div>'
                )
            for turn in st.session_state.chat_history:
                is_ai = turn["role"] != "user"
                side = "ai" if is_ai else "user"
                label = "AI" if is_ai else "You"
                content = _bold(html.escape(turn["content"])) if is_ai else html.escape(turn["content"])
                style = f' style="background:{turn["accent"]};"' if is_ai and turn.get("accent") else ""
                body_html += (
                    f'<div class="cw-row {side}"><div>'
                    f'<div class="cw-label">{label}</div>'
                    f'<div class="cw-bubble {side}"{style}>{content}</div>'
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

            # if a question was just sent, generate the reply now (the dots
            # above show for this render), then rerun with the real answer.
            if st.session_state.chat_pending:
                question = st.session_state.chat_pending
                answer = chatbot_answer(question, _wall_df, st.session_state.chat_history)
                accent = _guess_note_accent(answer, _wall_df)
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "accent": accent})
                st.session_state.chat_pending = None
                st.rerun()

            with st.container():
                st.markdown('<div id="cw-inputbar-anchor"></div>', unsafe_allow_html=True)
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

    components.html(f"<script>{_CHAT_MODAL_JS}</script>", height=0)

tab1, tab2, tab3 = st.tabs(["✏️ Leave a Message", "📝 Browse Wall", "📊 Insights"])

# ---- TAB 1: Submit a message ----
with tab1:
    st.subheader("Write your message")
    with st.form("new_message_form", clear_on_submit=True):
        target_name = st.text_input(
            "Recipient",
            placeholder="🔍 Search recipient... (e.g. Sir John, Room 204, Canteen staff)",
        )
        message = st.text_area("Message", height=150, placeholder="Write what's on your mind...")
        sender_name = st.text_input("From (optional — leave blank to stay Anonymous)")
        note_color = st.radio(
            "Choose your note color",
            COLOR_NAMES,
            format_func=lambda n: f"{NOTE_COLORS[n][0]} {n}",
            horizontal=True,
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
with tab2:
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
        render_wall(filtered.sort_values("id", ascending=False))

# ---- TAB 3: Insights dashboard ----
with tab3:
    st.subheader("Insights for the School")
    df = load_data()
    analyzed = df[df["sentiment"].notna() & (df["sentiment"] != "")]

    if analyzed.empty:
        st.info("No analyzed messages yet. Post a message in the first tab to see insights.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sentiment_counts = analyzed["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            fig = px.pie(sentiment_counts, names="sentiment", values="count", title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)

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

        st.markdown("### 💡 Suggestions Extracted for the School")
        suggestions = analyzed[analyzed["suggestion"].notna() & (analyzed["suggestion"] != "")]
        if suggestions.empty:
            st.info("No suggestions extracted yet.")
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

        st.info("💬 Tap the chat icon (bottom-right, any tab) to ask questions about the wall.")
