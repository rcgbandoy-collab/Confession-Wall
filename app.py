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

/* ---------- Floating purple AI chat: trigger button ---------- */
.st-key-chat_toggle_btn { position: fixed; top: 18px; right: 22px; z-index: 9997; }
.st-key-chat_toggle_btn button {
    background: linear-gradient(135deg, #7c3aed, #9333ea) !important;
    color: #fff !important; border: none !important; border-radius: 50% !important;
    width: 54px !important; height: 54px !important; padding: 0 !important;
    font-size: 22px !important; box-shadow: 0 6px 18px rgba(124, 58, 237, .45) !important;
    transition: transform .15s ease !important;
}
.st-key-chat_toggle_btn button:hover { transform: scale(1.06); }

/* ---------- Full-screen dim + blur backdrop (the whole site sits behind this) ---------- */
.st-key-chat_overlay {
    position: fixed !important; inset: 0 !important; z-index: 9998 !important;
    background: rgba(20, 10, 30, .45) !important;
    backdrop-filter: blur(6px) !important; -webkit-backdrop-filter: blur(6px) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    padding: 20px !important;
}

/* ---------- The chat card: sits above the blur, stays sharp ---------- */
.st-key-chat_modal {
    background: #fff !important; border-radius: 20px !important;
    width: 380px !important; max-width: 92vw !important; max-height: 82vh !important;
    box-shadow: 0 25px 60px rgba(0,0,0,.45) !important;
    display: flex !important; flex-direction: column !important; overflow: hidden !important;
    animation: cwOpen .18s ease-out;
}
@keyframes cwOpen {
    from { opacity: 0; transform: translateY(10px) scale(.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.st-key-chat_header {
    background: linear-gradient(135deg, #7c3aed, #9333ea) !important;
    padding: 6px 4px 6px 18px !important;
}
.st-key-chat_header button {
    background: transparent !important; border: none !important; box-shadow: none !important;
    color: #fff !important; font-size: 16px !important; width: 34px !important; height: 34px !important;
    margin-top: 2px !important;
}
.cw-header-title { font-family: 'Quicksand', sans-serif; font-weight: 700; font-size: 15px; color: #fff; padding-top: 10px; }
.cw-header-sub { font-family: 'Quicksand', sans-serif; font-size: 12px; color: rgba(255,255,255,.85); margin-top: 2px; padding-bottom: 10px; }

.cw-body { padding: 12px 14px 4px; max-height: 46vh; overflow-y: auto; scroll-behavior: smooth; }
.cw-row { display: flex; margin: 7px 0; animation: cwIn .2s ease; }
.cw-row.ai { justify-content: flex-start; }
.cw-row.user { justify-content: flex-end; }
@keyframes cwIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

.cw-label { font-family: 'Quicksand', sans-serif; font-size: 10px; color: #999; margin: 0 4px 2px; text-align: right; }
.cw-row.ai .cw-label { text-align: left; }

.cw-bubble {
    max-width: 78%; padding: 9px 14px; border-radius: 16px;
    font-family: 'Quicksand', sans-serif; font-size: 14px; line-height: 1.4;
    box-shadow: 0 1px 2px rgba(0,0,0,.08);
}
.cw-bubble.ai { background: #7c3aed; color: #fff; border-bottom-left-radius: 4px; }
.cw-bubble.user { background: #f1f0f5; color: #222; border-bottom-right-radius: 4px; }

.cw-typing {
    display: inline-flex; gap: 4px; padding: 12px 14px;
    background: #7c3aed; border-radius: 16px; border-bottom-left-radius: 4px;
}
.cw-typing span {
    width: 6px; height: 6px; border-radius: 50%; background: #fff;
    animation: cwBlink 1.2s infinite ease-in-out;
}
.cw-typing span:nth-child(2) { animation-delay: .2s; }
.cw-typing span:nth-child(3) { animation-delay: .4s; }
@keyframes cwBlink { 0%, 80%, 100% { opacity: .3; } 40% { opacity: 1; } }

.st-key-chat_inputbar { border-top: 1px solid #eee !important; padding: 10px 12px !important; background: #fff !important; }
.st-key-chat_inputbar input {
    border-radius: 999px !important; border: 1px solid #ddd !important;
    padding: 8px 14px !important; font-family: 'Quicksand', sans-serif !important;
}
.st-key-chat_inputbar button[kind="formSubmit"], .st-key-chat_inputbar button {
    background: #7c3aed !important; color: #fff !important; border: none !important;
    border-radius: 50% !important; width: 36px !important; height: 36px !important;
    padding: 0 !important;
}
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
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_pending" not in st.session_state:
    st.session_state.chat_pending = None

if not st.session_state.chat_open:
    # CLOSED: only the small floating purple icon is visible.
    with st.container(key="chat_toggle_btn"):
        if st.button("💬", key="chat_toggle"):
            st.session_state.chat_open = True
            st.rerun()
else:
    # OPEN: full-screen blurred/dimmed backdrop with a sharp floating card on top.
    with st.container(key="chat_overlay"):
        with st.container(key="chat_modal"):
            with st.container(key="chat_header"):
                hcol1, hcol2 = st.columns([6, 1])
                with hcol1:
                    st.markdown(
                        '<div class="cw-header-title">AI Assistant</div>'
                        '<div class="cw-header-sub">Ask anything about the messages posted here.</div>',
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
                    '<div class="cw-bubble ai">Ask me anything — about the wall, or anything else!</div>'
                    '</div></div>'
                )
            for turn in st.session_state.chat_history:
                is_ai = turn["role"] != "user"
                side = "ai" if is_ai else "user"
                label = "AI" if is_ai else "You"
                body_html += (
                    f'<div class="cw-row {side}"><div>'
                    f'<div class="cw-label">{label}</div>'
                    f'<div class="cw-bubble {side}">{html.escape(turn["content"])}</div>'
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
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
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

        st.info("💬 Tap the chat icon (top-right, any tab) to ask questions about the wall.")
