"""
Confession Wall — Graduating Student Farewell Messages
CS 315 - Application Development and Emerging Technologies | Activity 3

A GenAI-powered app where graduating students leave anonymous (or signed)
farewell messages addressed to a teacher, official, or location (e.g. canteen,
library). GenAI analyzes each message for sentiment, theme, and suggestions,
and an insights dashboard summarizes everything for the school.
"""

import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from huggingface_hub import InferenceClient

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DATA_PATH = "data/messages.csv"
COLUMNS = [
    "id", "timestamp", "target_type", "target_name", "message",
    "sender_name", "sentiment", "emoji_tag", "keywords", "suggestion", "views",
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


def chatbot_answer(question: str, df: pd.DataFrame) -> str:
    """Answer a question about the message dataset using GenAI."""
    sample = df[["target_type", "target_name", "message", "sentiment"]].to_dict(orient="records")
    prompt = f"""
You are a helpful assistant analyzing a dataset of graduating students' farewell
messages. Here is the data (as JSON records): {json.dumps(sample)[:6000]}

Answer this question about the data, briefly and clearly: "{question}"
"""
    try:
        return _chat(prompt, temperature=0.3)
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
# STYLING (Instagram-card look: white card, centered text, soft shadow)
# --------------------------------------------------------------------------
st.markdown("""
<style>
.confession-card {
    background: #ffffff;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    padding: 32px 28px;
    margin: 18px auto;
    max-width: 480px;
    text-align: center;
}
.confession-target {
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 10px;
}
.confession-message {
    font-size: 18px;
    line-height: 1.5;
    color: #222;
    margin: 14px 0;
}
.confession-tag {
    display: inline-block;
    background: #f4f0ff;
    color: #6b46c1;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 13px;
    margin-top: 8px;
}
.confession-sender {
    font-size: 12px;
    color: #aaa;
    margin-top: 14px;
}
</style>
""", unsafe_allow_html=True)


def render_card(row):
    keywords = row["keywords"]
    if isinstance(keywords, str) and keywords.startswith("["):
        try:
            keywords = ", ".join(json.loads(keywords.replace("'", '"')))
        except Exception:
            pass
    st.markdown(f"""
    <div class="confession-card">
        <div class="confession-target">To: {row['target_name']}</div>
        <div class="confession-message">"{row['message']}"</div>
        <div class="confession-tag">{row.get('emoji_tag', '')}</div>
        <div class="confession-sender">Message from: {row['sender_name']}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# APP LAYOUT
# --------------------------------------------------------------------------
st.title("🎓 Confession Wall")
st.caption("Leave a farewell message for a teacher, official, or place before you go.")

tab1, tab2, tab3 = st.tabs(["💌 Leave a Message", "📜 Browse Wall", "📊 Insights Dashboard"])

# ---- TAB 1: Submit a message ----
with tab1:
    st.subheader("Write your message")
    with st.form("new_message_form", clear_on_submit=True):
        target_name = st.selectbox(
            "Who is this message for?",
            ["Teacher", "Student", "Official", "Canteen", "Library", "Administration", "General"],
        )
        message = st.text_area("Message", height=150, placeholder="Write what's on your heart...")
        sender_name = st.text_input("From (optional — leave blank to stay Anonymous)")
        submitted = st.form_submit_button("Post to the Wall")

    if submitted:
        if not message.strip():
            st.warning("Please write your message.")
        else:
            with st.spinner("Analyzing your message..."):
                analysis = analyze_message(message, "Recipient", target_name)
            new_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "target_type": "Recipient",
                "target_name": target_name,
                "message": message.strip(),
                "sender_name": sender_name.strip() if sender_name.strip() else "Anonymous",
                "sentiment": analysis.get("sentiment", "Neutral"),
                "emoji_tag": analysis.get("emoji_tag", ""),
                "keywords": ", ".join(analysis.get("keywords", [])),
                "suggestion": analysis.get("suggestion", ""),
                "views": 0,
            }
            save_message(new_row)
            st.success("Your message has been posted to the wall. 🎓")
            render_card(new_row)

# ---- TAB 2: Browse the wall ----
with tab2:
    st.subheader("The Wall")
    df = load_data()

    name_filter = st.text_input("Search by recipient name")

    filtered = df.copy()
    if name_filter:
        filtered = filtered[filtered["target_name"].str.contains(name_filter, case=False, na=False)]

    if filtered.empty:
        st.info("No messages match your search yet.")
    else:
        for _, row in filtered.sort_values("id", ascending=False).iterrows():
            render_card(row)

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

        st.markdown("### 💬 Ask about the Wall")
        question = st.text_input("Ask a question about the messages (e.g. 'What do students say about the canteen?')")
        if st.button("Ask") and question.strip():
            with st.spinner("Thinking..."):
                answer = chatbot_answer(question, analyzed)
            st.markdown(f"**Answer:** {answer}")
