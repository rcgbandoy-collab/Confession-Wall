"""
Confession Wall — Graduating Student Farewell Messages
CS 315 - Application Development and Emerging Technologies | Activity 3

A GenAI-powered app where graduating students leave anonymous (or signed)
farewell messages addressed to a teacher, official, or location.

GenAI analyzes each message for:
- Sentiment
- Emoji tag
- Keywords / themes
- Suggestions

The app also provides:
- Paper-note wall
- Searchable recipients
- Note color picker
- Clickable notes
- AI meaning + emotion review
- Floating purple AI chatbot
- Lightweight RAG retrieval
- Insights dashboard
- Plotly visualizations
- GitHub dataset persistence
"""

# ==========================================================================
# IMPORTS
# ==========================================================================

import base64
import html
import json
import os
from datetime import datetime
from io import StringIO

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import InferenceClient


# ==========================================================================
# CONFIGURATION
# ==========================================================================

DATA_PATH = "data/messages.csv"

COLUMNS = [
    "id",
    "timestamp",
    "target_type",
    "target_name",
    "message",
    "sender_name",
    "sentiment",
    "emoji_tag",
    "keywords",
    "suggestion",
    "views",
    "note_color",
    "meaning",
    "emotion",
    "secondary_emotion",
    "intensity",
    "analyzed_at",
]


# ==========================================================================
# EMOTION CATEGORIES
# ==========================================================================

EMOTION_CATEGORIES = {
    "Happiness": "😊",
    "Sadness": "😢",
    "Love": "❤️",
    "Anger": "😡",
    "Loneliness": "😔",
    "Fear": "😨",
    "Gratitude": "😌",
    "Excitement": "🎉",
    "Confusion": "😕",
    "Heartbreak": "💔",
    "Neutral": "😐",
    "Support": "🤝",
    "Surprise": "😮",
}


# ==========================================================================
# NOTE COLORS
# ==========================================================================

# name -> (picker emoji, paper color, tape color, ink color, soft ink color)

NOTE_COLORS = {
    "White": (
        "⚪",
        "#fffdf6",
        "#e2d9c6",
        "#3b2f38",
        "#7a6a75",
    ),
    "Yellow": (
        "🟡",
        "#fff3b0",
        "#ffd54a",
        "#3b2f38",
        "#7a6a75",
    ),
    "Orange": (
        "🟠",
        "#ffd9b0",
        "#ffa94d",
        "#3b2a1a",
        "#7a5a3a",
    ),
    "Red": (
        "🔴",
        "#ffc9c9",
        "#ff8787",
        "#3b1f1f",
        "#7a4a4a",
    ),
    "Pink": (
        "🩷",
        "#ffd6e3",
        "#ff9ebd",
        "#3b2f38",
        "#7a6a75",
    ),
    "Purple": (
        "🟣",
        "#e4d7ff",
        "#b79cff",
        "#2f2a3b",
        "#6a6480",
    ),
    "Blue": (
        "🔵",
        "#cfe6ff",
        "#8ec5ff",
        "#1f2f3b",
        "#4a6072",
    ),
    "Teal": (
        "🩵",
        "#c6f0ea",
        "#7fd8cc",
        "#1f3b37",
        "#4a726c",
    ),
    "Green": (
        "🟢",
        "#d5f5df",
        "#8fdcaa",
        "#1f3b2a",
        "#4a725a",
    ),
    "Gray": (
        "🩶",
        "#e3e5e8",
        "#aeb4bb",
        "#2b2f33",
        "#666c72",
    ),
    "Brown": (
        "🟤",
        "#c9a27a",
        "#8b5e3c",
        "#2d1f14",
        "#5a4030",
    ),
    "Black": (
        "⚫",
        "#2b2b2f",
        "#8a8a92",
        "#f5f5f7",
        "#b5b5bd",
    ),
}

COLOR_NAMES = list(NOTE_COLORS.keys())

FALLBACK_COLORS = [
    color
    for color in COLOR_NAMES
    if color not in ("Black", "Brown")
]

TILTS = [-2.0, 1.5, -1.0, 2.0, -1.5, 1.0]


# ==========================================================================
# STREAMLIT CONFIG
# ==========================================================================

st.set_page_config(
    page_title="Confession Wall",
    page_icon="🎓",
    layout="wide",
)


# ==========================================================================
# HUGGING FACE GENAI
# ==========================================================================

HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

HF_TOKEN = st.secrets.get(
    "HF_TOKEN",
    os.getenv("HF_TOKEN", ""),
)

client = InferenceClient(
    token=HF_TOKEN
)


def _chat(prompt: str, temperature: float = 0.4) -> str:
    """
    Send a single prompt to the Hugging Face chat model.
    """

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not configured in Streamlit Secrets."
        )

    response = client.chat_completion(
        model=HF_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


# ==========================================================================
# INITIAL MESSAGE ANALYSIS
# ==========================================================================

def analyze_message(
    message: str,
    target_type: str,
    target_name: str,
) -> dict:
    """
    Analyze a newly posted message using GenAI.

    Returns:
        sentiment
        emoji_tag
        keywords
        suggestion
    """

    prompt = f"""
You are analyzing a farewell message written by a graduating student.

The message is addressed to:
{target_type} - {target_name}

Message:
"{message}"

Return ONLY a valid JSON object.

Do not use markdown.
Do not add explanations.
Do not add text before or after the JSON.

Use exactly these keys:

"sentiment":
Choose ONE:
- Grateful
- Nostalgic
- Critical
- Hopeful
- Mixed
- Neutral

"emoji_tag":
A short emoji plus a 1-2 word label.

Example:
"🌿 Grateful"

"keywords":
A list of up to 3 short theme keywords.

Example:
["teaching style", "patience"]

"suggestion":
If the message contains a suggestion, request, concern, or complaint
for the school, summarize it in ONE short sentence.

If there is no suggestion, return an empty string.
"""

    try:
        text = _chat(
            prompt,
            temperature=0.4
        )

        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "The AI did not return valid JSON."
            )

        text = text[start:end + 1]

        result = json.loads(text)

        sentiment = result.get(
            "sentiment",
            "Neutral"
        )

        allowed_sentiments = {
            "Grateful",
            "Nostalgic",
            "Critical",
            "Hopeful",
            "Mixed",
            "Neutral",
        }

        if sentiment not in allowed_sentiments:
            sentiment = "Neutral"

        keywords = result.get(
            "keywords",
            []
        )

        if not isinstance(keywords, list):
            keywords = []

        return {
            "sentiment": sentiment,
            "emoji_tag": str(
                result.get(
                    "emoji_tag",
                    ""
                )
            ),
            "keywords": keywords[:3],
            "suggestion": str(
                result.get(
                    "suggestion",
                    ""
                )
            ),
        }

    except Exception as e:

        st.error(
            f"GenAI analysis failed: {e}"
        )

        return {
            "sentiment": "Neutral",
            "emoji_tag": "💭 Unanalyzed",
            "keywords": [],
            "suggestion": "",
        }


# ==========================================================================
# LIGHTWEIGHT RAG RETRIEVAL
# ==========================================================================

def _retrieve_wall_context(
    question: str,
    df: pd.DataFrame,
    limit: int = 12,
):
    """
    Lightweight RAG system.

    Python retrieves relevant wall records first.
    Hugging Face then interprets the retrieved records.
    """

    q = str(
        question
    ).strip().lower()

    work = df.copy()

    if work.empty:
        return (
            "No wall records are available.",
            {"intent": "empty", "records": []},
        )

    work["_id"] = pd.to_numeric(
        work["id"],
        errors="coerce"
    ).fillna(0)

    work["_text"] = (
        work["target_name"]
        .fillna("")
        .astype(str)
        + " "
        + work["message"]
        .fillna("")
        .astype(str)
        + " "
        + work["sender_name"]
        .fillna("")
        .astype(str)
        + " "
        + work["keywords"]
        .fillna("")
        .astype(str)
    ).str.lower()

    latest_words = [
        "latest",
        "newest",
        "most recent",
        "recent",
        "pinakabag-o",
        "pinakalatest",
        "karon",
        "today",
    ]

    author_words = [
        "who posted",
        "who wrote",
        "author",
        "sender",
        "posted by",
        "kinsa nag",
        "kinsay nag",
        "sino ang nag",
    ]

    meaning_words = [
        "what does",
        "what did",
        "mean",
        "meaning",
        "meant",
        "what is this saying",
        "ano ibig sabihin",
        "unsa pasabot",
    ]

    # --------------------------------------------------------------
    # LATEST MESSAGE
    # --------------------------------------------------------------

    if any(
        word in q
        for word in latest_words
    ):

        ordered = (
            work
            .sort_values(
                "_id",
                ascending=False
            )
            .head(limit)
        )

        intent = "latest"

    else:

        # ----------------------------------------------------------
        # FIND KNOWN NAMES
        # ----------------------------------------------------------

        known_names = []

        for column in (
            "sender_name",
            "target_name",
        ):

            for value in (
                work[column]
                .fillna("")
                .astype(str)
            ):

                value = value.strip()

                if (
                    value
                    and value.lower()
                    not in {
                        "anonymous",
                        "nan",
                    }
                    and len(value) >= 2
                ):

                    known_names.append(value)

        known_names = sorted(
            set(known_names),
            key=len,
            reverse=True,
        )

        # ----------------------------------------------------------
        # SCORE RECORDS
        # ----------------------------------------------------------

        scored = []

        for _, row in work.iterrows():

            text_value = row["_text"]

            score = 0
            matched = []

            for name in known_names:

                if (
                    name.lower() in q
                    and name.lower() in text_value
                ):

                    score += 100
                    matched.append(name)

            tokens = (
                q.replace("?", " ")
                .replace(",", " ")
                .replace(".", " ")
                .split()
            )

            for token in tokens:

                if len(token) >= 3 and token in text_value:
                    score += 2
                    matched.append(token)

            scored.append(
                (
                    score,
                    row["_id"],
                    matched,
                    row,
                )
            )

        scored.sort(
            key=lambda x: (
                x[0],
                x[1],
            ),
            reverse=True,
        )

        positive = [
            item
            for item in scored
            if item[0] > 0
        ]

        if positive:

            ordered = pd.DataFrame(
                [
                    item[3]
                    for item in positive[:limit]
                ]
            )

        else:

            ordered = (
                work
                .sort_values(
                    "_id",
                    ascending=False
                )
                .head(limit)
            )

        if any(
            word in q
            for word in (
                author_words
                + meaning_words
            )
        ):

            intent = "author_or_meaning"

        else:

            intent = "general"

    # --------------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------------

    records = []

    for _, row in ordered.iterrows():

        try:
            note_id = int(
                float(
                    row.get(
                        "_id",
                        0
                    )
                )
            )

        except Exception:
            note_id = 0

        records.append(
            {
                "id": note_id,
                "timestamp": str(
                    row.get(
                        "timestamp",
                        ""
                    )
                ),
                "author": str(
                    row.get(
                        "sender_name",
                        "Anonymous"
                    )
                ),
                "recipient": str(
                    row.get(
                        "target_name",
                        ""
                    )
                ),
                "message": str(
                    row.get(
                        "message",
                        ""
                    )
                ),
                "color": str(
                    row.get(
                        "note_color",
                        ""
                    )
                ),
                "sentiment": str(
                    row.get(
                        "sentiment",
                        ""
                    )
                ),
                "emotion": str(
                    row.get(
                        "emotion",
                        ""
                    )
                ),
            }
        )

    context = json.dumps(
        records,
        ensure_ascii=False,
        indent=2,
    )

    return context, {
        "intent": intent,
        "records": records,
    }


# ==========================================================================
# CHATBOT
# ==========================================================================

def chatbot_answer(
    question: str,
    df: pd.DataFrame,
    history=None,
) -> str:

    if history is None:
        history = []

    wall_context, meta = _retrieve_wall_context(
        question,
        df,
        limit=12,
    )

    # --------------------------------------------------------------
    # DETERMINISTIC LATEST ANSWER
    # --------------------------------------------------------------

    if (
        meta.get("intent") == "latest"
        and meta.get("records")
    ):

        latest = meta["records"][0]

        author = (
            latest["author"]
            or "Anonymous"
        )

        recipient = latest["recipient"]

        if recipient:

            recipient_text = (
                f" to {recipient}"
            )

        else:

            recipient_text = ""

        return (
            f'The latest note was posted by '
            f'**{author}**{recipient_text}. '
            f'It says: "{latest["message"]}"'
        )

    # --------------------------------------------------------------
    # RECENT CHAT CONTEXT
    # --------------------------------------------------------------

    convo = ""

    for turn in history[-6:]:

        role = (
            "Student"
            if turn["role"] == "user"
            else "You"
        )

        convo += (
            f"{role}: "
            f"{turn['content']}\n"
        )

    # --------------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------------

    prompt = f"""
You are the AI assistant inside a Confession Wall.

Your job is to answer questions about messages actually posted
on the wall.

SOURCE OF TRUTH:

The RETRIEVED WALL RECORDS below were selected by the application
from the current wall.

Use those records for wall-related questions.

Do NOT invent:
- authors
- messages
- dates
- recipients
- relationships
- events

IMPORTANT:

author = the person who posted or wrote the note.

recipient = who or what the note was addressed to.

If the user asks what a message means:
Explain the message directly in 1-2 short sentences.

Do NOT start with filler such as:
"I understand..."
"That's a straightforward message..."
"Let me analyze..."

If the user asks about a newly posted note:
Use the newest relevant retrieved record.

If evidence is insufficient:
Say that you don't have enough information instead of guessing.

Keep answers concise and specific.

Markdown bold is allowed for names or important points.

RETRIEVED WALL RECORDS:

{wall_context}

RECENT CHAT:

{convo}

USER QUESTION:

{question}

Answer directly.
"""

    try:

        return _chat(
            prompt,
            temperature=0.25
        )

    except Exception as e:

        return (
            f"Sorry, I couldn't process that: {e}"
        )


# ==========================================================================
# EMOTION ANALYZER
# ==========================================================================

def analyze_emotion(
    message: str,
) -> dict:

    categories = ", ".join(
        EMOTION_CATEGORIES.keys()
    )

    prompt = f"""
You are reading one short message posted on a farewell/message wall.

Messages can range from:
- heartfelt notes
- casual greetings
- appreciation
- complaints
- simple messages

Read the message for what it actually says.

Do not over-interpret.

Do not invent:
- events
- relationships
- names
- psychological diagnoses

If the message is ambiguous, use wording such as:

"The message appears to express..."

If the message is clearly neutral or simply a casual greeting,
say so plainly and use "Neutral" as the emotion.

IMPORTANT:

Never use the word "confession" in your response.

Call it:
- "this message"
- "this note"

Match the tone to the actual message.

Message:

"{message}"

Return ONLY a valid JSON object.

Use exactly these keys:

"meaning":
ONE TO TWO short sentences explaining what the message appears
to express.

Go directly to the meaning.

Do not use filler such as:
"I understand..."
"Let me analyze..."

"emotion":
Choose ONE category from this exact list:

[{categories}]

"secondary_emotion":
Choose a second category from the same list,
or "" if none fits.

"intensity":
Choose ONE:
- Mild
- Moderate
- Strong
"""

    try:

        text = _chat(
            prompt,
            temperature=0.3
        )

        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "Invalid JSON returned by AI."
            )

        text = text[
            start:end + 1
        ]

        result = json.loads(text)

        emotion = result.get(
            "emotion",
            "Neutral"
        )

        secondary = result.get(
            "secondary_emotion",
            ""
        )

        intensity = result.get(
            "intensity",
            "Mild"
        )

        if emotion not in EMOTION_CATEGORIES:
            emotion = "Neutral"

        if secondary not in EMOTION_CATEGORIES:
            secondary = ""

        if intensity not in {
            "Mild",
            "Moderate",
            "Strong",
        }:
            intensity = "Mild"

        return {
            "meaning": str(
                result.get(
                    "meaning",
                    ""
                )
            ),
            "emotion": emotion,
            "secondary_emotion": secondary,
            "intensity": intensity,
        }

    except Exception as e:

        return {
            "meaning": (
                f"Analysis unavailable: {e}"
            ),
            "emotion": "Neutral",
            "secondary_emotion": "",
            "intensity": "Mild",
        }


# ==========================================================================
# GITHUB CONFIGURATION
# ==========================================================================

GITHUB_OWNER = st.secrets.get(
    "GITHUB_OWNER",
    "rcgbandoy-collab",
)

GITHUB_REPO = st.secrets.get(
    "GITHUB_REPO",
    "Confession-Wall",
)

GITHUB_BRANCH = st.secrets.get(
    "GITHUB_BRANCH",
    "main",
)

GITHUB_FILE = st.secrets.get(
    "GITHUB_FILE",
    "data/messages.csv",
)

GITHUB_TOKEN = st.secrets.get(
    "GITHUB_TOKEN",
    "",
)

GITHUB_API_URL = (
    f"https://api.github.com/repos/"
    f"{GITHUB_OWNER}/"
    f"{GITHUB_REPO}/contents/"
    f"{GITHUB_FILE}"
)


# ==========================================================================
# GITHUB HELPERS
# ==========================================================================

def github_headers():

    return {
        "Authorization": (
            f"Bearer {GITHUB_TOKEN}"
        ),
        "Accept": (
            "application/vnd.github+json"
        ),
        "X-GitHub-Api-Version": (
            "2022-11-28"
        ),
    }


# ==========================================================================
# DATA LOADING
# ==========================================================================

def load_data() -> pd.DataFrame:
    """
    Load the latest messages.csv from GitHub.

    If GitHub cannot be reached,
    fall back to the local dataset.
    """

    df = None

    # --------------------------------------------------------------
    # TRY GITHUB
    # --------------------------------------------------------------

    if GITHUB_TOKEN:

        try:

            response = requests.get(
                GITHUB_API_URL,
                headers=github_headers(),
                params={
                    "ref": GITHUB_BRANCH
                },
                timeout=20,
            )

            if response.status_code == 200:

                file_data = response.json()

                encoded_content = (
                    file_data["content"]
                )

                csv_bytes = base64.b64decode(
                    encoded_content
                )

                csv_text = csv_bytes.decode(
                    "utf-8"
                )

                df = pd.read_csv(
                    StringIO(csv_text)
                )

        except Exception:
            df = None

    # --------------------------------------------------------------
    # LOCAL FALLBACK
    # --------------------------------------------------------------

    if df is None:

        try:

            df = pd.read_csv(
                DATA_PATH
            )

        except Exception:

            df = pd.DataFrame(
                columns=COLUMNS
            )

    # --------------------------------------------------------------
    # ADD MISSING COLUMNS
    # --------------------------------------------------------------

    for column in COLUMNS:

        if column not in df.columns:
            df[column] = ""

    # --------------------------------------------------------------
    # CLEAN MESSAGE
    # --------------------------------------------------------------

    df["message"] = (
        df["message"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["message"] != ""
    ].copy()

    # --------------------------------------------------------------
    # CLEAN SENDER
    # --------------------------------------------------------------

    df["sender_name"] = (
        df["sender_name"]
        .fillna("Anonymous")
        .astype(str)
        .str.strip()
    )

    df.loc[
        df["sender_name"] == "",
        "sender_name"
    ] = "Anonymous"

    return df


# ==========================================================================
# SAVE DATASET TO GITHUB
# ==========================================================================

def save_dataframe_to_github(
    df: pd.DataFrame,
    commit_message: str,
) -> bool:

    if not GITHUB_TOKEN:

        st.error(
            "GitHub connection is not configured. "
            "Please check Streamlit Secrets."
        )

        return False

    try:

        # ----------------------------------------------------------
        # GET CURRENT FILE SHA
        # ----------------------------------------------------------

        response = requests.get(
            GITHUB_API_URL,
            headers=github_headers(),
            params={
                "ref": GITHUB_BRANCH
            },
            timeout=20,
        )

        if response.status_code != 200:

            st.error(
                "Could not read the GitHub dataset.\n\n"
                f"Status: {response.status_code}\n"
                f"{response.text}"
            )

            return False

        current_file = response.json()

        current_sha = current_file[
            "sha"
        ]

        # ----------------------------------------------------------
        # CONVERT DATAFRAME TO CSV
        # ----------------------------------------------------------

        csv_content = df.to_csv(
            index=False
        )

        encoded_content = base64.b64encode(
            csv_content.encode("utf-8")
        ).decode("utf-8")

        # ----------------------------------------------------------
        # UPDATE GITHUB
        # ----------------------------------------------------------

        payload = {
            "message": commit_message,
            "content": encoded_content,
            "sha": current_sha,
            "branch": GITHUB_BRANCH,
        }

        update_response = requests.put(
            GITHUB_API_URL,
            headers=github_headers(),
            json=payload,
            timeout=20,
        )

        if update_response.status_code in (
            200,
            201,
        ):

            # Update local copy too.
            os.makedirs(
                os.path.dirname(DATA_PATH),
                exist_ok=True,
            )

            df.to_csv(
                DATA_PATH,
                index=False
            )

            return True

        st.error(
            "GitHub dataset update failed.\n\n"
            f"Status: {update_response.status_code}\n"
            f"{update_response.text}"
        )

        return False

    except Exception as e:

        st.error(
            f"GitHub connection error: {e}"
        )

        return False


# ==========================================================================
# SAVE NEW MESSAGE
# ==========================================================================

def save_message(
    row: dict,
) -> bool:

    df = load_data()

    # --------------------------------------------------------------
    # CREATE NEW ID
    # --------------------------------------------------------------

    if len(df):

        numeric_ids = (
            pd.to_numeric(
                df["id"],
                errors="coerce",
            )
            .dropna()
        )

        if not numeric_ids.empty:

            new_id = (
                int(
                    numeric_ids.max()
                )
                + 1
            )

        else:

            new_id = 1

    else:

        new_id = 1

    row["id"] = new_id

    # --------------------------------------------------------------
    # ADD MESSAGE
    # --------------------------------------------------------------

    updated_df = pd.concat(
        [
            df,
            pd.DataFrame([row]),
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    return save_dataframe_to_github(
        updated_df,
        f"Add confession message #{new_id}",
    )


# ==========================================================================
# SAVE AI EMOTION ANALYSIS
# ==========================================================================

def update_message_analysis(
    note_id: int,
    analysis: dict,
):

    df = load_data()

    numeric_ids = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    mask = (
        numeric_ids
        == note_id
    )

    if not mask.any():
        return False

    df.loc[
        mask,
        "meaning"
    ] = analysis.get(
        "meaning",
        "",
    )

    df.loc[
        mask,
        "emotion"
    ] = analysis.get(
        "emotion",
        "Neutral",
    )

    df.loc[
        mask,
        "secondary_emotion"
    ] = analysis.get(
        "secondary_emotion",
        "",
    )

    df.loc[
        mask,
        "intensity"
    ] = analysis.get(
        "intensity",
        "Mild",
    )

    df.loc[
        mask,
        "analyzed_at"
    ] = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    return save_dataframe_to_github(
        df,
        f"Update AI analysis for note #{note_id}",
    )


# ==========================================================================
# GET OR GENERATE EMOTION ANALYSIS
# ==========================================================================

def get_or_analyze_emotion(
    row: pd.Series,
) -> dict:

    saved_emotion = str(
        row.get(
            "emotion",
            ""
        )
    ).strip()

    if saved_emotion:

        return {
            "meaning": row.get(
                "meaning",
                ""
            ),
            "emotion": row.get(
                "emotion",
                "Neutral"
            ),
            "secondary_emotion": row.get(
                "secondary_emotion",
                ""
            ),
            "intensity": row.get(
                "intensity",
                "Mild"
            ),
        }

    result = analyze_emotion(
        row["message"]
    )

    update_message_analysis(
        int(
            float(
                row["id"]
            )
        ),
        result,
    )

    return result


# ==========================================================================
# GLOBAL STYLING
# ==========================================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&family=Quicksand:wght@400;600;700&display=swap'
);


/* ================================================================
   PAPER WALL
   ================================================================ */

.wall {
    column-count: 3;
    column-gap: 26px;
    padding: 12px 6px 30px;
}

@media (max-width: 900px) {
    .wall {
        column-count: 2;
    }
}

@media (max-width: 600px) {
    .wall {
        column-count: 1;
    }
}


/* ================================================================
   NOTE
   ================================================================ */

.note {

    --tilt: 0deg;
    --tape: #ddd;
    --ink: #333;
    --sub: #777;

    position: relative;

    display: inline-block;

    width: 100%;

    break-inside: avoid;

    box-sizing: border-box;

    margin: 0 0 30px;

    padding: 30px 24px 18px;

    border-radius:
        6px 6px 18px 6px;

    box-shadow:
        0 6px 16px
        rgba(0, 0, 0, 0.20);

    transform:
        rotate(var(--tilt));

    transition:
        transform .25s ease,
        box-shadow .25s ease;

    text-align: left;

    overflow-wrap: anywhere;

    background-image:
        repeating-linear-gradient(
            transparent 0 27px,
            rgba(128,128,128,0.16)
            27px 28px
        );
}

.note:hover {

    transform:
        rotate(0deg)
        translateY(-6px)
        scale(1.02);

    box-shadow:
        0 14px 28px
        rgba(0,0,0,.28);
}

.note::before {

    content: '';

    position: absolute;

    top: -12px;

    left: 50%;

    width: 84px;

    height: 24px;

    transform:
        translateX(-50%)
        rotate(-3deg);

    background:
        var(--tape);

    opacity: .8;

    border-radius: 3px;
}

.note-to {

    font-family:
        'Quicksand',
        sans-serif;

    font-weight: 700;

    font-size: 13px;

    letter-spacing: 1px;

    text-transform: uppercase;

    color: var(--sub);
}

.note-msg {

    font-family:
        'Caveat',
        cursive;

    font-size: 25px;

    line-height: 28px;

    color: var(--ink);

    margin: 14px 0;
}

.note-tag {

    display: inline-block;

    background:
        rgba(128,128,128,.22);

    color: var(--ink);

    padding:
        3px 12px;

    border-radius:
        999px;

    font-size: 12px;

    font-family:
        'Quicksand',
        sans-serif;

    margin-top: 10px;
}

.note-from {

    font-family:
        'Caveat',
        cursive;

    font-size: 21px;

    color: var(--sub);

    margin-top: 10px;
}

.note-date {

    font-family:
        'Quicksand',
        sans-serif;

    font-size: 11px;

    color: var(--sub);

    margin-top: 4px;
}

.single {

    max-width: 420px;

    margin: 24px auto;
}


/* ================================================================
   CHAT BACKDROP
   ================================================================ */

.st-key-chat_overlay,
.st-key-note_overlay {

    position: fixed !important;

    inset: 0 !important;

    z-index: 9998 !important;

    background:
        rgba(20, 10, 30, .50) !important;

    backdrop-filter:
        blur(7px) !important;

    -webkit-backdrop-filter:
        blur(7px) !important;

    display: flex !important;

    align-items: center !important;

    justify-content: center !important;

    padding: 20px !important;

    overflow-y: auto !important;
}


/* ================================================================
   OUTSIDE CLICK
   ================================================================ */

.st-key-chat_backdrop_close,
.st-key-note_backdrop_close {

    position: fixed !important;

    inset: 0 !important;

    z-index: 1 !important;

    padding: 0 !important;

    margin: 0 !important;

    pointer-events: auto !important;
}

.st-key-chat_backdrop_close button,
.st-key-note_backdrop_close button {

    position: absolute !important;

    inset: 0 !important;

    width: 100% !important;

    height: 100% !important;

    opacity: 0 !important;

    background:
        transparent !important;

    border: 0 !important;

    cursor: default !important;
}


/* ================================================================
   CHAT MODAL
   ================================================================ */

.st-key-chat_modal {

    position: relative !important;

    z-index: 2 !important;

    background:
        transparent !important;

    border-radius: 0 !important;

    width: 460px !important;

    max-width: 92vw !important;

    max-height: 82vh !important;

    box-shadow: none !important;

    display: flex !important;

    flex-direction: column !important;

    overflow: hidden !important;

    animation:
        cwOpen .18s ease-out;
}

@keyframes cwOpen {

    from {
        opacity: 0;
        transform:
            translateY(10px)
            scale(.97);
    }

    to {
        opacity: 1;
        transform:
            translateY(0)
            scale(1);
    }
}


/* ================================================================
   CHAT HEADER
   ================================================================ */

.st-key-chat_header {

    background:
        transparent !important;

    padding:
        6px 4px 6px 18px !important;
}

.st-key-chat_header button {

    background:
        rgba(0,0,0,.25) !important;

    border: none !important;

    box-shadow: none !important;

    color: #fff !important;

    font-size: 16px !important;

    width: 34px !important;

    height: 34px !important;

    margin-top: 2px !important;

    border-radius: 50% !important;
}

.cw-header-title {

    font-family:
        'Quicksand',
        sans-serif;

    font-weight: 700;

    font-size: 15px;

    color: #fff;

    padding-top: 10px;

    text-shadow:
        0 1px 4px
        rgba(0,0,0,.5);
}

.cw-header-sub {

    font-family:
        'Quicksand',
        sans-serif;

    font-size: 12px;

    color:
        rgba(255,255,255,.85);

    margin-top: 2px;

    padding-bottom: 10px;

    text-shadow:
        0 1px 4px
        rgba(0,0,0,.5);
}


/* ================================================================
   CHAT BODY
   ================================================================ */

.cw-body {

    padding:
        12px 14px 4px;

    max-height: 46vh;

    overflow-y: auto;

    scroll-behavior:
        smooth;
}

.cw-row {

    display: flex;

    margin: 7px 0;

    animation:
        cwIn .2s ease;
}

.cw-row.ai {
    justify-content:
        flex-start;
}

.cw-row.user {
    justify-content:
        flex-end;
}

@keyframes cwIn {

    from {
        opacity: 0;
        transform:
            translateY(6px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }
}

.cw-label {

    font-family:
        'Quicksand',
        sans-serif;

    font-size: 10px;

    color:
        rgba(255,255,255,.8);

    margin:
        0 4px 2px;

    text-align: right;
}

.cw-row.ai .cw-label {
    text-align: left;
}


/* ================================================================
   CHAT BUBBLES
   ================================================================ */

.cw-bubble {

    max-width: 78%;

    padding:
        9px 14px;

    border-radius:
        16px;

    font-family:
        'Quicksand',
        sans-serif;

    font-size: 14px;

    line-height: 1.4;

    box-shadow:
        0 2px 6px
        rgba(0,0,0,.25);

    overflow-wrap:
        anywhere;
}

.cw-bubble.ai {

    background:
        rgba(120,120,132,.92);

    color: #fff;

    border-bottom-left-radius:
        4px;
}

.cw-bubble.user {

    background:
        #2563eb;

    color: #fff;

    border-bottom-right-radius:
        4px;
}


/* ================================================================
   TYPING INDICATOR
   ================================================================ */

.cw-typing {

    display: inline-flex;

    gap: 4px;

    padding:
        12px 14px;

    background:
        rgba(120,120,132,.92);

    border-radius:
        16px;

    border-bottom-left-radius:
        4px;
}

.cw-typing span {

    width: 6px;

    height: 6px;

    border-radius: 50%;

    background: #fff;

    animation:
        cwBlink 1.2s
        infinite ease-in-out;
}

.cw-typing span:nth-child(2) {
    animation-delay: .2s;
}

.cw-typing span:nth-child(3) {
    animation-delay: .4s;
}

@keyframes cwBlink {

    0%,
    80%,
    100% {
        opacity: .3;
    }

    40% {
        opacity: 1;
    }
}


/* ================================================================
   CHAT INPUT
   ================================================================ */

.st-key-chat_inputbar {

    border-top:
        none !important;

    padding:
        10px 12px !important;

    background:
        transparent !important;
}

.st-key-chat_inputbar input {

    border-radius:
        999px !important;

    border:
        none !important;

    padding:
        8px 14px !important;

    font-family:
        'Quicksand',
        sans-serif !important;

    background:
        rgba(255,255,255,.92) !important;

    color: #111 !important;
}

.st-key-chat_inputbar button {

    background:
        #2563eb !important;

    color: #fff !important;

    border:
        none !important;

    border-radius:
        50% !important;

    width: 36px !important;

    height: 36px !important;

    padding: 0 !important;
}


/* ================================================================
   CLICKABLE NOTES
   ================================================================ */

[class*="st-key-note_wrap_"] {

    position:
        relative !important;

    margin-bottom:
        30px !important;
}

[class*="st-key-note_wrap_"] .note {

    margin-bottom:
        0 !important;
}

[class*="st-key-note_wrap_"]
[data-testid="stButton"] {

    position:
        absolute !important;

    inset:
        0 !important;

    z-index:
        20 !important;

    margin:
        0 !important;

    padding:
        0 !important;

    pointer-events:
        auto !important;
}

[class*="st-key-note_wrap_"]
[data-testid="stButton"] button {

    width:
        100% !important;

    height:
        100% !important;

    min-height:
        100% !important;

    border:
        none !important;

    border-radius:
        8px 8px 18px 8px !important;

    background:
        transparent !important;

    color:
        transparent !important;

    box-shadow:
        none !important;

    padding:
        0 !important;

    opacity:
        1 !important;

    cursor:
        pointer !important;
}

[class*="st-key-note_wrap_"]
[data-testid="stButton"]
button p {

    color:
        transparent !important;
}


/* ================================================================
   NOTE ANALYZER
   ================================================================ */

.st-key-note_modal {

    position:
        relative !important;

    z-index:
        2 !important;

    background:
        #1c1626 !important;

    border-radius:
        22px !important;

    width:
        820px !important;

    max-width:
        94vw !important;

    max-height:
        88vh !important;

    box-shadow:
        0 25px 60px
        rgba(0,0,0,.5) !important;

    overflow:
        hidden !important;

    animation:
        cwOpen .18s ease-out;
}

.st-key-note_close button,
.st-key-chat_header button {

    background:
        rgba(255,255,255,.10) !important;

    border:
        none !important;

    color:
        #fff !important;

    border-radius:
        50% !important;

    width:
        32px !important;

    height:
        32px !important;

    padding:
        0 !important;
}

.cw-note-grid {

    display:
        flex;

    gap:
        0;

    overflow-y:
        auto;
}

.cw-note-left {

    flex:
        1;

    padding:
        22px;

    display:
        flex;

    align-items:
        flex-start;

    justify-content:
        center;
}

.cw-note-right {

    flex:
        1;

    padding:
        24px;

    border-left:
        1px solid
        rgba(255,255,255,.08);
}

@media (max-width: 720px) {

    .cw-note-grid {
        flex-direction:
            column;
    }

    .cw-note-right {

        border-left:
            none;

        border-top:
            1px solid
            rgba(255,255,255,.08);
    }
}

.cw-ai-badge {

    font-family:
        'Quicksand',
        sans-serif;

    font-weight:
        700;

    font-size:
        13px;

    color:
        #b79cff;

    letter-spacing:
        .5px;

    margin-bottom:
        14px;
}

.cw-meaning-label,
.cw-emotion-label {

    font-family:
        'Quicksand',
        sans-serif;

    font-weight:
        700;

    font-size:
        12px;

    letter-spacing:
        1px;

    text-transform:
        uppercase;

    color:
        #9a8fb0;

    margin:
        14px 0 6px;
}

.cw-meaning-text {

    font-family:
        'Quicksand',
        sans-serif;

    font-size:
        15px;

    line-height:
        1.5;

    color:
        #f0eaf7;
}

.cw-emotion-card {

    margin-top:
        14px;

    padding:
        14px;

    border-radius:
        14px;

    background:
        rgba(255,255,255,.06);

    border:
        1px solid
        rgba(255,255,255,.08);
}

.cw-emotion-badge {

    font-family:
        'Quicksand',
        sans-serif;

    font-size:
        17px;

    font-weight:
        700;

    color:
        #f0eaf7;
}

.cw-emotion-secondary {

    font-family:
        'Quicksand',
        sans-serif;

    font-size:
        13px;

    color:
        #b0a4c4;

    margin-top:
        5px;
}

.cw-intensity {

    font-family:
        'Quicksand',
        sans-serif;

    font-size:
        12px;

    color:
        #9a8fb0;

    margin-top:
        8px;
}


/* ================================================================
   NAVIGATION
   ================================================================ */

.cw-nav
[data-testid="stRadio"] > div {

    gap:
        22px;

    border-bottom:
        1px solid
        rgba(255,255,255,.12);

    padding-bottom:
        0;
}

.cw-nav
[data-testid="stRadio"]
label {

    padding:
        6px 2px 10px !important;
}

.cw-nav
[data-testid="stRadio"]
label > div:first-child {

    display:
        none;
}

.cw-nav
[data-testid="stRadio"]
label p {

    font-family:
        'Quicksand',
        sans-serif;

    font-weight:
        700;

    font-size:
        14px;

    color:
        #9a8fb0;
}

.cw-nav
[data-testid="stRadio"]
label:has(input:checked) p {

    color:
        #ff9ebd;
}

.cw-nav
[data-testid="stRadio"]
label:has(input:checked) {

    border-bottom:
        2px solid
        #ff9ebd;
}

</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================================
# FLOATING CHAT BUTTON JAVASCRIPT
# ==========================================================================

_CHAT_HEAD_JS = r"""
(function () {

    var win = window.parent;
    var doc = win.document;

    var SIZE = 56;


    function findButtonByText(text) {

        var buttons =
            doc.querySelectorAll(
                '[data-testid="stButton"] button'
            );

        for (
            var i = 0;
            i < buttons.length;
            i++
        ) {

            if (
                buttons[i]
                .textContent
                .trim()
                === text
            ) {

                return buttons[i];
            }
        }

        return null;
    }


    function place(
        wrap,
        left,
        top
    ) {

        var maxLeft =
            win.innerWidth
            - SIZE
            - 4;

        var maxTop =
            win.innerHeight
            - SIZE
            - 4;

        wrap.style.left =
            Math.max(
                4,
                Math.min(
                    left,
                    maxLeft
                )
            ) + "px";

        wrap.style.top =
            Math.max(
                4,
                Math.min(
                    top,
                    maxTop
                )
            ) + "px";

        wrap.style.right =
            "auto";

        wrap.style.bottom =
            "auto";
    }


    function bindOnce() {

        if (
            win.__cwHeadBound
        ) {
            return;
        }

        win.__cwHeadBound =
            true;


        doc.addEventListener(
            "pointermove",
            function (event) {

                var state =
                    win.__cwHeadState;

                if (
                    !state
                    || !state.down
                ) {
                    return;
                }

                var dx =
                    event.clientX
                    - state.x0;

                var dy =
                    event.clientY
                    - state.y0;


                if (
                    !state.moved
                    &&
                    Math.abs(dx)
                    +
                    Math.abs(dy)
                    > 4
                ) {

                    state.moved =
                        true;
                }


                if (
                    !state.moved
                ) {
                    return;
                }


                state.nextL =
                    state.left0
                    + dx;

                state.nextT =
                    state.top0
                    + dy;


                if (
                    !state.raf
                ) {

                    state.raf =
                        win.requestAnimationFrame(
                            function () {

                                state.raf =
                                    null;

                                place(
                                    state.wrap,
                                    state.nextL,
                                    state.nextT
                                );
                            }
                        );
                }
            }
        );


        doc.addEventListener(
            "pointerup",
            function () {

                var state =
                    win.__cwHeadState;

                if (
                    !state
                    || !state.down
                ) {
                    return;
                }


                state.down =
                    false;

                clearTimeout(
                    state.timer
                );

                state.wrap.style.transform =
                    "";


                if (
                    state.moved
                ) {

                    state.swallow =
                        true;

                    setTimeout(
                        function () {

                            state.swallow =
                                false;

                        },
                        400
                    );


                    var rect =
                        state.wrap
                        .getBoundingClientRect();


                    try {

                        win.sessionStorage.setItem(
                            "cwHeadPos",
                            JSON.stringify({
                                left: rect.left,
                                top: rect.top
                            })
                        );

                    } catch (error) {}
                }
            }
        );


        doc.addEventListener(
            "click",
            function (event) {

                var state =
                    win.__cwHeadState;

                if (
                    state
                    && state.swallow
                ) {

                    state.swallow =
                        false;

                    event.stopPropagation();

                    event.preventDefault();
                }
            },
            true
        );
    }


    function onDown(event) {

        if (
            event.pointerType
            === "mouse"
            &&
            event.button !== 0
        ) {
            return;
        }


        var wrap =
            event.currentTarget;

        var rect =
            wrap.getBoundingClientRect();


        var state = {

            wrap: wrap,

            down: true,

            moved: false,

            swallow: false,

            x0:
                event.clientX,

            y0:
                event.clientY,

            left0:
                rect.left,

            top0:
                rect.top,

            timer: null,

            raf: null
        };


        state.timer =
            setTimeout(
                function () {

                    wrap.style.transform =
                        "scale(1.1)";

                },
                180
            );


        win.__cwHeadState =
            state;
    }


    function styleFloatingHead() {

        bindOnce();


        var button =
            findButtonByText(
                "💬"
            );


        if (!button) {
            return;
        }


        var wrap =
            button.closest(
                '[data-testid="stButton"]'
            );


        if (
            !wrap
            || wrap.dataset.cwStyled
        ) {
            return;
        }


        wrap.dataset.cwStyled =
            "1";


        wrap.style.position =
            "fixed";

        wrap.style.zIndex =
            "999997";

        wrap.style.touchAction =
            "none";

        wrap.style.transition =
            "transform .15s ease";


        button.style.background =
            "linear-gradient(135deg,#7c3aed,#9333ea)";

        button.style.color =
            "#fff";

        button.style.border =
            "none";

        button.style.borderRadius =
            "50%";

        button.style.width =
            SIZE + "px";

        button.style.height =
            SIZE + "px";

        button.style.padding =
            "0";

        button.style.fontSize =
            "24px";

        button.style.boxShadow =
            "0 6px 18px rgba(124,58,237,.5)";

        button.style.cursor =
            "grab";

        button.style.touchAction =
            "none";


        var saved = null;


        try {

            saved =
                JSON.parse(
                    win.sessionStorage.getItem(
                        "cwHeadPos"
                    )
                );

        } catch (error) {}


        if (
            saved
            &&
            typeof saved.left
            === "number"
        ) {

            place(
                wrap,
                saved.left,
                saved.top
            );

        } else {

            place(
                wrap,
                win.innerWidth
                - SIZE
                - 24,

                win.innerHeight
                - SIZE
                - 24
            );
        }


        wrap.addEventListener(
            "pointerdown",
            onDown
        );
    }


    styleFloatingHead();

    setTimeout(
        styleFloatingHead,
        200
    );

    setTimeout(
        styleFloatingHead,
        600
    );

})();
"""


# ==========================================================================
# HTML HELPERS
# ==========================================================================

def _clean(value) -> str:
    """
    Convert values into safe HTML text.
    """

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    return html.escape(
        str(value)
    )


# ==========================================================================
# NOTE HTML
# ==========================================================================

def _note_html(row) -> str:
    """
    Generate HTML for a paper note.
    """

    color = str(
        row.get(
            "note_color",
            ""
        )
    )

    try:

        note_id = int(
            float(
                row.get(
                    "id",
                    0
                )
            )
        )

    except Exception:

        note_id = 0


    if color not in NOTE_COLORS:

        color = (
            FALLBACK_COLORS[
                note_id
                % len(FALLBACK_COLORS)
            ]
        )


    (
        _,
        paper,
        tape,
        ink,
        sub,
    ) = NOTE_COLORS[color]


    tilt =
        TILTS[
            note_id
            % len(TILTS)
        ]


    try:

        date = (
            pd.to_datetime(
                row.get(
                    "timestamp"
                )
            )
            .strftime(
                "%B %d, %Y"
            )
        )

    except Exception:

        date = ""


    message = (
        _clean(
            row.get(
                "message",
                ""
            )
        )
        .replace(
            "\n",
            "<br>"
        )
    )


    target = _clean(
        row.get(
            "target_name",
            ""
        )
    )


    sender = _clean(
        row.get(
            "sender_name",
            "Anonymous"
        )
    )


    tag = _clean(
        row.get(
            "emoji_tag",
            ""
        )
    )


    tag_html = ""

    if tag:

        tag_html = (
            f'<span class="note-tag">'
            f'{tag}'
            f'</span>'
        )


    return (
        f'<div class="note" '
        f'style="'
        f'background-color:{paper};'
        f'--tape:{tape};'
        f'--ink:{ink};'
        f'--sub:{sub};'
        f'--tilt:{tilt}deg;'
        f'">'

        f'<div class="note-to">'
        f'To: {target}'
        f'</div>'

        f'<div class="note-msg">'
        f'{message}'
        f'</div>'

        f'<div class="note-from">'
        f'— {sender}'
        f'</div>'

        f'<div class="note-date">'
        f'{date}'
        f'</div>'

        f'{tag_html}'

        f'</div>'
    )


# ==========================================================================
# PREVIEW CARD
# ==========================================================================

def render_card(
    row,
):

    st.markdown(
        f'<div class="single">'
        f'{_note_html(row)}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# NOTE MODAL
# ==========================================================================

def render_note_modal():

    df = load_data()

    numeric_ids = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    match = df[
        numeric_ids
        == st.session_state.selected_note_id
    ]

    if match.empty:

        st.session_state.note_modal_open = False

        return


    row = match.iloc[0]


    with st.container(
        key="note_overlay"
    ):

        # ----------------------------------------------------------
        # BACKDROP CLOSE
        # ----------------------------------------------------------

        with st.container(
            key="note_backdrop_close"
        ):

            if st.button(
                "close backdrop",
                key="note_backdrop_btn",
            ):

                st.session_state.note_modal_open = False

                st.session_state.selected_note_id = None

                st.rerun()


        # ----------------------------------------------------------
        # MODAL
        # ----------------------------------------------------------

        with st.container(
            key="note_modal"
        ):

            top1, top2 = st.columns(
                [8, 1]
            )

            with top1:

                st.markdown(
                    """
                    <div style="
                        padding:14px 18px 0;
                        font-family:Quicksand,sans-serif;
                        font-weight:700;
                        color:#b79cff;
                        font-size:14px;
                    ">
                    🔍 Note Meaning
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with top2:

                with st.container(
                    key="note_close"
                ):

                    if st.button(
                        "✕",
                        key="note_close_btn",
                    ):

                        st.session_state.note_modal_open = False

                        st.session_state.selected_note_id = None

                        st.rerun()


            left, right = st.columns(
                2
            )


            # ------------------------------------------------------
            # ORIGINAL NOTE
            # ------------------------------------------------------

            with left:

                st.markdown(
                    f"""
                    <div class="cw-note-left">
                        {_note_html(row)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # ------------------------------------------------------
            # AI ANALYSIS
            # ------------------------------------------------------

            with right:

                already_analyzed = (
                    str(
                        row.get(
                            "emotion",
                            ""
                        )
                    ).strip()
                    != ""
                )


                if already_analyzed:

                    result = (
                        get_or_analyze_emotion(
                            row
                        )
                    )

                else:

                    with st.spinner(
                        "✨ AI is reading this note..."
                    ):

                        result = (
                            get_or_analyze_emotion(
                                row
                            )
                        )


                color_name = str(
                    row.get(
                        "note_color",
                        ""
                    )
                )


                if color_name not in NOTE_COLORS:

                    try:

                        note_id = int(
                            float(
                                row["id"]
                            )
                        )

                    except Exception:

                        note_id = 0


                    color_name = (
                        FALLBACK_COLORS[
                            note_id
                            % len(
                                FALLBACK_COLORS
                            )
                        ]
                    )


                accent = (
                    NOTE_COLORS[
                        color_name
                    ][2]
                )


                emotion = result.get(
                    "emotion",
                    "Neutral"
                )

                secondary = result.get(
                    "secondary_emotion",
                    ""
                )

                emotion_icon = (
                    EMOTION_CATEGORIES.get(
                        emotion,
                        "😐"
                    )
                )

                secondary_icon = (
                    EMOTION_CATEGORIES.get(
                        secondary,
                        ""
                    )
                )


                secondary_html = ""

                if secondary:

                    secondary_html = (
                        f'<div '
                        f'class="cw-emotion-secondary">'
                        f'Secondary: '
                        f'{secondary_icon} '
                        f'{_clean(secondary)}'
                        f'</div>'
                    )


                st.markdown(
                    f"""
                    <div
                        class="cw-note-right"
                        style="
                            border-top:
                            3px solid {accent};
                        "
                    >

                        <div class="cw-ai-badge">
                            ✨ AI Interpretation
                        </div>

                        <div class="cw-meaning-label">
                            The Meaning
                        </div>

                        <div class="cw-meaning-text">
                            {_clean(
                                result.get(
                                    "meaning",
                                    ""
                                )
                            )}
                        </div>

                        <div class="cw-emotion-label">
                            Emotion Review
                        </div>

                        <div class="cw-emotion-card">

                            <div class="cw-emotion-badge">
                                {emotion_icon}
                                {_clean(emotion)}
                            </div>

                            {secondary_html}

                            <div class="cw-intensity">
                                Intensity:
                                {_clean(
                                    result.get(
                                        "intensity",
                                        "Mild"
                                    )
                                )}
                            </div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==========================================================================
# CHAT BUBBLE ACCENT
# ==========================================================================

def _guess_note_accent(
    answer: str,
    df: pd.DataFrame,
):

    lower_answer = (
        answer.lower()
    )


    for _, row in df.iterrows():

        sender = str(
            row.get(
                "sender_name",
                ""
            )
        ).strip()


        if (
            not sender
            or sender.lower()
            == "anonymous"
        ):

            continue


        if (
            sender.lower()
            in lower_answer
        ):

            color = row.get(
                "note_color",
                ""
            )


            if color in NOTE_COLORS:

                return (
                    NOTE_COLORS[
                        color
                    ][2]
                )


    return None


# ==========================================================================
# APP TITLE
# ==========================================================================

st.title(
    "🎓 Confession Wall"
)

st.caption(
    "A place to share thoughts, messages, "
    "and moments with the people around you."
)


# ==========================================================================
# SESSION STATE
# ==========================================================================

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


if "active_tab" not in st.session_state:

    st.session_state.active_tab = (
        "✏️ Leave a Message"
    )


# ==========================================================================
# CHAT / NOTE MODAL
# ==========================================================================

if st.session_state.note_modal_open:

    st.session_state.chat_open = False

    render_note_modal()


elif not st.session_state.chat_open:

    # --------------------------------------------------------------
    # CLOSED CHAT BUTTON
    # --------------------------------------------------------------

    with st.container(
        key="chat_toggle_btn"
    ):

        if st.button(
            "💬",
            key="chat_toggle"
        ):

            st.session_state.chat_open = True

            st.rerun()


    components.html(
        f"""
        <script>
        {_CHAT_HEAD_JS}
        </script>
        """,
        height=0,
    )


else:

    # --------------------------------------------------------------
    # OPEN CHAT
    # --------------------------------------------------------------

    with st.container(
        key="chat_overlay"
    ):

        # ----------------------------------------------------------
        # OUTSIDE CLICK
        # ----------------------------------------------------------

        with st.container(
            key="chat_backdrop_close"
        ):

            if st.button(
                "close backdrop",
                key="chat_backdrop_btn",
            ):

                st.session_state.chat_open = False

                st.rerun()


        # ----------------------------------------------------------
        # CHAT MODAL
        # ----------------------------------------------------------

        with st.container(
            key="chat_modal"
        ):

            # ------------------------------------------------------
            # HEADER
            # ------------------------------------------------------

            with st.container(
                key="chat_header"
            ):

                hcol1, hcol2 = st.columns(
                    [6, 1]
                )


                with hcol1:

                    st.markdown(
                        """
                        <div class="cw-header-title">
                            AI Assistant
                        </div>

                        <div class="cw-header-sub">
                            Ask anything about the
                            messages posted on the wall.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                with hcol2:

                    if st.button(
                        "✕",
                        key="chat_close",
                    ):

                        st.session_state.chat_open = False

                        st.rerun()


            # ------------------------------------------------------
            # CHAT BODY
            # ------------------------------------------------------

            body_html = (
                '<div class="cw-body" '
                'id="cw-body">'
            )


            if not st.session_state.chat_history:

                body_html += """
                <div class="cw-row ai">

                    <div>

                        <div class="cw-label">
                            AI
                        </div>

                        <div class="cw-bubble ai">
                            Ask me anything about
                            the messages posted
                            on the wall!
                        </div>

                    </div>

                </div>
                """


            for turn in (
                st.session_state.chat_history
            ):

                is_ai = (
                    turn["role"]
                    != "user"
                )

                side = (
                    "ai"
                    if is_ai
                    else "user"
                )

                label = (
                    "AI"
                    if is_ai
                    else "You"
                )

                accent = (
                    turn.get("accent")
                    if is_ai
                    else None
                )


                if accent:

                    style_attr = (
                        f' style="'
                        f'background:{accent};'
                        f'color:#1a1a1a;'
                        f'"'
                    )

                else:

                    style_attr = ""


                safe_content = html.escape(
                    str(
                        turn["content"]
                    )
                )


                body_html += (
                    f'<div '
                    f'class="cw-row {side}">'
                    f'<div>'
                    f'<div class="cw-label">'
                    f'{label}'
                    f'</div>'
                    f'<div '
                    f'class="cw-bubble {side}"'
                    f'{style_attr}>'
                    f'{safe_content}'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )


            # ------------------------------------------------------
            # TYPING
            # ------------------------------------------------------

            if (
                st.session_state.chat_pending
            ):

                body_html += """
                <div class="cw-row ai">

                    <div>

                        <div class="cw-label">
                            AI
                        </div>

                        <div class="cw-typing">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>

                    </div>

                </div>
                """


            body_html += "</div>"


            st.markdown(
                body_html,
                unsafe_allow_html=True,
            )


            # ------------------------------------------------------
            # AUTO SCROLL
            # ------------------------------------------------------

            components.html(
                """
                <script>

                var body =
                    window.parent.document
                    .getElementById(
                        'cw-body'
                    );

                if (body) {

                    body.scrollTop =
                        body.scrollHeight;
                }

                </script>
                """,
                height=0,
            )


            # ------------------------------------------------------
            # PROCESS QUESTION
            # ------------------------------------------------------

            if (
                st.session_state.chat_pending
            ):

                question = (
                    st.session_state.chat_pending
                )

                full_df = load_data()


                answer = chatbot_answer(
                    question,
                    full_df,
                    st.session_state.chat_history,
                )


                accent = (
                    _guess_note_accent(
                        answer,
                        full_df,
                    )
                )


                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "accent": accent,
                    }
                )


                st.session_state.chat_pending = None

                st.rerun()


            # ------------------------------------------------------
            # CHAT INPUT
            # ------------------------------------------------------

            with st.container(
                key="chat_inputbar"
            ):

                with st.form(
                    "chat_form",
                    clear_on_submit=True,
                ):

                    colA, colB = st.columns(
                        [5, 1]
                    )


                    with colA:

                        user_q = st.text_input(
                            "msg",
                            placeholder=(
                                "Type a message..."
                            ),
                            label_visibility=(
                                "collapsed"
                            ),
                        )


                    with colB:

                        send = (
                            st.form_submit_button(
                                "➤"
                            )
                        )


                if (
                    send
                    and user_q.strip()
                ):

                    st.session_state.chat_history.append(
                        {
                            "role": "user",
                            "content": user_q.strip(),
                        }
                    )

                    st.session_state.chat_pending = (
                        user_q.strip()
                    )

                    st.rerun()


# ==========================================================================
# NAVIGATION
# ==========================================================================

TAB_OPTIONS = [
    "✏️ Leave a Message",
    "📝 Browse Wall",
    "📊 Insights",
]


st.markdown(
    '<div class="cw-nav">',
    unsafe_allow_html=True,
)


st.session_state.active_tab = st.radio(
    "nav",
    TAB_OPTIONS,
    index=TAB_OPTIONS.index(
        st.session_state.active_tab
    ),
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
)


st.markdown(
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================================================
# TAB 1 — LEAVE A MESSAGE
# ==========================================================================

if (
    st.session_state.active_tab
    == TAB_OPTIONS[0]
):

    st.subheader(
        "Write your message"
    )


    with st.form(
        "new_message_form",
        clear_on_submit=True,
    ):

        target_name = st.text_input(
            "Recipient",
            placeholder=(
                "🔍 Search recipient..."
                " (e.g. Sir John, Room 204, Canteen staff)"
            ),
        )


        message = st.text_area(
            "Message",
            height=150,
            placeholder=(
                "Write what's on your mind..."
            ),
        )


        sender_name = st.text_input(
            "From (optional — leave blank to stay Anonymous)"
        )


        with st.expander(
            "🎨 Add Color"
        ):

            note_color = st.radio(
                "Choose your note color",
                COLOR_NAMES,
                format_func=lambda name:
                    f"{NOTE_COLORS[name][0]} {name}",
                horizontal=True,
                label_visibility="collapsed",
            )


        submitted = st.form_submit_button(
            "Post to the Wall"
        )


    if submitted:

        if not target_name.strip():

            st.warning(
                "Please type who or what "
                "this message is for."
            )

        elif not message.strip():

            st.warning(
                "Please write your message."
            )

        else:

            with st.spinner(
                "Analyzing your message..."
            ):

                analysis = analyze_message(
                    message.strip(),
                    "Recipient",
                    target_name.strip(),
                )


            new_row = {

                "timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),

                "target_type":
                    "Recipient",

                "target_name":
                    target_name.strip(),

                "message":
                    message.strip(),

                "sender_name":
                    (
                        sender_name.strip()
                        if sender_name.strip()
                        else "Anonymous"
                    ),

                "sentiment":
                    analysis.get(
                        "sentiment",
                        "Neutral"
                    ),

                "emoji_tag":
                    analysis.get(
                        "emoji_tag",
                        ""
                    ),

                "keywords":
                    ", ".join(
                        analysis.get(
                            "keywords",
                            []
                        )
                    ),

                "suggestion":
                    analysis.get(
                        "suggestion",
                        ""
                    ),

                "views":
                    0,

                "note_color":
                    note_color,

                "meaning":
                    "",

                "emotion":
                    "",

                "secondary_emotion":
                    "",

                "intensity":
                    "",

                "analyzed_at":
                    "",
            }


            saved = save_message(
                new_row
            )


            if saved:

                st.success(
                    "Your message has been posted "
                    "to the wall. 🎓"
                )

                render_card(
                    new_row
                )


# ==========================================================================
# TAB 2 — BROWSE WALL
# ==========================================================================

elif (
    st.session_state.active_tab
    == TAB_OPTIONS[1]
):

    st.subheader(
        "The Wall"
    )

    st.caption(
        "Tap any paper to open its meaning "
        "and emotion review."
    )


    df = load_data()


    name_filter = st.text_input(
        "Search by recipient name",
        placeholder=(
            "🔍 Search recipient..."
        ),
    )


    filtered = df.copy()


    if name_filter.strip():

        filtered = filtered[
            filtered[
                "target_name"
            ]
            .astype(str)
            .str.contains(
                name_filter.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]


    if filtered.empty:

        st.info(
            "No messages match your search yet."
        )

    else:

        filtered["_sort_id"] = (
            pd.to_numeric(
                filtered["id"],
                errors="coerce"
            )
            .fillna(0)
        )


        rows = (
            filtered
            .sort_values(
                "_sort_id",
                ascending=False,
            )
            .drop(
                columns=[
                    "_sort_id"
                ]
            )
            .to_dict(
                orient="records"
            )
        )


        cols = st.columns(3)


        for i, note_row in enumerate(
            rows
        ):

            with cols[
                i % 3
            ]:

                try:

                    note_id = int(
                        float(
                            note_row["id"]
                        )
                    )

                except Exception:

                    note_id = i + 1


                with st.container(
                    key=f"note_wrap_{note_id}"
                ):

                    st.markdown(
                        _note_html(
                            note_row
                        ),
                        unsafe_allow_html=True,
                    )


                    if st.button(
                        "open note",
                        key=f"view_note_{note_id}",
                    ):

                        st.session_state.selected_note_id = (
                            note_id
                        )

                        st.session_state.note_modal_open = True

                        st.session_state.chat_open = False

                        st.rerun()


# ==========================================================================
# TAB 3 — INSIGHTS
# ==========================================================================

elif (
    st.session_state.active_tab
    == TAB_OPTIONS[2]
):

    st.subheader(
        "💭 Confession Wall Insights"
    )

    st.caption(
        "A visual summary of what people "
        "are sharing on the wall."
    )


    df = load_data()


    total = len(df)


    analyzed = df[
        df["sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ]


    emotion_analyzed = df[
        df["emotion"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ]


    # ==============================================================
    # KPI DATA
    # ==============================================================

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )


    if not df.empty:

        today_count = int(
            df["timestamp"]
            .astype(str)
            .str.startswith(
                today
            )
            .sum()
        )

    else:

        today_count = 0


    topic_series = (
        analyzed[
            "keywords"
        ]
        .dropna()
        .astype(str)
        .str.split(", ")
        .explode()
    )


    topic_series = topic_series[
        topic_series
        .str.strip()
        != ""
    ]


    active_topics = (
        int(
            topic_series.nunique()
        )
        if not topic_series.empty
        else 0
    )


    analyzed_count = len(
        emotion_analyzed
    )


    # ==============================================================
    # INSIGHTS CSS
    # ==============================================================

    st.markdown(
        """
        <style>

        .ins-grid {

            display:
                grid;

            grid-template-columns:
                repeat(4,1fr);

            gap:
                14px;

            margin:
                10px 0 24px;
        }

        .ins-card {

            background:
                linear-gradient(
                    145deg,
                    #21192d,
                    #17121f
                );

            border:
                1px solid
                rgba(255,255,255,.08);

            border-radius:
                18px;

            padding:
                18px;
        }

        .ins-label {

            color:
                #a99db7;

            font:
                700 11px
                Quicksand,
                sans-serif;

            text-transform:
                uppercase;

            letter-spacing:
                1px;
        }

        .ins-value {

            color:
                #fff;

            font:
                700 30px
                Quicksand,
                sans-serif;

            margin-top:
                6px;
        }

        .ins-small {

            color:
                #7f748c;

            font:
                400 11px
                Quicksand,
                sans-serif;

            margin-top:
                4px;
        }

        .ins-section {

            background:
                #19141f;

            border:
                1px solid
                rgba(255,255,255,.07);

            border-radius:
                18px;

            padding:
                18px;

            margin:
                14px 0;
        }

        .ins-title {

            color:
                #f4edf8;

            font:
                700 16px
                Quicksand,
                sans-serif;

            margin-bottom:
                3px;
        }

        .ins-sub {

            color:
                #8f8499;

            font:
                400 11px
                Quicksand,
                sans-serif;
        }

        .emotion-row {

            display:
                flex;

            align-items:
                center;

            justify-content:
                space-between;

            padding:
                9px 0;

            border-bottom:
                1px solid
                rgba(255,255,255,.05);

            color:
                #ddd4e4;

            font:
                600 13px
                Quicksand,
                sans-serif;
        }

        .emotion-count {

            color:
                #a99db7;

            font-size:
                12px;
        }

        @media(max-width:800px) {

            .ins-grid {
                grid-template-columns:
                    repeat(2,1fr);
            }
        }

        @media(max-width:480px) {

            .ins-grid {
                grid-template-columns:
                    1fr;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


    # ==============================================================
    # KPI CARDS
    # ==============================================================

    st.markdown(
        f"""
        <div class="ins-grid">

            <div class="ins-card">

                <div class="ins-label">
                    💬 Total Notes
                </div>

                <div class="ins-value">
                    {total}
                </div>

                <div class="ins-small">
                    all posts on the wall
                </div>

            </div>


            <div class="ins-card">

                <div class="ins-label">
                    🕐 Posted Today
                </div>

                <div class="ins-value">
                    {today_count}
                </div>

                <div class="ins-small">
                    based on the current date
                </div>

            </div>


            <div class="ins-card">

                <div class="ins-label">
                    😊 Analyzed
                </div>

                <div class="ins-value">
                    {analyzed_count}
                </div>

                <div class="ins-small">
                    notes with emotion review
                </div>

            </div>


            <div class="ins-card">

                <div class="ins-label">
                    🔥 Topics
                </div>

                <div class="ins-value">
                    {active_topics}
                </div>

                <div class="ins-small">
                    distinct extracted themes
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ==============================================================
    # EMPTY DATA
    # ==============================================================

    if df.empty:

        st.info(
            "No messages have been posted yet."
        )


    else:

        # ==========================================================
        # SENTIMENT + TOPICS
        # ==========================================================

        left, right = st.columns(
            2
        )


        with left:

            st.markdown(
                """
                <div class="ins-section">

                    <div class="ins-title">
                        💗 Message Tone
                    </div>

                    <div class="ins-sub">
                        Based on the initial AI
                        analysis when notes were posted.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            if analyzed.empty:

                st.info(
                    "No analyzed notes yet."
                )

            else:

                counts = (
                    analyzed[
                        "sentiment"
                    ]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "sentiment",
                    "count",
                ]


                fig = px.pie(
                    counts,
                    names="sentiment",
                    values="count",
                    hole=.55,
                )


                fig.update_layout(
                    margin=dict(
                        l=10,
                        r=10,
                        t=10,
                        b=10,
                    ),
                    height=300,
                    paper_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),
                    plot_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),
                    font_color="#ddd4e4",
                    legend_title_text="",
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )


        with right:

            st.markdown(
                """
                <div class="ins-section">

                    <div class="ins-title">
                        🔥 What People Talk About
                    </div>

                    <div class="ins-sub">
                        Top themes extracted
                        from posted notes.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            if topic_series.empty:

                st.info(
                    "No theme data yet."
                )

            else:

                kw = (
                    topic_series
                    .value_counts()
                    .head(8)
                    .sort_values()
                )


                fig2 = px.bar(
                    x=kw.values,
                    y=kw.index,
                    orientation="h",
                )


                fig2.update_layout(
                    margin=dict(
                        l=10,
                        r=10,
                        t=10,
                        b=10,
                    ),
                    height=300,
                    paper_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),
                    plot_bgcolor=(
                        "rgba(0,0,0,0)"
                    ),
                    font_color="#ddd4e4",
                    xaxis_title="mentions",
                    yaxis_title="",
                )


                st.plotly_chart(
                    fig2,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )


        # ==========================================================
        # ACTIVITY OVER TIME
        # ==========================================================

        st.markdown(
            """
            <div class="ins-section">

                <div class="ins-title">
                    📈 Wall Activity
                </div>

                <div class="ins-sub">
                    How many notes were posted over time.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        dates = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        ).dt.date


        activity = (
            dates
            .value_counts()
            .sort_index()
            .reset_index()
        )


        activity.columns = [
            "date",
            "notes",
        ]


        if not activity.empty:

            fig3 = px.line(
                activity,
                x="date",
                y="notes",
                markers=True,
            )


            fig3.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10,
                ),
                height=260,
                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                plot_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                font_color="#ddd4e4",
                xaxis_title="",
                yaxis_title="notes",
            )


            st.plotly_chart(
                fig3,
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
            )


        # ==========================================================
        # EMOTIONS + WALL PULSE
        # ==========================================================

        left, right = st.columns(
            2
        )


        with left:

            st.markdown(
                """
                <div class="ins-section">

                    <div class="ins-title">
                        😊 Emotions on the Wall
                    </div>

                    <div class="ins-sub">
                        Emotion data saved by
                        the individual note analyzer.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            if emotion_analyzed.empty:

                st.info(
                    "Open notes on Browse Wall "
                    "to generate emotion reviews."
                )

            else:

                emotions = (
                    emotion_analyzed[
                        "emotion"
                    ]
                    .value_counts()
                )


                for (
                    emotion,
                    count,
                ) in emotions.head(7).items():

                    icon = (
                        EMOTION_CATEGORIES.get(
                            emotion,
                            "😐"
                        )
                    )


                    percentage = int(
                        round(
                            count
                            /
                            len(
                                emotion_analyzed
                            )
                            * 100
                        )
                    )


                    st.markdown(
                        f"""
                        <div class="emotion-row">

                            <span>
                                {icon}
                                {_clean(emotion)}
                            </span>

                            <span class="emotion-count">
                                {count}
                                ·
                                {percentage}%
                            </span>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


        with right:

            st.markdown(
                """
                <div class="ins-section">

                    <div class="ins-title">
                        💡 Wall Pulse
                    </div>

                    <div class="ins-sub">
                        A quick read from
                        the data currently available.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


            if analyzed.empty:

                st.info(
                    "More analyzed notes are needed "
                    "for wall insights."
                )

            else:

                top_sentiment = (
                    analyzed[
                        "sentiment"
                    ]
                    .value_counts()
                    .idxmax()
                )


                if not topic_series.empty:

                    top_theme = (
                        topic_series
                        .value_counts()
                        .idxmax()
                    )

                else:

                    top_theme = (
                        "No dominant theme yet"
                    )


                st.markdown(
                    f"- **Most common tone:** "
                    f"{_clean(top_sentiment)}"
                )

                st.markdown(
                    f"- **Most mentioned theme:** "
                    f"{_clean(top_theme)}"
                )

                st.markdown(
                    f"- **Notes analyzed:** "
                    f"{len(analyzed)} of {total}"
                )


                if not emotion_analyzed.empty:

                    top_emotion = (
                        emotion_analyzed[
                            "emotion"
                        ]
                        .value_counts()
                        .idxmax()
                    )


                    st.markdown(
                        f"- **Most common emotion:** "
                        f"{EMOTION_CATEGORIES.get(top_emotion, '😐')} "
                        f"{_clean(top_emotion)}"
                    )


        # ==========================================================
        # SUGGESTIONS
        # ==========================================================

        suggestions = analyzed[
            analyzed[
                "suggestion"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            != ""
        ]


        st.markdown(
            """
            <div class="ins-section">

                <div class="ins-title">
                    💡 Messages That Include Suggestions
                </div>

                <div class="ins-sub">
                    Suggestions or concerns
                    detected from individual notes.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        if suggestions.empty:

            st.info(
                "No suggestions have been extracted yet."
            )

        else:

            for _, row in (
                suggestions
                .head(8)
                .iterrows()
            ):

                st.markdown(
                    f"- **{_clean(row['target_name'])}:** "
                    f"{_clean(row['suggestion'])}"
                )


        # ==========================================================
        # DATASET PREVIEW
        # ==========================================================

        with st.expander(
            "📊 Dataset Preview"
        ):

            st.caption(
                "This is the current dataset used by "
                "the Confession Wall application."
            )

            preview_columns = [
                "id",
                "timestamp",
                "target_name",
                "message",
                "sender_name",
                "sentiment",
                "keywords",
                "suggestion",
            ]


            available_columns = [
                column
                for column in preview_columns
                if column in df.columns
            ]


            st.dataframe(
                df[
                    available_columns
                ],
                use_container_width=True,
                hide_index=True,
            )


        # ==========================================================
        # DATA MANAGEMENT
        # ==========================================================

        with st.expander(
            "💾 Data Management"
        ):

            st.caption(
                "Download a backup of the current "
                "wall data before changing or "
                "redeploying the app."
            )


            st.download_button(
                "Download messages.csv",
                data=df.to_csv(
                    index=False
                ),
                file_name=(
                    "messages_backup.csv"
                ),
                mime="text/csv",
            )
