# Confession Wall — Farewell Message Analyzer

A GenAI-powered Streamlit app for CS 315 Activity 3. Graduating students post
anonymous (or signed) farewell messages addressed to a teacher, official, or
location. GenAI analyzes each message for sentiment, theme, and extracts
suggestions for the school.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add
   your Hugging Face access token (get one free at
   https://huggingface.co/settings/tokens — "Read" access is enough).
3. Run the app:
   ```
   streamlit run app.py
   ```
## Features
1. Leave a Message
Free-text recipient field (not a fixed category) — students type who or what the note is for (e.g. "Sir John", "Room 204", "Canteen staff").
Optional sender name — blank posts stay Anonymous.
Note color picker — 12 colors (White, Yellow, Orange, Red, Pink, Purple, Blue, Teal, Green, Gray, Brown, Black) chosen before posting.
Every submitted message is sent to the AI model for automatic sentiment, keyword, and suggestion tagging (see AI Integration below) before it's saved.

3. Browse Wall
Digital bulletin-board layout — every note renders as an individual, slightly-tilted paper note in its chosen color, with a "tape" accent and a lined-paper texture.
Responsive masonry grid: 3 columns on desktop, 2 on tablet, 1 on mobile.
Search by recipient name — typing "Sir John" filters the wall to only notes addressed to Sir John.

5. Insights Dashboard (for the school)
Sentiment distribution pie chart across all analyzed notes.
Most common themes bar chart, built from AI-extracted keywords.
Suggestions extracted for the school — a running list of actionable complaints/suggestions the AI pulled out of student messages.
Data backup — one-click CSV download of the full dataset.

7. AI Assistant (floating chat)
Draggable, Messenger-style floating chat head — click to open, drag to reposition anywhere on screen.
Opens as a full-screen blurred backdrop with chat bubbles floating directly over it (AI on the left in grey, user on the right in blue; a bubble re-colors to match a note's own color when the AI is specifically discussing that note).

Typing indicator, auto-scroll to the latest message, and a "New chat" action that clears the conversation without touching any wall posts.
Understands every note currently on the wall as a structured record (author, recipient, color, timestamp, message) — not a single flattened wall of text — so it can correctly answer "who posted the latest note?", "what did [author] say?", or "summarize [author]'s message" without confusing one student's note for another's.


## How it maps to Activity 3

- **Dataset**: `data/messages.csv` — student farewell messages (self-collected)
- **Load/Clean**: Pandas in `load_data()` — strips blanks, fills missing sender names
- **GenAI Analysis**: `analyze_message()` — sentiment, emoji tag, keywords, suggestion extraction
- **Streamlit UI**: 3 tabs — submit message, browse wall (filterable), insights dashboard
- **Visualization**: Plotly pie chart (sentiment) + bar chart (top themes)
- **Chatbot**: "Ask about the Wall" box in the Insights tab
- **Next Goals**: add filters for target type/name (done), add chatbot (done) —
  could extend with graduation year/batch filters next

## Data Source / Citation

- Primary dataset: self-collected via the app's built-in submission form (Confession Wall feature), gathered from [your school/org name] graduating students.
- Seed/reference data (rows 6-13 in `messages.csv`) adapted in style from:
  - Miah, M.S.U. et al. (2023). *A Novel Dataset for Aspect-based Sentiment Analysis for Teacher Performance Evaluation.* Mendeley Data, V1. https://data.mendeley.com/datasets/b2yhc95rnx/1
  - He, J. (2020). *Big Data Set from RateMyProfessor.com for Professors' Teaching Evaluation.* Mendeley Data, V2. https://data.mendeley.com/datasets/fvtfjyvw7d/2 (also mirrored on Kaggle as "RateMyProfessor_Sample data")
