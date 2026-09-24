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
   your OpenAI API key.
3. Run the app:
   ```
   streamlit run app.py
   ```

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

Dataset self-collected from [your school/org name] graduating students via a
Google Form, [collection date]. Cite this in your submission document.