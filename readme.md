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
