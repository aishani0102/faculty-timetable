# Faculty Workload & Timetable Agent

RAG-based LangChain agent over Postgres (pgvector), deployed to Hugging
Face Spaces. No fine-tuning, no paid services.

## What it does

- **Availability** — who's teaching, and who's free, at a given day/time.
- **Workload** — a professor's or department's hours, checked against the
  12-hour policy.
- **Clash detection** — finds double-booked rooms or double-booked
  faculty in the timetable and suggests a fix for each.
- **Policy lookup** — answers questions about scheduling rules via RAG
  over the policy text.

The agent (LangChain, `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`)
reads the question, picks the right tool itself, and extracts arguments
like day, time, or department straight from your sentence, no separate
form fields needed.

## What's in here

```
data/                sample CSVs and policy text — replace with your real data
db/schema.sql         run once against your database
ingest.py             loads the CSVs and generates policy embeddings
tools.py               the four underlying tools: timetable, report, clash, RAG
agent.py               LangChain agent that wires the tools to a HF-hosted model
app.py                 Streamlit frontend
requirements.txt
.env.example
```

## Running it

1. Create a free Supabase (or Neon) project, run `db/schema.sql` in its SQL
   editor.
2. Put your real data in `data/`, matching the existing column headers.
3. `pip install -r requirements.txt`
4. `export DATABASE_URL=...` (your connection string) and run
   `python ingest.py`.
5. Get a free "Read" token from huggingface.co, `export HF_TOKEN=...`.
6. `streamlit run app.py` to test locally.
7. To deploy: create a Space (SDK: Streamlit, free CPU tier), upload
   `app.py`, `agent.py`, `tools.py`, `requirements.txt`, add `DATABASE_URL`
   and `HF_TOKEN` as repository secrets.

The included sample data ships with one deliberate room clash (Tuesday,
Room 305) so the clash-detection tool has something to find on a first run.

## Known limitations

- Free/small models occasionally deviate from the agent's expected
  output format. `handle_parsing_errors=True` and a `max_iterations`
  cap keep this from hanging; if you see frequent failures, try a
  larger instruct model in `agent.py`.
- The free Hugging Face Inference API can have a cold start of up to
  ~30 seconds if the model hasn't been called recently.
- `ingest.py` re-seeds `timetable` and `policies` on every run
  (delete-then-insert). Fine at this scale; swap for an upsert if the
  dataset grows.
