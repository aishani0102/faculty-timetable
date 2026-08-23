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


