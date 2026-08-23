CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faculty_workload (
    faculty_id     TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    department     TEXT,
    course         TEXT,
    hours_per_week INT
);

CREATE TABLE IF NOT EXISTS timetable (
    id          SERIAL PRIMARY KEY,
    day         TEXT NOT NULL,
    start_time  TIME NOT NULL,
    end_time    TIME NOT NULL,
    course      TEXT,
    faculty_id  TEXT REFERENCES faculty_workload(faculty_id),
    room        TEXT
);

-- 384 dimensions to match the all-MiniLM-L6-v2 embedding model used in ingest.py
CREATE TABLE IF NOT EXISTS policies (
    id        SERIAL PRIMARY KEY,
    content   TEXT NOT NULL,
    embedding VECTOR(384)
);

CREATE INDEX IF NOT EXISTS policies_embedding_idx
    ON policies USING hnsw (embedding vector_cosine_ops);
