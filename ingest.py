"""Loads data/*.csv and data/policies.txt into Postgres. Run after schema.sql.

    export DATABASE_URL="postgresql://user:password@host:port/dbname"
    python ingest.py
"""

import os
import csv

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ["DATABASE_URL"]


def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def load_faculty(conn, path="data/faculty_workload.csv"):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [
            (r["faculty_id"], r["name"], r["department"], r["course"], int(r["hours_per_week"]))
            for r in reader
        ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO faculty_workload (faculty_id, name, department, course, hours_per_week)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (faculty_id) DO UPDATE SET
                name = EXCLUDED.name,
                department = EXCLUDED.department,
                course = EXCLUDED.course,
                hours_per_week = EXCLUDED.hours_per_week
            """,
            rows,
        )
    conn.commit()
    print(f"Loaded {len(rows)} faculty rows")


def load_timetable(conn, path="data/timetable.csv"):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [
            (r["day"], r["start_time"], r["end_time"], r["course"], r["faculty_id"], r["room"])
            for r in reader
        ]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM timetable")  # simple re-seed, fine for a small dataset
        cur.executemany(
            """
            INSERT INTO timetable (day, start_time, end_time, course, faculty_id, room)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()
    print(f"Loaded {len(rows)} timetable rows")


def load_policies(conn, path="data/policies.txt"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    embeddings = model.encode(lines, normalize_embeddings=True)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM policies")
        for text, emb in zip(lines, embeddings):
            cur.execute(
                "INSERT INTO policies (content, embedding) VALUES (%s, %s)",
                (text, emb),
            )
    conn.commit()
    print(f"Loaded {len(lines)} policy embeddings")


if __name__ == "__main__":
    conn = get_connection()
    load_faculty(conn)
    load_timetable(conn)
    load_policies(conn)
    conn.close()
    print("Done.")
