import datetime
import os
import re

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ["DATABASE_URL"]
_embedder = None

VALID_DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
TIME_FORMATS = ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I %p", "%I:%M%p"]


def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def normalize_day(day: str) -> str:
    normalized = day.strip().capitalize()
    if normalized not in VALID_DAYS:
        raise ValueError(f"'{day}' is not a recognized day of the week (expected e.g. 'Tuesday')")
    return normalized


def normalize_time(time_str: str) -> str:
    cleaned = time_str.strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.datetime.strptime(cleaned, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    raise ValueError(f"'{time_str}' is not a recognized time (expected 24-hour format, e.g. '14:00')")


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def timetable_tool(day: str, time: str):
    """Who is teaching, and who is free, at a given day and time."""
    day = normalize_day(day)
    time = normalize_time(time)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.name, t.course, t.room
            FROM timetable t JOIN faculty_workload f USING (faculty_id)
            WHERE t.day = %s AND t.start_time <= %s AND t.end_time > %s
            """,
            (day, time, time),
        )
        busy = cur.fetchall()

        cur.execute(
            """
            SELECT name FROM faculty_workload
            WHERE faculty_id NOT IN (
                SELECT faculty_id FROM timetable
                WHERE day = %s AND start_time <= %s AND end_time > %s
            )
            """,
            (day, time, time),
        )
        free = [row[0] for row in cur.fetchall()]
    conn.close()
    return {"busy": busy, "free": free}


def normalize_room(room: str) -> str:
    return re.sub(r"(?i)^room\s*", "", room.strip()).strip()


def room_status(room: str, day: str, time: str):
    """Who is in a given room at a given day and time, or whether it's free. Flags a double-booking if more than one class is scheduled there."""
    room = normalize_room(room)
    day = normalize_day(day)
    time = normalize_time(time)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.name, t.course
            FROM timetable t JOIN faculty_workload f USING (faculty_id)
            WHERE t.room = %s AND t.day = %s AND t.start_time <= %s AND t.end_time > %s
            """,
            (room, day, time, time),
        )
        occupants = cur.fetchall()
    conn.close()
    return {"room": room, "occupants": occupants, "clash": len(occupants) > 1}


TITLE_PREFIX_RE = re.compile(r"^(dr|prof|mr|mrs|ms)\.?\s*", re.IGNORECASE)


def _strip_title(name: str) -> str:
    return TITLE_PREFIX_RE.sub("", name.strip()).strip()


def report_tool(department: str | None = None, faculty_name: str | None = None):
    """Workload rows for one professor, one department, or totals per department."""
    conn = get_connection()
    with conn.cursor() as cur:
        if faculty_name:
            cur.execute(
                "SELECT name, course, hours_per_week FROM faculty_workload WHERE name ILIKE %s",
                (f"%{_strip_title(faculty_name)}%",),
            )
        elif department:
            cur.execute(
                "SELECT name, course, hours_per_week FROM faculty_workload WHERE department = %s",
                (department,),
            )
        else:
            cur.execute(
                "SELECT department, SUM(hours_per_week) FROM faculty_workload GROUP BY department"
            )
        rows = cur.fetchall()
    conn.close()
    return rows


def _free_rooms(day: str, start: str, end: str):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT room FROM timetable
            WHERE room NOT IN (
                SELECT room FROM timetable
                WHERE day = %s AND start_time < %s AND end_time > %s
            )
            """,
            (day, end, start),
        )
        rooms = [row[0] for row in cur.fetchall()]
    conn.close()
    return rooms


def clash_tool():
    """Find double-booked rooms and double-booked faculty, with a suggested fix for each."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.day, a.room, a.start_time, a.end_time, a.course, b.course
            FROM timetable a
            JOIN timetable b ON a.day = b.day AND a.room = b.room AND a.id < b.id
            WHERE a.start_time < b.end_time AND b.start_time < a.end_time
            """
        )
        room_rows = cur.fetchall()

        cur.execute(
            """
            SELECT a.day, f.name, a.start_time, a.end_time, a.course, b.course
            FROM timetable a
            JOIN timetable b ON a.day = b.day AND a.faculty_id = b.faculty_id AND a.id < b.id
            JOIN faculty_workload f ON f.faculty_id = a.faculty_id
            WHERE a.start_time < b.end_time AND b.start_time < a.end_time
            """
        )
        faculty_rows = cur.fetchall()
    conn.close()

    clashes = []
    for day, room, start, end, course_a, course_b in room_rows:
        alt_rooms = [r for r in _free_rooms(day, str(start), str(end)) if r != room]
        clashes.append({
            "type": "room double-booking",
            "day": day,
            "time": f"{start}-{end}",
            "room": room,
            "courses": [course_a, course_b],
            "suggestion": f"move one class to room {alt_rooms[0]}" if alt_rooms else "no free room in the current data",
        })
    for day, name, start, end, course_a, course_b in faculty_rows:
        clashes.append({
            "type": "faculty double-booking",
            "day": day,
            "time": f"{start}-{end}",
            "faculty": name,
            "courses": [course_a, course_b],
        })
    return clashes


def rag_tool(query: str, k: int = 3):
    """Retrieve the k most relevant policy snippets for a natural-language query."""
    embedder = _get_embedder()
    query_vec = embedder.encode(query, normalize_embeddings=True)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM policies ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, k),
        )
        results = [row[0] for row in cur.fetchall()]
    conn.close()
    return results


def dataset_summary():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM faculty_workload")
        faculty_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT department) FROM faculty_workload")
        dept_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM timetable")
        slot_count = cur.fetchone()[0]
    conn.close()
    return {"faculty": faculty_count, "departments": dept_count, "slots": slot_count}
