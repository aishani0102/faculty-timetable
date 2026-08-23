import streamlit as st
import pandas as pd
import altair as alt

from agent import answer
from tools import dataset_summary, get_connection, report_tool, clash_tool

st.set_page_config(page_title="Faculty Workload & Timetable Agent", page_icon="🗓️", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper Functions for Data Visibility ---
@st.cache_data(ttl=60)  # Caches the data for 60 seconds so it doesn't query the DB on every click
def get_faculty_directory():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT name, department, course, hours_per_week FROM faculty_workload ORDER BY name")
        rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["Name", "Department", "Course", "Hours/Week"])


@st.cache_data(ttl=60)
def get_department_workload():
    # report_tool() with no args returns total hours per department
    rows = report_tool()
    return pd.DataFrame(rows, columns=["Department", "Total Hours"]).set_index("Department")
# ------------------------------------------

with st.sidebar:
    st.header("Dataset Overview")
    try:
        stats = dataset_summary()
        cols = st.columns(3)
        cols[0].metric("Faculty", stats["faculty"])
        cols[1].metric("Departments", stats["departments"])
        cols[2].metric("Slots", stats["slots"])

        st.divider()
        st.subheader("Faculty Directory")
        st.dataframe(get_faculty_directory(), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Workload by Department")

        # Custom chart that forces the Y-axis to start at 0
        df_chart = get_department_workload().reset_index()
        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X('Department', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Total Hours', scale=alt.Scale(domainMin=0))
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.caption("Set DATABASE_URL and run ingest.py to see dataset stats here.")

st.title("Faculty Workload & Timetable Agent")
st.caption("Ask about faculty availability, workload, or scheduling clashes, in plain English.")

# --- Passive Clash Alert System ---
try:
    clashes = clash_tool()
    if clashes:
        st.warning(f"⚠️ **Alert:** Detected {len(clashes)} scheduling clash(es) in the current timetable. Ask the agent to 'find the scheduling clashes' for details and fixes!")
except Exception:
    pass
# ----------------------------------

EXAMPLES = [
    "What is Prof. Sharma's workload this week?",
    "Which faculty is free on Tuesday at 2 PM?",
    "Summarize CSE department workload.",
    "Are there any scheduling clashes this week?",
]

if "query" not in st.session_state:
    st.session_state.query = ""

st.write("Try an example, or type your own below:")
cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.query = example

query = st.text_area("Your question", key="query", height=80)

if st.button("Ask", type="primary") and query:
    with st.spinner("Thinking..."):
        try:
            result, steps = answer(query)
        except Exception:
            st.error("Something went wrong answering that question. Try rephrasing it, e.g. with an explicit day and 24-hour time.")
            st.stop()
    st.subheader("Answer")
    st.write(result)
    if steps:
        with st.expander("How the agent got there"):
            for action, observation in steps:
                st.markdown(f"**Tool:** `{action.tool}`  \n**Input:** `{action.tool_input}`  \n**Result:** {observation}")
                st.divider()
