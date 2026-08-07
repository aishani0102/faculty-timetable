import streamlit as st

from agent import answer
from tools import dataset_summary

st.set_page_config(page_title="Faculty Workload & Timetable Agent", page_icon="🗓️", layout="centered")

st.markdown(
    """
    <style>
    div.stButton > button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset")
    try:
        stats = dataset_summary()
        st.metric("Faculty", stats["faculty"])
        st.metric("Departments", stats["departments"])
        st.metric("Weekly slots", stats["slots"])
    except Exception:
        st.caption("Set DATABASE_URL and run ingest.py to see dataset stats here.")
    st.divider()
    st.caption("LangChain agent over Postgres + pgvector. No fine-tuning.")

st.title("🗓️ Faculty Workload & Timetable Agent")
st.caption("Ask about faculty availability, workload, or scheduling clashes, in plain English.")

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
        result, steps = answer(query)
    st.subheader("Answer")
    st.write(result)
    if steps:
        with st.expander("How the agent got there"):
            for action, observation in steps:
                st.markdown(f"**Tool:** `{action.tool}`  \n**Input:** `{action.tool_input}`  \n**Result:** {observation}")
                st.divider()
