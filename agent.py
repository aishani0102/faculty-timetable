import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from tools import timetable_tool, report_tool, clash_tool, rag_tool

HF_TOKEN = os.environ["HF_TOKEN"]
MODEL = "Qwen/Qwen2.5-7B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=MODEL,
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256,
    temperature=0.1,
)
chat_model = ChatHuggingFace(llm=llm)


class TimetableInput(BaseModel):
    day: str = Field(description="Day of the week, e.g. Tuesday")
    time: str = Field(description="24-hour time, e.g. 14:00")


class ReportInput(BaseModel):
    department: str = Field(default="", description="Department code, e.g. CSE. Leave blank if not relevant.")
    faculty_name: str = Field(default="", description="Professor's name, e.g. Sharma. Leave blank if not relevant.")


class RagInput(BaseModel):
    query: str = Field(description="The policy question to search for")


def _timetable(day: str, time: str) -> str:
    data = timetable_tool(day, time)
    return f"Busy: {data['busy']}. Free: {data['free']}."


def _report(department: str = "", faculty_name: str = "") -> str:
    rows = report_tool(department=department or None, faculty_name=faculty_name or None)
    total = sum(row[-1] for row in rows) if rows else 0
    return f"Rows: {rows}. Total hours: {total}."


def _clash(query: str = "") -> str:
    clashes = clash_tool()
    return f"Clashes: {clashes}" if clashes else "No clashes found in the current timetable."


def _rag(query: str) -> str:
    return " | ".join(rag_tool(query))


tools = [
    StructuredTool.from_function(
        func=_timetable,
        name="timetable_lookup",
        description="Find who is teaching, or who is free, at a given day and time.",
        args_schema=TimetableInput,
    ),
    StructuredTool.from_function(
        func=_report,
        name="workload_report",
        description="Get a professor's or a department's weekly teaching hours.",
        args_schema=ReportInput,
    ),
    StructuredTool.from_function(
        func=_clash,
        name="clash_detector",
        description="Find double-booked rooms or faculty in the timetable, with a suggested fix.",
    ),
    StructuredTool.from_function(
        func=_rag,
        name="policy_search",
        description="Look up university scheduling policies and rules.",
        args_schema=RagInput,
    ),
]

agent_executor = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,
    return_intermediate_steps=True,
)


def answer(query: str):
    result = agent_executor.invoke({"input": query})
    return result["output"], result.get("intermediate_steps", [])
