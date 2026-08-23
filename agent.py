import os

import psycopg2
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from tools import timetable_tool, report_tool, clash_tool, rag_tool, room_status

HF_TOKEN = os.environ["HF_TOKEN"]
MODEL = "Qwen/Qwen2.5-7B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=MODEL,
    provider="featherless-ai",
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


class RoomInput(BaseModel):
    room: str = Field(description="Room number, e.g. 305")
    day: str = Field(description="Day of the week, e.g. Tuesday")
    time: str = Field(description="24-hour time, e.g. 14:00")


def _timetable(day: str, time: str) -> str:
    try:
        data = timetable_tool(day, time)
    except ValueError as e:
        return f"Could not run that lookup: {e}"
    except psycopg2.Error:
        return "Could not run that lookup: the day or time given wasn't understood. Try e.g. day='Tuesday', time='14:00'."
    return f"Busy: {data['busy']}. Free: {data['free']}."


def _report(department: str = "", faculty_name: str = "") -> str:
    try:
        rows = report_tool(department=department or None, faculty_name=faculty_name or None)
    except psycopg2.Error:
        return "Could not run that report: the department or faculty name given wasn't understood."
    total = sum(row[-1] for row in rows) if rows else 0
    return f"Rows: {rows}. Total hours: {total}."


def _clash(query: str = "") -> str:
    try:
        clashes = clash_tool()
    except psycopg2.Error:
        return "Could not check for clashes due to a database error."
    return f"Clashes: {clashes}" if clashes else "No clashes found in the current timetable."


def _rag(query: str) -> str:
    try:
        return " | ".join(rag_tool(query))
    except psycopg2.Error:
        return "Could not search policies due to a database error."


def _room(room: str, day: str, time: str) -> str:
    try:
        data = room_status(room, day, time)
    except ValueError as e:
        return f"Could not check that room: {e}"
    except psycopg2.Error:
        return "Could not check that room: the room, day, or time given wasn't understood."
    if not data["occupants"]:
        return f"Room {data['room']} is free at that time."
    if data["clash"]:
        return f"Room {data['room']} is double-booked at that time: {data['occupants']}."
    return f"Room {data['room']} is occupied: {data['occupants']}."


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
    StructuredTool.from_function(
        func=_room,
        name="room_status",
        description="Check who is in a specific room at a given day and time, or whether it's free. Use this for any question about a room number.",
        args_schema=RoomInput,
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
