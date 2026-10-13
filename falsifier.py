import json
import os
import re
import textwrap
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from report import build_pdf, steps_from_messages

load_dotenv()  # reads GROQ_API_KEY from the .env file next to this script
if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("GROQ_API_KEY not found. Put it in a .env file next to falsifier.py")

MODEL = "openai/gpt-oss-120b"  # change to any tool-capable model from list_models.py

# ---------- 1. Synthetic data with a planted truth ----------
# Real cause: from Jul 1, Cairo and Alexandria drop 25% in ALL categories.
# The Aug 1 price increase on Category A has NO effect.
rng = np.random.default_rng(42)
rows = []
for d in pd.date_range("2026-01-01", "2026-09-30", freq="D"):
    for cat in ["A", "B", "C"]:
        for region in ["Cairo", "Alexandria", "Giza"]:
            rev = rng.normal(1000, 80)
            if d >= pd.Timestamp("2026-07-01") and region in ("Cairo", "Alexandria"):
                rev *= 0.75
            rows.append((d, cat, region, rev))
DF = pd.DataFrame(rows, columns=["date", "category", "region", "revenue"])


def subset(filters):
    """filters can be a dict or a JSON string like '{"category": "A"}'."""
    if isinstance(filters, str):
        filters = json.loads(filters) if filters.strip() else {}
    d = DF
    for col, val in (filters or {}).items():
        d = d[d[col] == val]
    return d


def daily_avg(d, start, end):
    return d[d.date.between(start, end)].groupby("date").revenue.sum().mean()


# ---------- 2. Tools (pandas does all the math) ----------
@tool
def describe_data() -> str:
    """Return column names, date range, and the distinct categories and regions."""
    return (
        f"columns: {list(DF.columns)}\n"
        f"date range: {DF.date.min().date()} to {DF.date.max().date()}\n"
        f"categories: {sorted(DF.category.unique())}\n"
        f"regions: {sorted(DF.region.unique())}"
    )


@tool
def find_changepoint(filters: str) -> str:
    """Find the date where daily revenue for a slice shifted most (largest level
    change). filters is a JSON string such as '{"category": "A"}' (use '{}' for
    all data). Use this to learn WHEN a drop actually started."""
    s = subset(filters).groupby("date").revenue.sum()
    if len(s) < 40:
        return "Not enough data."
    best_i, best_score = None, 0
    for i in range(14, len(s) - 14):
        score = abs(s.iloc[i:].mean() - s.iloc[:i].mean()) / (s.std() + 1e-9)
        if score > best_score:
            best_i, best_score = i, score
    before, after = s.iloc[:best_i].mean(), s.iloc[best_i:].mean()
    return (f"{filters}: biggest shift starts {s.index[best_i].date()} "
            f"({before:.0f}/day -> {after:.0f}/day, {(after - before) / before * 100:+.1f}%)")


@tool
def breakdown_by(dimension: str, before_start: str, before_end: str,
                 after_start: str, after_end: str, filters: str = "{}") -> str:
    """Compare average DAILY revenue between two periods for EVERY value of a
    dimension in one call. dimension is 'category' or 'region'. Dates are
    YYYY-MM-DD. filters is an optional JSON string to restrict the data (default
    '{}' = all data). Use this to see where a change is concentrated and which
    segments stayed flat (control groups)."""
    if dimension not in ("category", "region"):
        return "dimension must be 'category' or 'region'"
    d = subset(filters)
    lines = [f"[{before_start}..{before_end} vs {after_start}..{after_end}] filters={filters}"]
    for val, g in d.groupby(dimension):
        b = daily_avg(g, before_start, before_end)
        a = daily_avg(g, after_start, after_end)
        lines.append(f"{dimension}={val}: {b:.0f}/day -> {a:.0f}/day ({(a - b) / b * 100:+.1f}%)")
    return "\n".join(lines)


@tool
def compare_periods(filters: str, before_start: str, before_end: str,
                    after_start: str, after_end: str) -> str:
    """Compare average DAILY revenue for ONE specific slice between two periods,
    e.g. filters='{"category": "A", "region": "Giza"}' (use '{}' for all data).
    Dates are YYYY-MM-DD. Use for specific combinations; use breakdown_by to
    cover every category or region at once."""
    d = subset(filters)
    if d.empty:
        return "No rows match these filters."
    b, a = daily_avg(d, before_start, before_end), daily_avg(d, after_start, after_end)
    return (f"{filters}: {b:.0f}/day -> {a:.0f}/day ({(a - b) / b * 100:+.1f}%) "
            f"[{before_start}..{before_end} vs {after_start}..{after_end}]")


tools = [describe_data, find_changepoint, breakdown_by, compare_periods]


# ---------- 3. State, prompts and models ----------
class Verdict(BaseModel):
    verdict: Literal["supported", "contradicted", "inconclusive"]
    summary: str = Field(description="2-3 plain sentences a manager can read, no jargon.")
    evidence: list[str] = Field(description="At most 4 short items, each with one number from the tool results.")
    likely_cause: str = Field(description="Where the change is concentrated, only if the results show it; otherwise 'not determined'.")
    caveats: str = Field(description="One short sentence on limits of this analysis.")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    verdict: Optional[Verdict]
    warnings: Optional[list[str]]
    report_pdf: Optional[bytes]


SYSTEM = """You are a claim falsifier. You are given a business claim about causes. \
Do NOT try to confirm it; try hard to DISPROVE it using the tools.
1. Write down the predictions the claim makes (when the change should start, which \
groups should be affected, which should not).
2. Run these checks, requesting several tool calls in one turn when possible:
   a. describe_data, then find_changepoint on all data ('{}') to learn WHEN the change started.
   b. breakdown_by 'category' and breakdown_by 'region', using a "before" period that ends \
BEFORE the earliest changepoint (never overlapping the drop) and an "after" period that \
starts at the changepoint.
   c. At least one control check: a slice the claimed cause did not touch.
3. Never state a number that did not come from a tool result.
4. Use at most 8 tool calls. Then finish with a short summary of what the checks found.
Tool arguments: filters must be a JSON string like '{"category": "A"}'."""

VERDICT_SYSTEM = """You judge a business claim using ONLY the tool results you are given.
Rules: call it 'contradicted' if a check the claim depends on clearly fails (for example \
the change began before the claimed cause, or groups untouched by the cause changed just \
as much). Call it 'supported' only if it survived at least three different checks including \
a control group. Otherwise 'inconclusive'. Surviving checks does not prove causation.
Never say anything about a category or region that does not appear in the results. For \
likely_cause: look at each breakdown. If within one dimension some segments changed a lot \
(for example -25%) while others stayed near zero (for example -0.6%), name the segments \
that changed as where the change is concentrated, with their numbers. Write 'not determined' \
only if no breakdown shows such a split. Copy every number and date exactly as it appears \
in the results; never invent a date, period or figure. Be short and plain, written for a \
business manager."""

llm = ChatGroq(model=MODEL, temperature=0, max_retries=6)  # retries on rate limits
agent_llm = llm.bind_tools(tools)
judge_llm = llm.with_structured_output(Verdict)


# ---------- 4. Nodes and graph ----------
def agent(state: State):
    msgs = [{"role": "system", "content": SYSTEM}] + state["messages"]
    return {"messages": [agent_llm.invoke(msgs)]}


NUM = re.compile(r"\d+(?:\.\d+)?")


def ungrounded_numbers(v, results):
    """Numbers in the verdict text that never appear in the tool results."""
    text = " ".join(v.evidence + [v.summary, v.likely_cause])
    return sorted({n for n in NUM.findall(text) if len(n) >= 3 and n not in results})


def verdict_node(state: State):
    # The judge sees only the claim and the raw tool results (cheaper and more faithful).
    claim = state["messages"][0].content
    results = "\n".join(m.content for m in state["messages"] if m.type == "tool")
    prompt = f"{claim}\n\nTool results:\n{results}\n\nGive the structured verdict."

    def ask(extra=""):
        return judge_llm.invoke([
            {"role": "system", "content": VERDICT_SYSTEM},
            {"role": "user", "content": prompt + extra},
        ])

    v = ask()
    bad = ungrounded_numbers(v, results)
    if bad:  # grounding check: one retry if the verdict used numbers not in the results
        v = ask(f"\n\nYour last answer used numbers that are not in the tool results: {bad}. "
                "Redo it using only numbers copied from the results.")
        bad = ungrounded_numbers(v, results)
    return {"verdict": v, "warnings": bad}


def report_node(state: State):
    claim = state["messages"][0].content.replace("Claim to test: ", "", 1)
    steps = steps_from_messages(state["messages"])
    pdf = build_pdf(claim, state["verdict"], state.get("warnings") or [], steps, DF)
    return {"report_pdf": pdf}


graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("verdict", verdict_node)
graph.add_node("report", report_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "verdict"})
graph.add_edge("tools", "agent")
graph.add_edge("verdict", "report")
graph.add_edge("report", END)
app = graph.compile()


# ---------- 5. Run it with a clean, readable output ----------
def wrap(text, indent="  "):
    return textwrap.fill(text, width=78, initial_indent=indent, subsequent_indent=indent)


if __name__ == "__main__":
    claim = "Sales dropped in Q3 because we raised Category A prices on 2026-08-01."
    print(f"\nCLAIM: {claim}\n\nInvestigating...")

    verdict, warnings, report_pdf = None, None, None
    for step in app.stream(
        {"messages": [("user", f"Claim to test: {claim}")], "verdict": None, "warnings": None, "report_pdf": None},
        config={"recursion_limit": 40},
        stream_mode="updates",
    ):
        for node, update in step.items():
            if node == "agent":
                for call in update["messages"][-1].tool_calls:
                    a = call["args"]
                    detail = a.get("dimension") or a.get("filters", "")
                    print(f"\n  > {call['name']} {detail}")
            elif node == "tools":
                for m in update["messages"]:
                    for line in m.content.splitlines():
                        print(wrap(line, indent="    "))
            elif node == "verdict":
                verdict = update["verdict"]
                warnings = update.get("warnings")
            elif node == "report":
                report_pdf = update["report_pdf"]

    if verdict:
        print("\n" + "=" * 78)
        print(f"VERDICT: {verdict.verdict.upper()}")
        print("=" * 78)
        print("\nSUMMARY")
        print(wrap(verdict.summary))
        print("\nKEY EVIDENCE")
        for i, e in enumerate(verdict.evidence, 1):
            print(wrap(f"{i}. {e}"))
        print("\nLIKELY CAUSE")
        print(wrap(verdict.likely_cause))
        print("\nCAVEATS")
        print(wrap(verdict.caveats))
        print("\nGROUNDING CHECK")
        if warnings:
            print(wrap(f"WARNING: these numbers are not in the tool results: {warnings}"))
        else:
            print(wrap("OK: every number in the verdict appears in the tool results."))
        print()

    if report_pdf:
        out = Path(__file__).with_name("claim_report.pdf")
        out.write_bytes(report_pdf)
        print(f"PDF report saved to: {out}\n")
