import os

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Claim Falsifier", page_icon="🔎", layout="centered")

# API key: Streamlit secrets when deployed, .env file when running locally.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to a .env file (local) or to the app's Secrets (deployed).")
    st.stop()

import falsifier  # imported after the key is set
from report import breakdown_chart, steps_from_messages, timeline_chart

EXAMPLES = [
    "Sales dropped in Q3 because we raised Category A prices on 2026-08-01.",
    "Category A revenue fell in Q3 because of problems specific to Category A.",
    "Giza is the region driving the Q3 decline.",
    "The Q3 decline is concentrated in Cairo and Alexandria, not spread evenly across regions.",
]

st.title("Claim Falsifier")
st.write(
    "Type a business claim about the sales data. The agent tries to **disprove** it with pandas checks "
    "and returns a verdict (supported, contradicted or inconclusive) backed by numbers that come from "
    "the data, not from the language model."
)

with st.expander("About the demo dataset"):
    st.caption(
        "Synthetic daily revenue for 3 categories (A, B, C) and 3 regions (Alexandria, Cairo, Giza), "
        "January to September 2026. Claims must be about this data."
    )
    st.dataframe(falsifier.DF.sample(8, random_state=1).sort_values("date"), hide_index=True)

example = st.selectbox("Try an example claim", ["(write my own)"] + EXAMPLES, index=1)
claim = st.text_area(
    "Claim to test",
    value="" if example == "(write my own)" else example,
    placeholder="e.g. Sales dropped in Q3 because of ...",
    height=90,
).strip()

if st.button("Test claim", type="primary", disabled=not claim):
    st.session_state.pop("result", None)
    msgs, verdict, warnings, pdf = [], None, None, None
    try:
        with st.status("Investigating...", expanded=True) as status:
            for step in falsifier.app.stream(
                {"messages": [("user", f"Claim to test: {claim}")], "verdict": None,
                 "warnings": None, "report_pdf": None},
                config={"recursion_limit": 40},
                stream_mode="updates",
            ):
                for node, update in step.items():
                    if node == "agent":
                        m = update["messages"][-1]
                        msgs.append(m)
                        for c in m.tool_calls:
                            a = c["args"]
                            st.write(f"Running **{c['name']}** {a.get('dimension') or a.get('filters', '')}")
                    elif node == "tools":
                        for m in update["messages"]:
                            msgs.append(m)
                            st.code(m.content, language=None)
                    elif node == "verdict":
                        verdict, warnings = update["verdict"], update.get("warnings")
                        st.write("Writing the verdict...")
                    elif node == "report":
                        pdf = update["report_pdf"]
            status.update(label="Investigation complete", state="complete", expanded=False)
    except Exception as e:
        if "RateLimit" in type(e).__name__:
            st.error("The free Groq tier is rate-limited right now. Wait a minute and try again.")
        else:
            st.error(f"Something went wrong ({type(e).__name__}). Please try again.")
        with st.expander("Technical details"):
            st.code(str(e))
    else:
        if verdict:
            steps = steps_from_messages(msgs)
            results_text = "\n".join(s["result"] for s in steps)
            st.session_state["result"] = {
                "claim": claim, "verdict": verdict, "warnings": warnings, "pdf": pdf, "steps": steps,
                "timeline": timeline_chart(falsifier.DF, claim, results_text)[0],
                "breakdown": (breakdown_chart(results_text) or (None,))[0],
            }

res = st.session_state.get("result")
if res:
    v = res["verdict"]
    banner = {"supported": st.success, "contradicted": st.error, "inconclusive": st.warning}[v.verdict]
    banner(f"Verdict: {v.verdict.upper()}")

    st.subheader("Summary")
    st.write(v.summary)
    st.subheader("Key evidence")
    for i, e in enumerate(v.evidence, 1):
        st.write(f"{i}. {e}")
    st.subheader("Likely cause")
    st.write(v.likely_cause)

    st.subheader("What the data shows")
    st.image(res["timeline"])
    if res["breakdown"]:
        st.image(res["breakdown"])

    st.subheader("Caveats")
    st.write(v.caveats)
    if res["warnings"]:
        st.warning(f"Grounding check: these numbers were not found in the tool results: {res['warnings']}")
    else:
        st.caption("Grounding check passed: every number in the verdict appears in the tool results.")

    with st.expander("The checks the agent ran"):
        for s in res["steps"]:
            st.markdown(f"**{s['name']}** `{s['args']}`")
            st.code(s["result"], language=None)

    if res["pdf"]:
        st.download_button("Download PDF report", data=res["pdf"], file_name="claim_report.pdf",
                           mime="application/pdf")
