"""PDF report and charts for the Claim Falsifier."""
import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

VERDICT_COLORS = {"supported": "#27ae60", "contradicted": "#c0392b", "inconclusive": "#e69f00"}
INK = "#1f2d3d"
GREY = "#7f8c8d"

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SEG = re.compile(r"(category|region)=(\w+): \d+/day -> \d+/day \(([+-]?\d+(?:\.\d+)?)%\)")

# Characters the built-in PDF fonts cannot draw are replaced with plain equivalents.
_REPL = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2192": "->",
    "\u00a0": " ", "\u202f": " ", "\u2248": "~", "\u2026": "...",
}


def clean(text):
    text = str(text)
    for old, new in _REPL.items():
        text = text.replace(old, new)
    return escape(text.encode("latin-1", "replace").decode("latin-1"))


# ---------- helpers that read the agent's trace ----------
def steps_from_messages(messages):
    """Pair every tool call with its result, in the order the agent ran them."""
    results = {m.tool_call_id: m.content for m in messages if getattr(m, "type", "") == "tool"}
    steps = []
    for m in messages:
        if getattr(m, "type", "") == "ai":
            for c in getattr(m, "tool_calls", None) or []:
                steps.append({"name": c["name"], "args": c["args"], "result": results.get(c["id"], "")})
    return steps


def parse_segments(results_text):
    """Read 'category=A: 2996/day -> 2476/day (-17.4%)' lines from unfiltered breakdowns."""
    segs, active = {}, False
    for line in results_text.splitlines():
        if line.startswith("["):
            active = line.rstrip().endswith(("filters={}", "filters="))
            continue
        m = _SEG.match(line.strip()) if active else None
        if m:
            segs[(m.group(1), m.group(2))] = float(m.group(3))
    return segs


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


# ---------- charts: each returns (png_bytes, width_in, height_in) ----------
def timeline_chart(df, claim, results_text):
    w, h = 7.5, 3.2
    daily = df.groupby("date").revenue.sum()
    smooth = daily.rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(smooth.index, smooth.values, color="#1f4e79", linewidth=1.8,
            label="Total daily revenue (7-day average)")

    cp = (re.search(r"^\{\}: biggest shift starts (\d{4}-\d{2}-\d{2})", results_text, re.M)
          or re.search(r"biggest shift starts (\d{4}-\d{2}-\d{2})", results_text))
    if cp:
        ax.axvline(pd.Timestamp(cp.group(1)), color="#c0392b", linestyle="--", linewidth=1.6,
                   label=f"Detected shift ({cp.group(1)})")
    claimed = _DATE.search(claim)
    if claimed:
        ax.axvline(pd.Timestamp(claimed.group(0)), color="#e69f00", linestyle=":", linewidth=2.2,
                   label=f"Claimed cause ({claimed.group(0)})")

    ax.set_title("When did revenue change?", loc="left", fontsize=11, color=INK)
    ax.set_ylabel("Revenue per day", fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _to_png(fig), w, h


def breakdown_chart(results_text):
    segs = parse_segments(results_text)
    if not segs:
        return None
    items = sorted(segs.items(), key=lambda kv: kv[1], reverse=True)  # most negative ends on top
    labels = [f"{dim.capitalize()}: {val}" for (dim, val), _ in items]
    values = [v for _, v in items]
    colors_ = ["#c0392b" if v <= -5 else "#27ae60" if v >= 5 else "#95a5a6" for v in values]

    w, h = 7.5, max(2.2, 0.38 * len(items) + 1.0)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.barh(labels, values, color=colors_)
    span = max(abs(min(values)), abs(max(values)), 5)
    ax.set_xlim(min(0, min(values)) - span * 0.25, max(0, max(values)) + span * 0.25)
    for i, v in enumerate(values):
        ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), i, f"{v:+.1f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_title("Where did it change? (after vs before)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Change in average daily revenue (%)", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _to_png(fig), w, h


# ---------- the PDF ----------
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(GREY))
    canvas.drawString(2 * cm, 1.1 * cm,
                      "Claim Falsifier - automatically generated. Surviving checks does not prove causation.")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(claim, verdict, warnings, steps, df):
    results_text = "\n".join(s["result"] for s in steps)
    base = getSampleStyleSheet()
    st = {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
                                leading=26, alignment=0, textColor=colors.HexColor(INK), spaceAfter=2),
        "sub": ParagraphStyle("sub", parent=base["Normal"], fontSize=9,
                              textColor=colors.HexColor(GREY), spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                             textColor=colors.HexColor(INK), spaceBefore=12, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=10.5, leading=15),
        "item": ParagraphStyle("item", parent=base["BodyText"], fontSize=10.5, leading=15,
                               leftIndent=14, firstLineIndent=-14, spaceAfter=2),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=8.5, leading=11,
                                textColor=colors.HexColor("#555555")),
        "claim": ParagraphStyle("claim", parent=base["BodyText"], fontName="Helvetica-Oblique",
                                fontSize=11.5, leading=16),
        "badge": ParagraphStyle("badge", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=13,
                                textColor=colors.white, alignment=1),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.5, leading=11),
    }
    full_w = 17 * cm
    story = [
        Paragraph("Claim Falsifier Report", st["title"]),
        Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}", st["sub"]),
    ]

    claim_box = Table([[Paragraph("CLAIM TESTED", st["small"])], [Paragraph(clean(claim), st["claim"])]],
                      colWidths=[full_w])
    claim_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [claim_box, Spacer(1, 10)]

    badge = Table([[Paragraph(verdict.verdict.upper(), st["badge"])]], colWidths=[6 * cm], hAlign="LEFT")
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(VERDICT_COLORS[verdict.verdict])),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [badge,
              Paragraph("Summary", st["h2"]), Paragraph(clean(verdict.summary), st["body"]),
              Paragraph("Key evidence", st["h2"])]
    for i, e in enumerate(verdict.evidence, 1):
        story.append(Paragraph(f"{i}.&nbsp;&nbsp;{clean(e)}", st["item"]))
    story += [Paragraph("Likely cause", st["h2"]), Paragraph(clean(verdict.likely_cause), st["body"])]

    charts = [c for c in (timeline_chart(df, claim, results_text), breakdown_chart(results_text)) if c]
    if charts:
        story.append(Paragraph("What the data shows", st["h2"]))
        for png, w, h in charts:
            story += [Image(io.BytesIO(png), width=full_w, height=full_w * h / w), Spacer(1, 6)]

    story += [Paragraph("Caveats", st["h2"]), Paragraph(clean(verdict.caveats), st["body"]),
              Paragraph("Methodology", st["h2"]),
              Paragraph("Every number in this report comes from the pandas checks below. The language model "
                        "chose which checks to run and wrote the explanation; it did not calculate anything.",
                        st["body"]), Spacer(1, 6)]

    rows = [[Paragraph("<b>#</b>", st["cell"]), Paragraph("<b>Check</b>", st["cell"]),
             Paragraph("<b>Result</b>", st["cell"])]]
    for i, s in enumerate(steps, 1):
        args = ", ".join(f"{k}={v}" for k, v in s["args"].items())
        check = f"<b>{clean(s['name'])}</b>" + (
            f"<br/><font color='{GREY}'>{clean(args)}</font>" if args else "")
        result = "<br/>".join(clean(line) for line in s["result"].splitlines())
        rows.append([Paragraph(str(i), st["cell"]), Paragraph(check, st["cell"]),
                     Paragraph(result, st["cell"])])
    table = Table(rows, colWidths=[0.9 * cm, 5.1 * cm, 11 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaeef2")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd6dd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)

    if warnings:
        grounding = (f"<b>Grounding check:</b> WARNING - these numbers in the verdict were not found in "
                     f"the tool results: {clean(', '.join(map(str, warnings)))}")
    else:
        grounding = "<b>Grounding check:</b> passed. Every number in the verdict appears in the tool results."
    story += [Spacer(1, 10), Paragraph(grounding, st["small"])]

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="Claim Falsifier Report").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
