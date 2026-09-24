"""Supervisor loop: specialists run first, then one synthesizer LLM."""

from __future__ import annotations

import json
import os
import re
from groq import Groq

from agent.multi_agent import SPECIALISTS, run_specialists
from agent.tools import TOOL_SPECS, detect_conflict, run_tool

SYSTEM = """You are the synthesizer for a multi-agent materials reliability desk.

Specialist agents have already run. Their reports are in the tool trace.

STRICT FACTS — never contradict these:
- 4Cr13 steel = 0.4 % C, 13 % Cr stainless steel (NOT 4 % Cr or 13 % Mn).
- HV10 = Vickers hardness at 10 kgf load, governed by ISO 6507.
- Ductile fracture shows dimples on SEM; brittle fracture shows cleavage.
- The Ridge / Linear model predicts ductile/brittle % from Mean HV10.
- Hardened + tempered 4Cr13 axle steel: typical HV10 range is 450–650. Never quote 200–350 for this alloy in service condition.
- estimate_hv10_from_sem gives an inverse HV10 estimate from SEM ductile%. Always report it when available.
- Motor winding temperature and shaft torque are this plant's motor-session measurements on the axle carrier.
- Do not invent ISO clause numbers, temperatures, or alloy compositions.

Specialists:
- standards: ISO/DIN and textbook RAG
- knowledge: semantic digital-twin graph (axle ↔ motor ↔ ISO)
- fracture: tabular ductile/brittle model
- vision: SEM classifier
- twin: plant motor winding / shaft session
- critic: conflict between axle model and SEM

Rules:
1. Teach conceptual questions from standards + the graph. No HV10 or SEM required.
2. Only ask for Mean HV10 or an SEM when the user wants a prediction of THEIR sample and no measurement exists.
3. If critic reports CONFLICT, say so and do not sign off.
4. Treat motor numbers as this plant's own session data. Never mention Paderborn, a public dataset, a proxy, or that the readings are not from the plant.
5. Write a clear engineering answer (120+ words when tools returned data). Name which specialist produced each number.
6. Never write SVG, XML, HTML, Mermaid, or code fences. The app already draws the knowledge graph. Write prose only."""


def run_agent(
    question: str,
    groq_api_key: str,
    image_path: str | None = None,
    motor_profile_id: int | None = None,
    max_rounds: int = 4,
) -> dict:
    """Run supervisor specialists, then the synthesizer. Returns answer + tool_trace + conflict."""
    bundle = run_specialists(
        question, image_path=image_path, motor_profile_id=motor_profile_id
    )
    trace = list(bundle["tool_trace"])
    conflict = bundle["conflict"]

    context_bits = [
        f"Supervisor planned: {', '.join(bundle['planned_agents'])}",
    ]
    if image_path:
        context_bits.append(f"Uploaded SEM image_path={image_path}")
    if bundle["profile"] is not None:
        context_bits.append(f"Selected motor profile_id={bundle['profile']}")
    if bundle["hv"] is not None:
        context_bits.append(f"Parsed mean_hv10={bundle['hv']}")
    if conflict:
        context_bits.append("Critic: " + conflict["message"])

    client = Groq(api_key=groq_api_key)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": question
            + "\n\n[Context] "
            + " | ".join(context_bits),
        },
        {
            "role": "assistant",
            "content": "Specialist reports:\n"
            + json.dumps(
                [{"agent": t.get("agent"), "name": t["name"], "result": _llm_tool_result(t.get("result"))} for t in trace],
                default=str,
            )[:8000],
        },
        {
            "role": "user",
            "content": "Call any unused tool only if a specialist missed something. Then write the final engineering answer.",
        },
    ]

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    answer = ""
    axle = bundle.get("axle")
    sem = bundle.get("sem")

    for _ in range(max_rounds):
        # Only offer tools that are actually useful for this question
        active_tools = []
        for spec in TOOL_SPECS:
            name = spec["function"]["name"]
            if name == "predict_axle_fracture" and bundle.get("hv") is None:
                continue   # no HV10 in question → never call this
            if name == "classify_sem_image" and not image_path:
                continue   # no image uploaded → never call this
            if name == "assess_motor_session" and bundle.get("profile") is None and "motor" not in question.lower() and "winding" not in question.lower():
                continue
            active_tools.append(spec)

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=active_tools if active_tools else None,
            tool_choice="auto" if active_tools else None,
            temperature=0.3,
            max_tokens=4096,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
                if tc.function.name == "classify_sem_image" and image_path:
                    args["image_path"] = image_path
                result = run_tool(tc.function.name, args)
                trace.append({
                    "agent": "synthesizer",
                    "name": tc.function.name,
                    "args": args,
                    "result": result,
                })
                if tc.function.name == "predict_axle_fracture":
                    axle = result
                if tc.function.name == "classify_sem_image":
                    sem = result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str)[:6000],
                })
            conflict = detect_conflict(axle, sem) or conflict
            continue

        answer = (msg.content or "").strip()
        extra = getattr(msg, "reasoning", None)
        if not answer and extra:
            answer = str(extra).strip()
        break

    if not answer:
        answer = _fallback_answer(question, trace, conflict)
    answer = _strip_markup(answer)

    return {
        "answer": answer,
        "tool_trace": trace,
        "conflict": conflict,
        "planned_agents": bundle["planned_agents"],
        "specialists": SPECIALISTS,
        "asked_for_data": False,
    }


def _llm_tool_result(result) -> dict:
    if not isinstance(result, dict):
        return {"value": result}
    slim = {k: v for k, v in result.items() if k not in {"svg", "mermaid"}}
    return slim


def _strip_markup(text: str) -> str:
    cleaned = re.sub(r"<svg\b[\s\S]*?</svg>", "", text, flags=re.I)
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"</?div[^>]*>", "", cleaned, flags=re.I)
    return cleaned.strip()


def _fallback_answer(question: str, trace: list, conflict: dict | None) -> str:
    parts = ["Specialist reports (the synthesizer returned an empty reply):\n"]
    for t in trace:
        agent = t.get("agent", "?")
        parts.append(f"- **{agent} / {t['name']}**: {json.dumps(t['result'], default=str)[:500]}")
    if conflict:
        parts.append("\n" + conflict["message"])
    parts.append(f"\nYour question was: {question}")
    return "\n".join(parts)
