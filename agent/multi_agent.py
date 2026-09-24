"""Supervisor + specialist agents. Each specialist owns a slice of the twin."""

from __future__ import annotations

from agent.routing import extract_hv, extract_profile, route_specialists
from agent.tools import detect_conflict, run_tool

SPECIALISTS = {
    "standards": "ISO / textbook RAG",
    "knowledge": "Semantic knowledge graph",
    "fracture": "Axle ductile/brittle model",
    "vision": "SEM fracture classifier",
    "twin": "Motor winding / shaft digital twin",
    "critic": "Conflict check across models",
}


def _step(agent: str, name: str, args: dict, result: dict) -> dict:
    return {"agent": agent, "name": name, "args": args, "result": result}


def run_specialists(
    question: str,
    *,
    image_path: str | None = None,
    motor_profile_id: int | None = None,
) -> dict:
    """Run the supervisor routing + specialist tools (no LLM)."""
    hv = extract_hv(question)
    profile = motor_profile_id if motor_profile_id is not None else extract_profile(question)
    planned = route_specialists(
        question, image_path=image_path, motor_profile_id=profile
    )
    trace: list[dict] = []
    axle = None
    sem = None
    motor = None
    conflict = None

    if "standards" in planned:
        args = {"query": question, "top_k": 4}
        if hv is not None:
            args["query"] = f"{question} Vickers hardness HV10 ISO 6507"
        result = run_tool("search_iso_standards", args)
        trace.append(_step("standards", "search_iso_standards", args, result))

    if "knowledge" in planned:
        args = {"query": question, "max_edges": 12}
        result = run_tool("query_knowledge_graph", args)
        trace.append(_step("knowledge", "query_knowledge_graph", args, result))

    if "fracture" in planned and hv is not None:
        args = {"mean_hv10": hv}
        result = run_tool("predict_axle_fracture", args)
        axle = result
        trace.append(_step("fracture", "predict_axle_fracture", args, result))

    if "vision" in planned and image_path:
        args = {"image_path": image_path}
        result = run_tool("classify_sem_image", args)
        sem = result
        trace.append(_step("vision", "classify_sem_image", args, result))

        # If no HV10 provided, inverse-predict it from the SEM ductile%
        if hv is None and result.get("ok") and result.get("ductile_pct") is not None:
            inv_args = {"ductile_pct": result["ductile_pct"]}
            inv_result = run_tool("estimate_hv10_from_sem", inv_args)
            trace.append(_step("vision", "estimate_hv10_from_sem", inv_args, inv_result))

    if "twin" in planned:
        args = {} if profile is None else {"profile_id": int(profile)}
        result = run_tool("assess_motor_session", args)
        motor = result
        trace.append(_step("twin", "assess_motor_session", args, result))

    if "critic" in planned:
        conflict = detect_conflict(axle, sem)
        if conflict:
            trace.append(_step("critic", "detect_conflict", {}, conflict))

    return {
        "planned_agents": planned,
        "tool_trace": trace,
        "conflict": conflict,
        "hv": hv,
        "profile": profile,
        "axle": axle,
        "sem": sem,
        "motor": motor,
    }
