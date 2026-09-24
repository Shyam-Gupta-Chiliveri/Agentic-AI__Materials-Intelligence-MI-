from agent.knowledge_graph import query_knowledge_graph, resolve_entity
from agent.orchestrator import _llm_tool_result, _strip_markup
from agent.tools import detect_conflict, run_tool


def test_llm_does_not_see_svg():
    result = query_knowledge_graph("Vickers hardness")
    slim = _llm_tool_result(result)
    assert "svg" not in slim
    assert "edges" in slim
    cleaned = _strip_markup("Intro <svg xmlns='x'><text>PMSM motor</text></svg> outro")
    assert "svg" not in cleaned.lower()
    assert "Intro" in cleaned and "outro" in cleaned


def test_resolve_hardness():
    assert "HV10" in resolve_entity("what is vickers hardness")


def test_graph_links_axle_to_motor():
    result = query_knowledge_graph("how does the axle carry the motor shaft")
    assert result["ok"]
    rels = {(e["from"], e["relation"], e["to"]) for e in result["edges"]}
    assert any(rel == "carries" or rel == "loads" for _, rel, _ in rels)


def test_mermaid_is_returned():
    result = query_knowledge_graph("Vickers hardness ISO 6507")
    assert "flowchart LR" in result["mermaid"] or "graph LR" in result["mermaid"]
    assert "HV10" in result["mermaid"]
    assert "<svg" in result["svg"]
    assert "Digital" in result["svg"] and "twin" in result["svg"]
    assert "HV10" in result["svg"] or "hardness" in result["svg"]


def test_mermaid_dark_uses_light_labels():
    from agent.knowledge_graph import edges_to_mermaid, edges_to_svg

    result = query_knowledge_graph("Vickers hardness ISO 6507")
    mermaid = edges_to_mermaid(result["edges"], dark=True)
    assert "primaryTextColor':'#f2f4f8'" in mermaid
    assert "color:#f2f4f8" in mermaid
    svg = edges_to_svg(result["edges"], dark=True)
    assert 'fill="#f2f4f8"' in svg


def test_workflow_nodes_present():
    result = query_knowledge_graph("procedure to make perfect ductile material heat treatment")
    nodes = {e["from"] for e in result["edges"]} | {e["to"] for e in result["edges"]}
    assert any(n in nodes for n in ("heat_treatment", "temper", "sectioning", "polishing", "tensile_test"))
    rels = {e["relation"] for e in result["edges"]}
    assert any(r in rels for r in ("followed_by", "first_step", "promotes", "reduces"))


def test_layout_fits_common_questions():
    from agent.knowledge_graph import _layout_nodes, query_knowledge_graph

    questions = [
        "Vickers hardness ISO 6507",
        "This hardened steel axle has Mean HV10 = 520. Is it a brittle fracture risk",
        "Use motor profile 29. Copper windings at 141 C, torque peaks at 261 Nm",
        "How do I prepare a metallographic sample",
        "SEM dimples and cleavages",
        "how does the axle carry the motor shaft",
    ]
    nw, nh = 158, 46
    for q in questions:
        result = query_knowledge_graph(q)
        nodes = []
        for e in result["edges"]:
            for n in (e["from"], e["to"]):
                if n not in nodes:
                    nodes.append(n)
        pos, width, height, _ = _layout_nodes(nodes, result["edges"], nw, nh)
        assert width <= 1400
        for n, (x, y) in pos.items():
            assert 8 <= x <= width - nw - 8, f"{q}: {n} x={x} width={width}"
            assert 40 <= y <= height - nh - 8, f"{q}: {n} y={y} height={height}"
        assert "viewBox" in result["svg"]


def test_skip_labels_stay_below_title():
    import re
    from agent.knowledge_graph import edges_to_svg, query_knowledge_graph

    result = query_knowledge_graph(
        "This hardened steel axle has Mean HV10 = 520. Is it a brittle fracture risk"
    )
    svg = edges_to_svg(result["edges"], dark=True)
    assert "Digital" in svg and "twin" in svg
    for m in re.finditer(r'translate\(([-\d.]+),([-\d.]+)\) rotate', svg):
        ly = float(m.group(2))
        assert ly >= 72, f"label y={ly} overlaps the heading"


def test_graph_iso_6507():
    result = query_knowledge_graph("ISO 6507 Vickers")
    assert any(e["to"] == "ISO_6507" or e["from"] == "ISO_6507" for e in result["edges"])


def test_run_tool_graph():
    out = run_tool("query_knowledge_graph", {"query": "SEM dimples"})
    assert out["ok"]
    assert out["edges"]


def test_run_tool_unknown():
    out = run_tool("not_a_tool", {})
    assert out["ok"] is False


def test_conflict_agree():
    axle = {"ok": True, "brittle_pct": 70.0}
    sem = {"ok": True, "brittle_pct": 63.0}
    result = detect_conflict(axle, sem)
    assert result is not None
    assert result["conflict"] is False


def test_conflict_disagree():
    axle = {"ok": True, "brittle_pct": 80.0}
    sem = {"ok": True, "brittle_pct": 40.0}
    result = detect_conflict(axle, sem)
    assert result is not None
    assert result["conflict"] is True
