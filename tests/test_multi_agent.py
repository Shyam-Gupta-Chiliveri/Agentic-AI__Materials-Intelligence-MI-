from agent.multi_agent import run_specialists


def test_specialists_always_search_graph():
    bundle = run_specialists("What is hardness in materials science?")
    names = [t["name"] for t in bundle["tool_trace"]]
    assert "query_knowledge_graph" in names
    assert "standards" in bundle["planned_agents"]


def test_specialists_skip_fracture_without_hv():
    bundle = run_specialists("Explain sample preparation for SEM")
    names = [t["name"] for t in bundle["tool_trace"]]
    assert "predict_axle_fracture" not in names
