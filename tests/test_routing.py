from agent.routing import extract_hv, extract_profile, is_conceptual, route_specialists, wants_prediction


def test_extract_hv_equals():
    assert extract_hv("This axle has HV10 = 520. Is it brittle?") == 520.0


def test_extract_hv_suffix():
    assert extract_hv("measured 480 HV10 after tempering") == 480.0


def test_extract_hv_absent():
    assert extract_hv("What is hardness in materials science?") is None


def test_extract_profile():
    assert extract_profile("use motor profile 29") == 29


def test_conceptual_hardness_question():
    q = "How does hardness work in materials science and what is it?"
    assert is_conceptual(q)
    assert not wants_prediction(q)


def test_prediction_question():
    q = "Is this axle a brittle fracture risk?"
    assert wants_prediction(q)


def test_supervisor_routes_standards_and_graph():
    agents = route_specialists("What is Vickers hardness?")
    assert "standards" in agents
    assert "knowledge" in agents
    assert "vision" not in agents


def test_supervisor_routes_fracture_when_hv_present():
    agents = route_specialists("HV10 = 520 — brittle?")
    assert "fracture" in agents
    assert "critic" in agents


def test_supervisor_routes_vision_and_twin():
    agents = route_specialists(
        "Classify this surface and check the hot motor",
        image_path="/tmp/sem.tif",
        motor_profile_id=29,
    )
    assert "vision" in agents
    assert "twin" in agents
