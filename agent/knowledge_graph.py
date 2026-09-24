"""Semantic knowledge graph for the axle–motor–standards digital twin."""

from __future__ import annotations

from collections import defaultdict

# Directed labeled edges: (src, relation, dst, note)
TRIPLES: list[tuple[str, str, str, str]] = [
    ("4Cr13_axle", "made_of", "4Cr13_steel", "Hardened stainless axle steel (≈0.4% C, 13% Cr)."),
    ("4Cr13_axle", "measured_by", "HV10", "Vickers hardness, ISO 6507, 10 kgf load."),
    ("4Cr13_axle", "carries", "PMSM_motor", "Axle is the mechanical carrier for the electric drive."),
    ("4Cr13_axle", "fails_as", "ductile_fracture", "Dimples on SEM when the matrix tears slowly."),
    ("4Cr13_axle", "fails_as", "brittle_fracture", "Cleavages on SEM when hardness is high / toughness low."),
    ("HV10", "governed_by", "ISO_6507", "Vickers hardness test method."),
    ("HV10", "inversely_related_to", "ductile_fracture", "Higher HV10 usually lowers ductile % on this axle set."),
    ("HV10", "positively_related_to", "brittle_fracture", "Ridge model: high HV10 → higher brittle %."),
    ("grain_size", "governed_by", "ISO_643", "Austenitic grain size rating."),
    ("SEM_image", "shows", "dimples", "Equiaxed microvoids = ductile rupture."),
    ("SEM_image", "shows", "cleavages", "Faceted transgranular planes = brittle rupture."),
    ("dimples", "indicates", "ductile_fracture", "U-Net / classifier maps dimple area to ductile %."),
    ("cleavages", "indicates", "brittle_fracture", "Cleavage area maps to brittle %."),
    ("PMSM_motor", "has_part", "copper_windings", "Stator copper; winding temperature from the plant motor session."),
    ("PMSM_motor", "has_part", "motor_shaft", "Shaft torque is the mechanical load into the axle."),
    ("copper_windings", "risk", "thermal_ageing", "Insulation life drops as winding °C rises (especially >120 °C)."),
    ("motor_shaft", "loads", "4Cr13_axle", "Torque / bending on the shaft is seen by the hardened axle."),
    ("PMSM_session", "instance_of", "PMSM_motor", "One profile_id is one plant motor operating session."),
    ("ISO_6507", "type", "ISO_standard", "Hardness testing."),
    ("ISO_643", "type", "ISO_standard", "Metallographic grain size."),
    ("ISO_6892", "type", "ISO_standard", "Tensile testing of metallic materials."),
    ("sample_prep", "used_for", "SEM_image", "Section, mount, grind, polish, etch before SEM."),
    ("sample_prep", "described_in", "metallography_textbooks", "Lab procedure from the local PDF corpus."),
    # ── Sample-prep workflow ──────────────────────────────────────────────────
    ("sample_prep", "first_step", "sectioning", "Cut a representative block from the axle."),
    ("sectioning", "followed_by", "mounting", "Embed in low-viscosity epoxy to support the sample."),
    ("mounting", "followed_by", "grinding", "Progressively finer SiC grit: 240 → 400 → 800 → 1200."),
    ("grinding", "followed_by", "polishing", "Diamond paste 3 µm → 0.25 µm for mirror finish."),
    ("polishing", "followed_by", "etching", "2 % nital reagent reveals grain boundaries."),
    ("etching", "reveals", "grain_size", "Austenitic grain size rated per ISO 643."),
    ("etching", "prepares_for", "SEM_image", "Etched surface is ready for SEM examination."),
    # ── Heat-treatment workflow ───────────────────────────────────────────────
    ("4Cr13_axle", "requires", "heat_treatment", "Heat treatment controls hardness and ductile behaviour."),
    ("heat_treatment", "first_step", "solution_anneal", "~950 °C / 60 min dissolves carbides."),
    ("solution_anneal", "followed_by", "quench", "Rapid water or oil quench forms martensite."),
    ("quench", "followed_by", "temper", "~350–500 °C / 60 min relieves stress, raises toughness."),
    ("temper", "reduces", "HV10", "Tempering lowers hardness towards the ductile range."),
    ("temper", "promotes", "ductile_fracture", "Tempered martensite tears with dimples, not cleavage."),
    # ── Verification ─────────────────────────────────────────────────────────
    ("tensile_test", "governed_by", "ISO_6892", "Tensile testing per ISO 6892-1, dog-bone specimen."),
    ("tensile_test", "verifies", "ductile_fracture", "Elongation ≥ 10 % confirms ductile behaviour."),
    ("HV10", "guides", "heat_treatment", "Target HV10 determines tempering temperature selection."),
]

ALIASES = {
    "hardness": "HV10",
    "vickers": "HV10",
    "hv": "HV10",
    "hv10": "HV10",
    "axle": "4Cr13_axle",
    "steel": "4Cr13_steel",
    "4cr13": "4Cr13_steel",
    "motor": "PMSM_motor",
    "pmsm": "PMSM_motor",
    "winding": "copper_windings",
    "windings": "copper_windings",
    "copper": "copper_windings",
    "shaft": "motor_shaft",
    "torque": "motor_shaft",
    "sem": "SEM_image",
    "dimple": "dimples",
    "cleavage": "cleavages",
    "brittle": "brittle_fracture",
    "ductile": "ductile_fracture",
    "iso 6507": "ISO_6507",
    "iso6507": "ISO_6507",
    "iso 643": "ISO_643",
    "grain": "grain_size",
    "sample prep": "sample_prep",
    "metallography": "sample_prep",
    "session": "PMSM_session",
    "metallographic": "sample_prep",
    "prepare": "sample_prep",
    "sample": "sample_prep",
    # procedure / workflow aliases
    "procedure": "sample_prep",
    "workflow": "heat_treatment",
    "heat treatment": "heat_treatment",
    "heat": "heat_treatment",
    "anneal": "solution_anneal",
    "solution": "solution_anneal",
    "quench": "quench",
    "temper": "temper",
    "tempering": "temper",
    "grind": "grinding",
    "polish": "polishing",
    "etch": "etching",
    "section": "sectioning",
    "mount": "mounting",
    "tensile": "tensile_test",
    "elongation": "tensile_test",
    "ductile material": "ductile_fracture",
    "perfect ductile": "ductile_fracture",
}


def _index():
    out = defaultdict(list)
    nodes = set()
    for src, rel, dst, note in TRIPLES:
        nodes.add(src)
        nodes.add(dst)
        out[src.lower()].append((src, rel, dst, note))
        out[dst.lower()].append((src, rel, dst, note))
        out[rel.lower()].append((src, rel, dst, note))
    return out, nodes


_INDEX, _NODES = _index()


def resolve_entity(query: str) -> list[str]:
    q = query.lower()
    hits = []
    for alias, node in ALIASES.items():
        if alias in q and node not in hits:
            hits.append(node)
    for node in _NODES:
        token = node.replace("_", " ").lower()
        if token in q and node not in hits:
            hits.append(node)
    return hits


def query_knowledge_graph(query: str, max_edges: int = 10) -> dict:
    """Return neighborhood triples for entities mentioned in the query."""
    entities = resolve_entity(query)
    if not entities:
        entities = ["4Cr13_axle"]

    # Auto-expand: if procedure/ductile keywords present, include workflow hubs
    q_low = query.lower()
    if any(kw in q_low for kw in ("procedure", "ductile material", "heat treatment", "process", "make", "produce")):
        for hub in ("heat_treatment", "sample_prep"):
            if hub not in entities:
                entities.append(hub)
    if any(kw in q_low for kw in ("sample prep", "metallograph", "grind", "polish", "etch")):
        if "sample_prep" not in entities:
            entities.append("sample_prep")

    mentioned = set(entities)
    workflow  = {"followed_by", "first_step"}
    optional  = {"carries", "has_part", "instance_of", "loads"}
    vision    = {"shows", "indicates"}
    sem_asked = bool(mentioned & {"SEM_image", "dimples", "cleavages", "sample_prep"})
    primary, wf_edges, secondary, extra = [], [], [], []
    seen: set[tuple[str, str, str]] = set()
    for ent in entities:
        for src, rel, dst, note in _INDEX.get(ent.lower(), []):
            key = (src, rel, dst)
            if key in seen:
                continue
            seen.add(key)
            edge = {"from": src, "relation": rel, "to": dst, "note": note}
            if src in mentioned and dst in mentioned:
                primary.append(edge)
            elif rel in workflow:
                wf_edges.append(edge)
            elif rel in optional or (rel in vision and not sem_asked):
                extra.append(edge)
            else:
                secondary.append(edge)
    edges = (primary + wf_edges + secondary + extra)[:max_edges]

    mermaid = edges_to_mermaid(edges)
    svg = edges_to_svg(edges)
    return {
        "ok": True,
        "query": query,
        "entities": entities,
        "edges": edges,
        "mermaid": mermaid,
        "svg": svg,
        "n_nodes": len(_NODES),
        "n_triples": len(TRIPLES),
        "note": "Local semantic digital-twin graph (axle ↔ motor ↔ ISO). Not a live plant historian.",
    }


_NODE_LABELS = {
    "4Cr13_axle": "4Cr13 axle",
    "4Cr13_steel": "4Cr13 steel",
    "HV10": "HV10 hardness",
    "ISO_6507": "ISO 6507",
    "ISO_643": "ISO 643",
    "ISO_6892": "ISO 6892",
    "ISO_standard": "ISO standard",
    "ductile_fracture": "Ductile fracture",
    "brittle_fracture": "Brittle fracture",
    "SEM_image": "SEM image",
    "PMSM_motor": "PMSM motor",
    "PMSM_session": "Motor session",
    "copper_windings": "Copper windings",
    "motor_shaft": "Motor shaft",
    "grain_size": "Grain size",
    "sample_prep": "Sample prep",
    "metallography_textbooks": "Textbooks",
    "thermal_ageing": "Thermal ageing",
    # workflow nodes
    "sectioning": "Sectioning",
    "mounting": "Mounting",
    "grinding": "Grinding",
    "polishing": "Polishing",
    "etching": "Etching",
    "heat_treatment": "Heat treatment",
    "solution_anneal": "Solution anneal",
    "quench": "Quench",
    "temper": "Temper",
    "tensile_test": "Tensile test",
}

# Three readable columns: source → system → outcome. Always fits the chat width.
_LAYER = {
    "4Cr13_steel": 0, "4Cr13_axle": 0, "sample_prep": 0, "PMSM_session": 0,
    "heat_treatment": 0, "sectioning": 0, "mounting": 0,
    "HV10": 1, "SEM_image": 1, "grain_size": 1, "PMSM_motor": 1,
    "copper_windings": 1, "motor_shaft": 1, "dimples": 1, "cleavages": 1,
    "grinding": 1, "polishing": 1, "etching": 1,
    "solution_anneal": 1, "quench": 1, "temper": 1,
    "ISO_6507": 2, "ISO_643": 2, "ISO_6892": 2, "ISO_standard": 2,
    "metallography_textbooks": 2, "ductile_fracture": 2,
    "brittle_fracture": 2, "thermal_ageing": 2, "tensile_test": 2,
}

_NODE_STYLE = {
    "4Cr13_axle": ("steel", "#fff4ef", "#d97757"),
    "4Cr13_steel": ("steel", "#fff4ef", "#d97757"),
    "HV10": ("measure", "#fff8e8", "#d4a017"),
    "ISO_6507": ("iso", "#eef3ff", "#4c6ad9"),
    "ISO_643": ("iso", "#eef3ff", "#4c6ad9"),
    "ISO_6892": ("iso", "#eef3ff", "#4c6ad9"),
    "ISO_standard": ("iso", "#eef3ff", "#4c6ad9"),
    "ductile_fracture": ("ok", "#eefaf3", "#2f9e63"),
    "dimples": ("ok", "#eefaf3", "#2f9e63"),
    "brittle_fracture": ("risk", "#fff1f0", "#c45c3a"),
    "cleavages": ("risk", "#fff1f0", "#c45c3a"),
    "PMSM_motor": ("motor", "#f4f0ff", "#6d5bd0"),
    "PMSM_session": ("motor", "#f4f0ff", "#6d5bd0"),
    "copper_windings": ("motor", "#f4f0ff", "#6d5bd0"),
    "motor_shaft": ("motor", "#f4f0ff", "#6d5bd0"),
    "SEM_image": ("sem", "#eaf7f8", "#2a9d8f"),
    "sample_prep": ("sem", "#eaf7f8", "#2a9d8f"),
    "grain_size": ("measure", "#fff8e8", "#d4a017"),
    "thermal_ageing": ("risk", "#fff1f0", "#c45c3a"),
    "metallography_textbooks": ("iso", "#eef3ff", "#4c6ad9"),
    # workflow / process nodes — warm teal-green
    "sectioning":     ("process", "#edf7ef", "#3a9e5f"),
    "mounting":       ("process", "#edf7ef", "#3a9e5f"),
    "grinding":       ("process", "#edf7ef", "#3a9e5f"),
    "polishing":      ("process", "#edf7ef", "#3a9e5f"),
    "etching":        ("process", "#edf7ef", "#3a9e5f"),
    "heat_treatment": ("process", "#edf7ef", "#3a9e5f"),
    "solution_anneal":("process", "#edf7ef", "#3a9e5f"),
    "quench":         ("process", "#edf7ef", "#3a9e5f"),
    "temper":         ("process", "#edf7ef", "#3a9e5f"),
    "tensile_test":   ("process", "#edf7ef", "#3a9e5f"),
}

_DARK_STYLE = {
    "steel": ("#3a241c", "#e8a07c"),
    "measure": ("#3a3018", "#e0c36a"),
    "iso": ("#1e2a4a", "#8aa4ff"),
    "ok": ("#163026", "#5dcaa0"),
    "risk": ("#3a1d1a", "#e08b78"),
    "motor": ("#2a2248", "#b5a6ff"),
    "sem": ("#163033", "#6ec8bf"),
    "process": ("#163324", "#5bbf7a"),
}


def _label(node: str) -> str:
    return _NODE_LABELS.get(node, node.replace("_", " "))


_SVG_FONT = "Arial,Helvetica,sans-serif"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_words(text: str) -> str:
    """Emit tspans so the space between words cannot collapse."""
    words = [w for w in text.split(" ") if w]
    if not words:
        return ""
    parts = [f"<tspan>{_esc(words[0])}</tspan>"]
    for word in words[1:]:
        parts.append(f'<tspan dx="6">{_esc(word)}</tspan>')
    return "".join(parts)


def _node_colors(node: str, dark: bool) -> tuple[str, str, str]:
    kind, fill, stroke = _NODE_STYLE.get(node, ("iso", "#eef3ff", "#4c6ad9"))
    if dark:
        fill, stroke = _DARK_STYLE.get(kind, (fill, stroke))
    return kind, fill, stroke


def _layout_nodes(
    nodes: list[str], edges: list[dict], nw: float, nh: float
) -> tuple[dict[str, tuple[float, float]], float, float, dict[str, int]]:
    """Left-to-right layered layout that always stays inside the canvas."""
    buckets: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for n in nodes:
        buckets[_LAYER.get(n, 1)].append(n)
    used = [i for i in (0, 1, 2) if buckets[i]]

    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        adj[e["from"]].append(e["to"])
        adj[e["to"]].append(e["from"])
    rank = {n: i for i, n in enumerate(nodes)}

    for _ in range(4):
        for layer in used:
            def _key(n: str, layer: int = layer) -> tuple[float, int]:
                neigh = [rank[m] for m in adj[n] if m in rank]
                bary = sum(neigh) / len(neigh) if neigh else 0.0
                return (bary, rank[n])

            buckets[layer].sort(key=_key)
            for i, n in enumerate(buckets[layer]):
                rank[n] = layer * 20 + i

    pad_x, pad_top, pad_bot = 44, 118, 68
    gap_x, gap_y = 420, 108
    col_x = {layer: pad_x + i * gap_x for i, layer in enumerate(used)}
    max_rows = max(len(buckets[i]) for i in used) or 1
    layer_of = {}
    pos: dict[str, tuple[float, float]] = {}
    for layer in used:
        group = buckets[layer]
        extra = ((max_rows - len(group)) * gap_y) / 2
        for i, n in enumerate(group):
            pos[n] = (col_x[layer], pad_top + extra + i * gap_y)
            layer_of[n] = layer
    width = pad_x + (len(used) - 1) * gap_x + nw + pad_x
    height = pad_top + max_rows * gap_y + pad_bot
    return pos, width, height, layer_of


_KIND_META = {
    "steel": ("Axle", "#d97757"),
    "measure": ("Measure", "#d4a017"),
    "iso": ("ISO", "#4c6ad9"),
    "ok": ("Ductile", "#2f9e63"),
    "risk": ("Brittle", "#c45c3a"),
    "motor": ("Motor", "#6d5bd0"),
    "sem": ("SEM", "#2a9d8f"),
    "process": ("Process", "#3a9e5f"),
}

_REL_LABEL = {
    "made_of": "made of",
    "measured_by": "measured by",
    "carries": "carries",
    "fails_as": "fails as",
    "governed_by": "governed by",
    "inversely_related_to": "inversely related",
    "positively_related_to": "positively related",
    "shows": "shows",
    "indicates": "indicates",
    "has_part": "has part",
    "risk": "risk",
    "loads": "loads",
    "instance_of": "instance of",
    "type": "type",
    "used_for": "used for",
    "described_in": "described in",
    # workflow relations
    "first_step": "first step",
    "followed_by": "→ then",
    "reveals": "reveals",
    "prepares_for": "prepares for",
    "requires": "requires",
    "reduces": "reduces",
    "promotes": "promotes",
    "verifies": "verifies",
    "guides": "guides",
}


def _card(x: float, y: float, nw: float, nh: float, node: str, dark: bool, uid: int) -> list[str]:
    kind, fill, stroke = _node_colors(node, dark)
    return [
        f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        f'<circle cx="{x + 18:.0f}" cy="{y + nh / 2:.0f}" r="5" fill="{stroke}"/>',
        f'<text x="{x + nw / 2 + 8:.0f}" y="{y + nh / 2 + 5:.0f}" text-anchor="middle" '
        f'fill="{"#f2f4f8" if dark else stroke}" font-size="14.5" font-weight="700" font-family="{_SVG_FONT}">'
        f"{_svg_words(_label(node))}</text>",
    ]


def edges_to_svg(edges: list[dict], *, dark: bool = False) -> str:
    """Connected left-to-right graph with one copy of each node and gutter labels."""
    nodes: list[str] = []
    for e in edges:
        for n in (e["from"], e["to"]):
            if n not in nodes:
                nodes.append(n)
    if not nodes:
        return ""

    uid = abs(hash(tuple((e["from"], e["relation"], e["to"]) for e in edges))) % 100000
    nw, nh = 210, 58
    pos, width, height, layer_of = _layout_nodes(nodes, edges, nw, nh)
    bg = "#141822" if dark else "#fbf8f3"
    ink = "#f2f4f8" if dark else "#2c2824"
    muted = "#9aa3b5" if dark else "#8a847a"
    pill_bg = "#1c2230" if dark else "#ffffff"
    pill_stroke = "#2d3548" if dark else "#ece6dc"
    glow_a = "#6d5bd0" if dark else "#f3d6c4"
    glow_b = "#2a9d8f" if dark else "#d7e4ff"

    kinds_used = []
    for n in nodes:
        kind = _node_colors(n, dark)[0]
        if kind not in kinds_used:
            kinds_used.append(kind)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="100%" preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-label="Knowledge graph" xml:space="preserve">',
        "<defs>",
        f'<linearGradient id="kg-bg-{uid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{bg}"/>'
        f'<stop offset="100%" stop-color="{"#1a1f2e" if dark else "#f3efe8"}"/>'
        "</linearGradient>",
        f'<filter id="kg-soft-{uid}" x="-20%" y="-20%" width="140%" height="140%">'
        f'<feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="{"#000000" if dark else "#8a7a68"}" '
        f'flood-opacity="{"0.32" if dark else "0.12"}"/>'
        "</filter>",
    ]
    for kind, (_, accent) in _KIND_META.items():
        parts.append(
            f'<marker id="kg-arrow-{uid}-{kind}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6.5" markerHeight="6.5" orient="auto">'
            f'<path d="M 0 1.2 L 10 5 L 0 8.8 z" fill="{accent}"/></marker>'
        )
    parts.append("</defs>")
    parts.append(f'<rect width="{width:.0f}" height="{height:.0f}" rx="20" fill="url(#kg-bg-{uid})"/>')
    parts.append(f'<circle cx="40" cy="8" r="56" fill="{glow_a}" opacity="{"0.14" if dark else "0.22"}"/>')
    parts.append(
        f'<circle cx="{width - 30:.0f}" cy="{height - 6:.0f}" r="70" fill="{glow_b}" '
        f'opacity="{"0.12" if dark else "0.2"}"/>'
    )
    parts.append(
        f'<text x="22" y="30" fill="{ink}" font-size="15" font-weight="700" '
        f'font-family="{_SVG_FONT}">{_svg_words("Digital twin graph")}</text>'
    )
    parts.append(
        f'<text x="22" y="46" fill="{muted}" font-size="11" font-family="{_SVG_FONT}">'
        f"{_svg_words('Axle · motor · ISO · fracture')}</text>"
    )

    labels: list[tuple[float, float, float, str]] = []
    # Pre-calculate gutter X positions from actual node positions
    col_xs: dict[int, list[float]] = defaultdict(list)
    for n, (nx, ny) in pos.items():
        col_xs[layer_of[n]].append(nx)
    layers_sorted = sorted(col_xs)
    gutter_x: dict[tuple[int,int], float] = {}
    for a, b in zip(layers_sorted, layers_sorted[1:]):
        right_of_a = max(col_xs[a]) + nw
        left_of_b  = min(col_xs[b])
        gutter_x[(a, b)] = (right_of_a + left_of_b) / 2

    gutter_slots: dict[tuple[int, int], int] = defaultdict(int)
    title_clear = 72
    for e in edges:
        x1, y1 = pos[e["from"]]
        x2, y2 = pos[e["to"]]
        kind, _, accent = _node_colors(e["to"], dark)
        l1, l2 = layer_of[e["from"]], layer_of[e["to"]]
        rel = _REL_LABEL.get(str(e.get("relation", "")), str(e.get("relation", "")).replace("_", " "))
        tw = max(88, min(220, 9.0 * len(rel) + 30))
        angle = 0.0

        if l1 == l2:
            right = l1 < max(layer_of.values())
            sx = x1 + (nw if right else 0)
            ex = x2 + (nw if right else 0)
            sy, ey = y1 + nh / 2, y2 + nh / 2
            bump = (x1 + nw + 26) if right else (x1 - 26)
            path = f"M {sx:.1f} {sy:.1f} C {bump:.1f} {sy:.1f}, {bump:.1f} {ey:.1f}, {ex:.1f} {ey:.1f}"
            lx = bump + (tw / 2 + 4) * (1 if right else -1)
            ly = (sy + ey) / 2
        elif abs(l2 - l1) == 1:
            lkey = (min(l1, l2), max(l1, l2))
            slot = gutter_slots[lkey]
            gutter_slots[lkey] += 1
            gx = gutter_x.get(lkey, (x1 + nw + x2) / 2)
            if l2 > l1:
                sx, sy, ex, ey = x1 + nw, y1 + nh / 2, x2, y2 + nh / 2
            else:
                sx, sy, ex, ey = x1, y1 + nh / 2, x2 + nw, y2 + nh / 2
            mx = (sx + ex) / 2
            path = f"M {sx:.1f} {sy:.1f} C {mx:.1f} {sy:.1f}, {mx:.1f} {ey:.1f}, {ex:.1f} {ey:.1f}"
            import math as _math
            dx_line, dy_line = ex - sx, ey - sy
            angle = _math.degrees(_math.atan2(dy_line, dx_line))
            if angle > 90 or angle < -90:
                angle += 180
            if abs(angle) > 40:
                angle = 0.0
            lx = gx
            ly = sy + slot * 26 + (ey - sy) * 0.15
        else:
            # Long-range edges travel under the nodes so they never cross the title.
            sx, sy = x1 + nw / 2, y1 + nh
            ex, ey = x2 + nw / 2, y2 + nh
            skip = gutter_slots.get((min(l1, l2), max(l1, l2)), 0)
            gutter_slots[(min(l1, l2), max(l1, l2))] = skip + 1
            lane = height - 44 - skip * 26
            path = (
                f"M {sx:.1f} {sy:.1f} C {sx:.1f} {lane:.1f}, {ex:.1f} {lane:.1f}, {ex:.1f} {ey:.1f}"
            )
            lx, ly, angle = (sx + ex) / 2, lane, 0.0

        ly = min(max(ly, title_clear), height - 28)
        parts.append(
            f'<path d="{path}" fill="none" stroke="{accent}" stroke-width="1.8" '
            f'stroke-opacity="0.75" marker-end="url(#kg-arrow-{uid}-{kind})"/>'
        )
        labels.append((lx, ly, angle, rel, accent))

    for n, (x, y) in pos.items():
        parts.extend(_card(x, y, nw, nh, n, dark, uid))

    for lx, ly, angle, rel, accent in labels:
        tw = max(72, min(180, 8.6 * len(rel) + 24))
        # white backing rect, then coloured bold text
        parts.append(
            f'<g transform="translate({lx:.1f},{ly:.1f}) rotate({angle:.1f})">'
            f'<rect x="{-tw/2:.1f}" y="-12" width="{tw:.1f}" height="20" rx="10" '
            f'fill="{"#1c2230" if dark else "#ffffff"}" stroke="{accent}" stroke-width="1.3" opacity="0.95"/>'
            f'<text x="0" y="5" text-anchor="middle" fill="{accent}" '
            f'font-size="12.5" font-weight="700" font-family="{_SVG_FONT}">{_svg_words(rel)}</text>'
            f'</g>'
        )

    lx = 22
    ly = height - 14
    for kind in kinds_used:
        name, accent = _KIND_META[kind]
        parts.append(f'<circle cx="{lx}" cy="{ly - 3}" r="3.6" fill="{accent}"/>')
        parts.append(
            f'<text x="{lx + 9}" y="{ly}" fill="{muted}" font-size="10" '
            f'font-family="{_SVG_FONT}">{_svg_words(name)}</text>'
        )
        lx += 11 + 7.4 * len(name) + 16

    parts.append("</svg>")
    return "".join(parts)


def edges_to_mermaid(edges: list[dict], *, dark: bool = False) -> str:
    """Build a Mermaid flowchart from graph edges."""
    if dark:
        init = (
            "%%{init: {'theme':'base','themeVariables':{"
            "'darkMode':true,"
            "'primaryColor':'#1a1f2e','primaryTextColor':'#f2f4f8',"
            "'primaryBorderColor':'#d97757',"
            "'lineColor':'#9aa3b5','secondaryColor':'#163026',"
            "'tertiaryColor':'#2a2248','edgeLabelBackground':'#1c2230',"
            "'textColor':'#f2f4f8','tertiaryTextColor':'#f2f4f8',"
            "'fontSize':'16px','fontFamily':'Arial'},"
            "'flowchart':{'nodeSpacing':60,'rankSpacing':80,'padding':18,'htmlLabels':false}"
            "}}%%"
        )
        label_color = "#f2f4f8"
    else:
        init = (
            "%%{init: {'theme':'base','themeVariables':{"
            "'darkMode':false,"
            "'primaryColor':'#fff4ef','primaryTextColor':'#1a1a1a',"
            "'primaryBorderColor':'#d97757',"
            "'lineColor':'#8a847a','secondaryColor':'#eef3ff',"
            "'tertiaryColor':'#eefaf3','edgeLabelBackground':'#ffffff',"
            "'textColor':'#1a1a1a','tertiaryTextColor':'#1a1a1a',"
            "'fontSize':'16px','fontFamily':'Arial'},"
            "'flowchart':{'nodeSpacing':60,'rankSpacing':80,'padding':18,'htmlLabels':false}"
            "}}%%"
        )
        label_color = "#1a1a1a"
    lines = [init, "flowchart LR"]
    seen_nodes: set[str] = set()
    for e in edges:
        src, dst = e["from"], e["to"]
        rel = _REL_LABEL.get(str(e.get("relation", "")), str(e.get("relation", "")).replace("_", " "))
        if src not in seen_nodes:
            lines.append(f'  {src}["{_label(src)}"]')
            seen_nodes.add(src)
        if dst not in seen_nodes:
            lines.append(f'  {dst}["{_label(dst)}"]')
            seen_nodes.add(dst)
        lines.append(f"  {src} -->|{rel}| {dst}")
    if not seen_nodes:
        lines.append('  empty["No matching links"]')
        seen_nodes.add("empty")
    for n in seen_nodes:
        _kind, fill, stroke = _node_colors(n, dark)
        lines.append(f"  style {n} fill:{fill},stroke:{stroke},color:{label_color}")
    return "\n".join(lines)
