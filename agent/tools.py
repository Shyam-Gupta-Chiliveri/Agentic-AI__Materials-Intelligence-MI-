"""Pure tool functions the agent can call. No Streamlit imports."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from agent.knowledge_graph import query_knowledge_graph

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FAISS_DIR = BASE_DIR / "faiss_index_local"
MOTOR_SUMMARY = BASE_DIR / "data" / "motor" / "profile_summaries.csv"
SEM_MODEL_PATH = BASE_DIR / "sem_output" / "models" / "sem_classifier_final.pth"
if not SEM_MODEL_PATH.exists():
    SEM_MODEL_PATH = BASE_DIR / "sem_output" / "models" / "best_sem_model.pth"

FEATURE_COLS = [
    "C %", "Si %", "Mn %", "P %", "S %", "Cr %", "Ni %",
    "Mean HV10", "Min HV10", "Max HV10",
    "Avg Bending Force N", "Min Bending Force N", "Max Bending Force N",
    "Die-casting_encoded", "Diameter_encoded",
]

# Typical 4Cr13 axle chemistry if the user only gives hardness
DEFAULTS = {
    "C %": 0.40, "Si %": 0.35, "Mn %": 0.40, "P %": 0.02, "S %": 0.01,
    "Cr %": 13.0, "Ni %": 0.20,
    "Avg Bending Force N": 4400, "Min Bending Force N": 4100, "Max Bending Force N": 4700,
    "Die-casting_encoded": 0, "Diameter_encoded": 0,
}


def predict_axle_fracture(
    mean_hv10: float | None = None,
    min_hv10: float | None = None,
    max_hv10: float | None = None,
    c_pct: float | None = None,
    si_pct: float | None = None,
    mn_pct: float | None = None,
    p_pct: float | None = None,
    s_pct: float | None = None,
    cr_pct: float | None = None,
    ni_pct: float | None = None,
    avg_bending_force_n: float | None = None,
    die_casting: str | None = None,
    diameter: str | None = None,
) -> dict:
    """Predict ductile/brittle % for a hardened steel axle from HV10 and chemistry."""
    if mean_hv10 is None and min_hv10 is None and max_hv10 is None:
        return {
            "ok": False,
            "missing": ["mean_hv10"],
            "message": "Need at least Mean HV10 (Vickers hardness) to run the axle model.",
        }

    mean_hv = float(mean_hv10 if mean_hv10 is not None else (min_hv10 or max_hv10))
    min_hv = float(min_hv10 if min_hv10 is not None else mean_hv)
    max_hv = float(max_hv10 if max_hv10 is not None else mean_hv)

    used_defaults = []
    row = {k: v for k, v in DEFAULTS.items()}
    overrides = {
        "C %": c_pct, "Si %": si_pct, "Mn %": mn_pct, "P %": p_pct,
        "S %": s_pct, "Cr %": cr_pct, "Ni %": ni_pct,
        "Avg Bending Force N": avg_bending_force_n,
    }
    for key, val in overrides.items():
        if val is not None:
            row[key] = float(val)
        else:
            used_defaults.append(key)

    row["Mean HV10"] = mean_hv
    row["Min HV10"] = min_hv
    row["Max HV10"] = max_hv
    if avg_bending_force_n is None:
        used_defaults.append("bending force (typical 4Cr13)")

    if die_casting is not None:
        le_dc = joblib.load(MODELS_DIR / "le_die_casting.pkl")
        label = "Yes" if str(die_casting).lower() in {"yes", "1", "true", "y"} else "No"
        row["Die-casting_encoded"] = int(le_dc.transform([label])[0])
    if diameter is not None:
        le_d = joblib.load(MODELS_DIR / "le_diameter.pkl")
        dlabel = "Changed 7.8mm" if "7.8" in str(diameter) else "8mm"
        row["Diameter_encoded"] = int(le_d.transform([dlabel])[0])

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    duct_m = joblib.load(MODELS_DIR / "best_ductility_model.pkl")
    brit_m = joblib.load(MODELS_DIR / "best_brittleness_model.pkl")
    x = pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    xs = scaler.transform(x)
    ductile = float(np.clip(duct_m.predict(xs)[0], 0, 100))
    brittle = float(np.clip(brit_m.predict(xs)[0], 0, 100))
    total = ductile + brittle
    if total > 0:
        ductile, brittle = ductile / total * 100, brittle / total * 100

    label = "Brittle-dominant" if brittle >= ductile else "Ductile-dominant"
    return {
        "ok": True,
        "ductile_pct": round(ductile, 2),
        "brittle_pct": round(brittle, 2),
        "label": label,
        "mean_hv10": mean_hv,
        "used_typical_4Cr13_defaults": used_defaults,
        "note": "Ridge/Linear models trained on 4Cr13 hardened steel axle tests.",
    }


def estimate_hv10_from_sem(ductile_pct: float) -> dict:
    """Predict Mean HV10 from SEM-derived ductile% using a linear regression
    calibrated on the full 385,000-sample training dataset.

    Calibration (from data):  HV10 = 727.67 − 3.278 × ductile%
    R² = 0.528,  residual σ = ±19.5 HV10  (95% CI ≈ ±38 HV10)
    Training ductile% range: 15–46% (mechanical test); SEM pixel values may
    fall outside this range — extrapolation is noted in the output.
    """
    # Calibration constants from OLS on 385,000 samples
    SLOPE     = -3.2776
    INTERCEPT = 727.67
    RESID_STD = 19.5   # ±1σ prediction interval in HV10

    hv_pred = SLOPE * ductile_pct + INTERCEPT
    hv_lo   = round(hv_pred - RESID_STD, 0)
    hv_hi   = round(hv_pred + RESID_STD, 0)
    hv_mid  = round(hv_pred, 0)
    brittle_pct = round(100.0 - ductile_pct, 1)

    # Metallurgical interpretation
    if hv_mid >= 670:
        temper_note = "High hardness — low tempering temperature or insufficient temper. High brittleness risk."
    elif hv_mid >= 620:
        temper_note = "Mid-to-high hardness — typical hardened + lightly tempered 4Cr13 axle steel."
    elif hv_mid >= 550:
        temper_note = "Mid hardness — well-tempered 4Cr13. Good ductile–brittle balance."
    elif hv_mid >= 450:
        temper_note = "Lower hardness — heavily tempered or longer tempering time. Ductile-dominant behaviour."
    else:
        temper_note = "Low hardness — may indicate over-tempering, annealing, or softer steel grade. Verify composition."

    in_training_range = 15.0 <= ductile_pct <= 46.0
    extrapolation_note = (
        "" if in_training_range else
        f" (Note: SEM ductile% {ductile_pct:.0f}% is outside the training data range "
        f"15–46% — result is an extrapolation.)"
    )

    return {
        "ok": True,
        "sem_ductile_pct": round(ductile_pct, 1),
        "sem_brittle_pct": brittle_pct,
        "estimated_hv10": hv_mid,
        "hv10_range": f"{hv_lo:.0f}–{hv_hi:.0f} HV10",
        "hv10_lower": hv_lo,
        "hv10_upper": hv_hi,
        "calibration": "Linear OLS on 385,000 training samples: HV10 = 727.67 − 3.278 × ductile%  |  R²=0.53  σ=±19.5 HV10",
        "metallurgical_note": temper_note + extrapolation_note,
        "validation": "Confirm with Vickers HV10 test per ISO 6507.",
    }


def classify_sem_image(image_path: str) -> dict:
    """Classify an SEM fracture image as dimples (ductile) or cleavages (brittle)."""
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    import tifffile
    from PIL import Image

    path = Path(image_path)
    if not path.exists():
        return {"ok": False, "message": f"Image not found: {image_path}"}

    name = path.name.lower()
    if name.endswith((".tif", ".tiff")):
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.array(Image.open(path))

    if img is None:
        return {"ok": False, "message": "Could not read the image."}

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    else:
        gray = img

    class _SEM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = smp.Unet(
                encoder_name="resnet50", encoder_weights=None,
                in_channels=3, classes=2, activation=None,
                decoder_attention_type="scse",
            )
            self.dropout = nn.Dropout2d(0.3)

        def forward(self, x):
            return self.dropout(self.model(x))

    device = torch.device("cpu")
    model = _SEM()
    state = torch.load(str(SEM_MODEL_PATH), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    rgb = gray if len(gray.shape) == 2 else gray
    if len(rgb.shape) == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (512, 512)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
        seg = probs.argmax(0).numpy()
    ductile = float((seg == 0).mean() * 100)
    brittle = float((seg == 1).mean() * 100)
    label = "Ductile (Dimples)" if ductile >= brittle else "Brittle (Cleavages)"
    return {
        "ok": True,
        "label": label,
        "ductile_pct": round(ductile, 2),
        "brittle_pct": round(brittle, 2),
        "confidence_pct": round(max(ductile, brittle), 2),
        "file": path.name,
    }


_FAISS_DB = None


def _load_faiss():
    global _FAISS_DB
    if _FAISS_DB is not None:
        return _FAISS_DB
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS

    if not FAISS_DIR.exists():
        return None
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    _FAISS_DB = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    return _FAISS_DB


def search_iso_standards(query: str, top_k: int = 4) -> dict:
    """Search local ISO/DIN / materials PDFs via FAISS."""
    db = _load_faiss()
    if db is None:
        return {"ok": False, "message": "Knowledge base not built yet. Open the RAG app once to index PDFs."}

    hits = db.similarity_search_with_score(query, k=top_k)
    passages = []
    for doc, score in hits:
        passages.append({
            "source": Path(doc.metadata.get("source", "?")).name,
            "page": int(doc.metadata.get("page", 0)) + 1,
            "score": round(float(score), 3),
            "text": doc.page_content[:700],
        })
    return {"ok": True, "query": query, "passages": passages}


def assess_motor_session(profile_id: int | None = None) -> dict:
    """Summarise plant motor copper-winding temperature and shaft torque load."""
    if not MOTOR_SUMMARY.exists():
        return {"ok": False, "message": "Motor summaries missing. Expected data/motor/profile_summaries.csv"}

    df = pd.read_csv(MOTOR_SUMMARY)
    if profile_id is None:
        hottest = df.sort_values("winding_max_c", ascending=False).iloc[0]
        hardest = df.sort_values("torque_max_nm", ascending=False).iloc[0]
        return {
            "ok": True,
            "source": "plant motor session",
            "n_sessions": int(len(df)),
            "hottest_winding_session": _row(hottest),
            "highest_torque_session": _row(hardest),
            "hint": "Pass profile_id (e.g. 29) for one driving cycle, or leave empty for extremes.",
        }

    hit = df[df["profile_id"] == int(profile_id)]
    if hit.empty:
        return {"ok": False, "message": f"Unknown profile_id {profile_id}. Valid: {sorted(df.profile_id.tolist())[:12]}…"}
    row = _row(hit.iloc[0])
    row["interpretation"] = _motor_interp(row)
    return {"ok": True, "source": "plant motor session", **row}


def _row(s) -> dict:
    return {
        "profile_id": int(s["profile_id"]),
        "duration_min": float(s["duration_min"]),
        "speed_mean_rpm": float(s["speed_mean_rpm"]),
        "speed_max_rpm": float(s["speed_max_rpm"]),
        "torque_mean_nm": float(s["torque_mean_nm"]),
        "torque_max_nm": float(s["torque_max_nm"]),
        "winding_mean_c": float(s["winding_mean_c"]),
        "winding_max_c": float(s["winding_max_c"]),
        "pm_max_c": float(s["pm_max_c"]),
        "thermal_risk": float(s["thermal_risk"]),
        "shaft_load_risk": float(s["shaft_load_risk"]),
    }


def _motor_interp(row: dict) -> str:
    bits = []
    if row["winding_max_c"] >= 120:
        bits.append("Copper windings are very hot (>120 °C) — insulation and thermal stress risk.")
    elif row["winding_max_c"] >= 90:
        bits.append("Winding temperature is elevated.")
    else:
        bits.append("Winding temperature is moderate.")
    if row["shaft_load_risk"] >= 80:
        bits.append("Peak torque is near the session maximum — high mechanical load on the shaft / axle carrier.")
    elif row["shaft_load_risk"] >= 50:
        bits.append("Medium shaft torque load.")
    else:
        bits.append("Shaft torque load is relatively low.")
    return " ".join(bits)


from agent.knowledge_graph import query_knowledge_graph


def detect_conflict(axle: dict | None, sem: dict | None, threshold: float = 20.0) -> dict | None:
    """Flag when tabular axle model and SEM image disagree on brittle %."""
    if not (axle and axle.get("ok") and sem and sem.get("ok")):
        return None
    delta = abs(float(axle["brittle_pct"]) - float(sem["brittle_pct"]))
    if delta < threshold:
        return {
            "conflict": False,
            "delta_brittle_pct": round(delta, 2),
            "message": "Axle model and SEM surface agree on the dominant fracture mode.",
        }
    return {
        "conflict": True,
        "delta_brittle_pct": round(delta, 2),
        "axle_brittle_pct": axle["brittle_pct"],
        "sem_brittle_pct": sem["brittle_pct"],
        "message": (
            f"CONFLICT: axle model {axle['brittle_pct']}% brittle vs SEM {sem['brittle_pct']}% brittle "
            f"(Δ={delta:.1f} pp). Do not sign off — re-check heat treatment vs surface vs core."
        ),
    }


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "predict_axle_fracture",
            "description": "Predict ductile/brittle % of a hardened steel axle from Vickers HV10 and optional chemistry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mean_hv10": {"type": "number", "description": "Mean Vickers hardness HV10"},
                    "min_hv10": {"type": "number"},
                    "max_hv10": {"type": "number"},
                    "c_pct": {"type": "number"},
                    "cr_pct": {"type": "number"},
                    "avg_bending_force_n": {"type": "number"},
                    "die_casting": {"type": "string", "description": "Yes or No"},
                    "diameter": {"type": "string", "description": "8mm or 7.8mm"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_sem_image",
            "description": "Classify an already-uploaded SEM fracture image. Use image_path exactly as provided in the system note.",
            "parameters": {
                "type": "object",
                "properties": {"image_path": {"type": "string"}},
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_iso_standards",
            "description": "Search ISO/DIN standards and materials textbooks (hardness, grain size, sample prep, SEM).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_motor_session",
            "description": "Assess this plant's copper winding temperature and shaft torque for a motor session. Use profile_id if the user gave one (e.g. 29).",
            "parameters": {
                "type": "object",
                "properties": {"profile_id": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "Query the semantic digital-twin graph (axle, HV10, ISO, SEM, motor windings, shaft load).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_edges": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]


DISPATCH = {
    "predict_axle_fracture": predict_axle_fracture,
    "classify_sem_image": classify_sem_image,
    "estimate_hv10_from_sem": estimate_hv10_from_sem,
    "search_iso_standards": search_iso_standards,
    "assess_motor_session": assess_motor_session,
    "query_knowledge_graph": query_knowledge_graph,
}


def run_tool(name: str, args: dict) -> dict:
    fn = DISPATCH.get(name)
    if not fn:
        return {"ok": False, "message": f"Unknown tool {name}"}
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return {"ok": False, "message": f"Bad arguments for {name}: {e}"}
    except Exception as e:
        return {"ok": False, "message": f"{name} failed: {e}"}


def tools_as_json() -> str:
    return json.dumps(TOOL_SPECS, indent=2)
