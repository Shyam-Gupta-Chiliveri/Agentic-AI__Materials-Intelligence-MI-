import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image
import tifffile
import io
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEM Image Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #2c7bb6; text-align: center; margin-bottom: 0.5rem; }
    .sub-header  { font-size: 1rem; color: #666; text-align: center; margin-bottom: 1.5rem; }
    .result-ductile  { background: #d4edda; padding: 1.2rem; border-radius: 10px;
                       border-left: 6px solid #28a745; margin: 1rem 0; }
    .result-brittle  { background: #f8d7da; padding: 1.2rem; border-radius: 10px;
                       border-left: 6px solid #dc3545; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "sem_output" / "models" / "sem_classifier_final.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "sem_output" / "models" / "best_sem_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model definition (must match training) ───────────────────────────────────
class DuctileBrittleSegmentationModel(nn.Module):
    def __init__(self, encoder='resnet50', encoder_weights=None,
                 classes=2, dropout=0.3):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None,
            decoder_attention_type='scse'
        )
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading SEM classifier model…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = DuctileBrittleSegmentationModel(encoder_weights=None)
    state = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
    # Handle various checkpoint formats
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(image_array, size=512):
    """Convert any SEM image to a 512×512 3-channel float tensor."""
    img = image_array.copy()

    # Grayscale → 3-channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0

    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

# ─── Inference ────────────────────────────────────────────────────────────────
def predict(model, image_array):
    tensor = preprocess_image(image_array).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)                        # (1, 2, 512, 512)
        probs  = F.softmax(logits, dim=1)[0]          # (2, 512, 512)
        seg_map = probs.argmax(dim=0).cpu().numpy()   # (512, 512)

    ductile_pct = float((seg_map == 0).mean() * 100)
    brittle_pct = float((seg_map == 1).mean() * 100)
    label       = "Ductile (Dimples)" if ductile_pct >= brittle_pct else "Brittle (Cleavages)"
    confidence  = max(ductile_pct, brittle_pct)

    return label, confidence, ductile_pct, brittle_pct, seg_map, probs.cpu().numpy()

def make_overlay(original_gray, seg_map):
    """Colour-coded segmentation overlay on the original image."""
    rgb = cv2.cvtColor(
        cv2.normalize(original_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    )
    rgb = cv2.resize(rgb, (512, 512))
    overlay = rgb.copy()
    overlay[seg_map == 0] = [0,   200, 100]   # green  → ductile
    overlay[seg_map == 1] = [220,  50,  50]   # red    → brittle
    blended = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)
    return blended

def load_image_file(uploaded):
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(('.tif', '.tiff')):
        img = tifffile.imread(io.BytesIO(data))
    else:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.array(Image.open(io.BytesIO(data)))
    if img is None:
        raise ValueError("Could not decode image.")
    return img

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<h1 class="main-header">🔬 SEM Fracture Surface Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a Scanning Electron Microscope image to classify it as <b>Ductile</b> or <b>Brittle</b> fracture</p>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
**Model:** U-Net + ResNet50  
**Task:** Ductile vs Brittle fracture segmentation  
**Input size:** 512 × 512 px  
**Val. accuracy:** ~90.7 %  
**Training images:** 2 000
        """)
        st.markdown("---")
        st.header("🎨 Classes")
        st.markdown("""
- 🟢 **Ductile (Dimples)** — rounded cup-like micro-voids
- 🔴 **Brittle (Cleavages)** — flat faceted fracture surfaces
        """)
        st.markdown("---")
        st.header("📁 Supported Formats")
        st.markdown("`.tif` `.tiff` `.png` `.jpg` `.bmp`")

        if MODEL_PATH.exists():
            st.success(f"✅ Model loaded from\n`{MODEL_PATH.name}`")
        else:
            st.error("❌ Model file not found")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model()
    if model is None:
        st.error(f"❌ Model not found at `{MODEL_PATH}`\n\nPlease run the training notebook first:\n`notebooks/03_SEM_Ductile_Brittle_Classification.ipynb`")
        return

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("---")
    uploaded = st.file_uploader(
        "📤 Upload a SEM image",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
        help="Grayscale or RGB SEM fracture surface image"
    )

    # ── Demo with reference images ────────────────────────────────────────────
    if uploaded is None:
        ref_ductile = BASE_DIR / "Dimples_with_Ductility.png"
        ref_brittle = BASE_DIR / "Cleavages_with_Brittleness.jpg"
        if ref_ductile.exists() or ref_brittle.exists():
            st.info("👆 Upload an image above, or click a demo button below:")
            c1, c2 = st.columns(2)
            with c1:
                if ref_ductile.exists() and st.button("🟢 Demo: Ductile sample"):
                    st.session_state.demo = str(ref_ductile)
            with c2:
                if ref_brittle.exists() and st.button("🔴 Demo: Brittle sample"):
                    st.session_state.demo = str(ref_brittle)

        if 'demo' in st.session_state:
            img_array = cv2.imread(st.session_state.demo, cv2.IMREAD_GRAYSCALE)
            if img_array is not None:
                _run_prediction(model, img_array, Path(st.session_state.demo).name)
        return

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        img_array = load_image_file(uploaded)
        if len(img_array.shape) == 3 and img_array.shape[2] in (3, 4):
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY if img_array.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        else:
            gray = img_array
        _run_prediction(model, gray, uploaded.name)
    except Exception as e:
        st.error(f"❌ Could not process image: {e}")


def _run_prediction(model, gray, filename):
    with st.spinner("🔍 Analysing fracture surface…"):
        label, confidence, ductile_pct, brittle_pct, seg_map, probs = predict(model, gray)

    overlay = make_overlay(gray, seg_map)

    # ── Results layout ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📷 Original")
        disp = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        st.image(cv2.resize(disp, (400, 400)), caption=filename, use_container_width=True)
    with col2:
        st.subheader("🗺️ Segmentation")
        st.image(cv2.resize(overlay, (400, 400)),
                 caption="🟢 Ductile  🔴 Brittle", use_container_width=True)
    with col3:
        st.subheader("📊 Result")
        css_class = "result-ductile" if "Ductile" in label else "result-brittle"
        icon      = "🟢" if "Ductile" in label else "🔴"
        st.markdown(f"""
<div class="{css_class}">
<h3>{icon} {label}</h3>
<p><b>Confidence:</b> {confidence:.1f}%</p>
</div>""", unsafe_allow_html=True)

        st.markdown("**Pixel breakdown:**")
        st.progress(ductile_pct / 100, text=f"🟢 Ductile: {ductile_pct:.1f}%")
        st.progress(brittle_pct / 100, text=f"🔴 Brittle: {brittle_pct:.1f}%")

        st.markdown("---")
        st.markdown(f"**File:** `{filename}`")
        st.markdown(f"**Device:** `{DEVICE}`")
        st.markdown(f"**Image size:** `{gray.shape[1]}×{gray.shape[0]}`")


if __name__ == "__main__":
    main()
