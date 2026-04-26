import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import timm
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import io, base64, datetime, os
import pandas as pd

st.set_page_config(
    page_title="RetinaAI — DR Screening",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS — Light main area, dark sidebar
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

/* ── Main area (light) ─────────────────────────────────── */
.stApp { background: #f1f5f9; }
.main  { background: #f1f5f9; }
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar (dark) ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #94a3b8;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] label {
    color: #64748b !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSlider * { color: #64748b !important; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #94a3b8 !important;
    text-transform: none !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #1e3a5f !important;
    color: #60a5fa !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    width: 100% !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
    margin: 1rem 0 !important;
}

/* ── Topbar ─────────────────────────────────────────────── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2rem; height: 58px;
    background: white;
    border-bottom: 1px solid #e2e8f0;
    margin: 0 -2rem 1.5rem -2rem;
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}
.logo-wrap { display: flex; align-items: center; gap: 10px; }
.logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 1rem;
}
.logo-text { font-size: 1rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; }
.logo-sub  { font-size: 0.6rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; }
.topbar-right { display: flex; align-items: center; gap: 12px; }
.topbar-badge {
    background: #eff6ff; border: 1px solid #bfdbfe;
    color: #2563eb; font-size: 0.68rem; font-weight: 600;
    padding: 4px 10px; border-radius: 20px;
}
.topbar-dot {
    display: inline-block; width: 6px; height: 6px;
    background: #22c55e; border-radius: 50%;
    box-shadow: 0 0 6px #22c55e; vertical-align: middle;
    margin-right: 4px;
    animation: pulse 2s ease infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-bottom: 2px solid #e2e8f0;
    gap: 0; padding: 0 0.5rem;
    border-radius: 12px 12px 0 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border: none;
    color: #64748b; font-size: 0.84rem; font-weight: 500;
    padding: 0.85rem 1.4rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: #2563eb; border-bottom: 2px solid #2563eb;
    background: transparent; font-weight: 600;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent; padding: 1.5rem 0 0 0;
}

/* ── Cards ─────────────────────────────────────────────── */
.card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.card-title {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #2563eb; margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 8px;
}
.card-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #e2e8f0, transparent);
}

/* ── Result banner ─────────────────────────────────────── */
.rbanner {
    border-radius: 16px; padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    display: flex; align-items: center;
    gap: 1.5rem; flex-wrap: wrap;
    border: 1px solid;
}
.rb0 { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border-color: #86efac; }
.rb1 { background: linear-gradient(135deg,#fefce8,#fef9c3); border-color: #fde047; }
.rb2 { background: linear-gradient(135deg,#fff7ed,#ffedd5); border-color: #fdba74; }
.rb3 { background: linear-gradient(135deg,#fef2f2,#fee2e2); border-color: #fca5a5; }
.rb4 { background: linear-gradient(135deg,#faf5ff,#f3e8ff); border-color: #d8b4fe; }

/* ── Grade value in banner ─────────────────────────────── */
.grade-num {
    font-size: 3rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.grade-sub { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b; margin-bottom: 6px; }

/* ── Badges ─────────────────────────────────────────────── */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
}
.b0 { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.b1 { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
.b2 { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
.b3 { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.b4 { background: #f3e8ff; color: #7e22ce; border: 1px solid #d8b4fe; }

/* ── Prob bars ──────────────────────────────────────────── */
.prow { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
.plbl { width: 100px; font-size: 0.76rem; color: #64748b; }
.pbg  { flex: 1; background: #f1f5f9; border-radius: 4px; height: 6px; overflow: hidden; }
.pfill{ height: 6px; border-radius: 4px; }
.pval { width: 42px; text-align: right; font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace; color: #64748b; }

/* ── Gauge ──────────────────────────────────────────────── */
.gauge-wrap { text-align: center; }
.gauge-lbl  { font-size: 0.62rem; color: #94a3b8; letter-spacing: 0.1em;
              text-transform: uppercase; margin-top: 2px; }

/* ── Remarks box ────────────────────────────────────────── */
.remark-box {
    border-radius: 12px; padding: 1rem 1.25rem; margin-top: 0.5rem;
}
.remark0 { background: #f0fdf4; border: 1px solid #86efac; }
.remark1 { background: #fefce8; border: 1px solid #fde047; }
.remark2 { background: #fff7ed; border: 1px solid #fdba74; }
.remark3 { background: #fef2f2; border: 1px solid #fca5a5; }
.remark4 { background: #faf5ff; border: 1px solid #d8b4fe; }
.remark-title { font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
.remark-body  { font-size: 0.84rem; color: #475569; line-height: 1.7; }
.remark-item  { display: flex; gap: 8px; margin-bottom: 5px; }
.remark-dot   { width: 6px; height: 6px; border-radius: 50%;
                margin-top: 6px; flex-shrink: 0; }

/* ── Warning ────────────────────────────────────────────── */
.warn-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 10px; padding: 0.8rem 1rem;
    margin: 0.75rem 0; font-size: 0.82rem;
    color: #92400e; display: flex; gap: 8px; align-items: flex-start;
}

/* ── Disclaimer ─────────────────────────────────────────── */
.disclaimer {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 10px; padding: 0.75rem 1rem;
    font-size: 0.76rem; color: #78716c;
    line-height: 1.6; margin-top: 1rem;
}

/* ── Stat cards ─────────────────────────────────────────── */
.stat-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stat-val  { font-size: 1.6rem; font-weight: 800;
             font-family: 'JetBrains Mono', monospace; }
.stat-lbl  { font-size: 0.68rem; color: #94a3b8;
             text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }
.stat-pct  { font-size: 0.78rem; margin-top: 3px; font-weight: 600; }

/* ── Sidebar brand ──────────────────────────────────────── */
.sb-brand {
    background: linear-gradient(135deg, #1e3a5f, #1e2d4f);
    border-bottom: 1px solid #1e293b;
    padding: 1rem 1.25rem;
    margin: -1rem -1rem 1rem -1rem;
    display: flex; align-items: center; gap: 10px;
}
.sb-logo  { width: 28px; height: 28px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    border-radius: 7px; display: flex; align-items: center;
    justify-content: center; font-size: 0.9rem;
}
.sb-title { font-size: 0.9rem; font-weight: 700; color: #f1f5f9 !important; }
.sb-sub   { font-size: 0.58rem; color: #475569 !important;
            letter-spacing: 0.1em; text-transform: uppercase; }

/* ── Section header ─────────────────────────────────────── */
.sb-section {
    font-size: 0.62rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #334155 !important;
    margin: 1rem 0 0.5rem 0; padding-top: 0.75rem;
    border-top: 1px solid #1e293b;
}

/* ── Upload area ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    background: white !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563eb !important;
}
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.55rem 1.4rem !important;
    font-size: 0.84rem !important; font-weight: 600 !important;
    cursor: pointer !important; transition: opacity 0.2s !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}
.stButton button:hover { opacity: 0.88 !important; }

/* ── Error/alert ─────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: #fef2f2 !important;
    border: 1px solid #fca5a5 !important;
    border-radius: 10px !important; color: #b91c1c !important;
}

/* ── History table ──────────────────────────────────────── */
.stDataFrame { background: white !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
IMG_SIZE     = 256
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
GRADE_COLORS = ["#16a34a", "#ca8a04", "#ea580c", "#dc2626", "#9333ea"]
GRADE_ICONS  = ["✅", "🟡", "🟠", "🔴", "🟣"]
BADGE_CLS    = ["b0", "b1", "b2", "b3", "b4"]
RB_CLS       = ["rb0", "rb1", "rb2", "rb3", "rb4"]
REMARK_CLS   = ["remark0", "remark1", "remark2", "remark3", "remark4"]

# Clinical descriptions
GRADE_DESC = {
    0: "No retinopathy detected. Annual screening recommended. Optimize glycemic control.",
    1: "Mild NPDR — microaneurysms present. HbA1c optimization and 12-month follow-up indicated.",
    2: "Moderate NPDR — hemorrhages and hard exudates visible. Refer to ophthalmologist within 3–6 months.",
    3: "Severe NPDR — extensive hemorrhages, venous beading, IRMA. Urgent referral < 4 weeks. High risk of conversion to PDR.",
    4: "PDR — neovascularization of disc/retina. Immediate vitreoretinal referral. Pan-retinal photocoagulation or anti-VEGF indicated."
}

# Clinical remarks per grade (bullet-style)
GRADE_REMARKS = {
    0: [
        "No microaneurysms, hemorrhages or exudates identified.",
        "Maintain HbA1c < 7% and blood pressure < 130/80 mmHg.",
        "Annual dilated fundoscopy recommended.",
        "Reassure patient — no active retinal pathology found."
    ],
    1: [
        "Microaneurysms are the earliest sign of DR; monitor closely.",
        "Intensify glycemic management: target HbA1c < 7%.",
        "Repeat retinal examination in 6–12 months.",
        "Educate patient on modifiable risk factors (smoking, hypertension, dyslipidaemia)."
    ],
    2: [
        "Intraretinal hemorrhages and/or hard exudates present.",
        "Macular edema assessment via OCT is strongly advised.",
        "Refer to ophthalmology within 3–6 months for full evaluation.",
        "Optimise systemic risk factors; consider nephrology review."
    ],
    3: [
        "Extensive intraretinal hemorrhages in ≥ 4 quadrants (4-2-1 rule).",
        "High short-term risk of progression to proliferative DR (> 50% in 1 year).",
        "Urgent ophthalmology referral required within 4 weeks.",
        "Evaluate for clinically significant macular edema (CSME) with OCT.",
        "Strict blood pressure and lipid control — priority intervention."
    ],
    4: [
        "Neovascularization detected (disc and/or retina) — proliferative stage.",
        "Same-week vitreoretinal referral is mandatory.",
        "Pan-retinal photocoagulation (PRP) or intravitreal anti-VEGF injection indicated.",
        "Screen for vitreous hemorrhage and tractional retinal detachment.",
        "Hospital admission may be required if vision is acutely threatened."
    ]
}

RECOMMENDATIONS = {
    0: ("✅ Routine Annual Follow-up",   "#16a34a"),
    1: ("🟡 Intensified Monitoring — 6–12 months", "#ca8a04"),
    2: ("🟠 Ophthalmology Referral — 3–6 months",  "#ea580c"),
    3: ("🔴 Urgent Referral — within 4 weeks",     "#dc2626"),
    4: ("🟣 Immediate Treatment — same week",       "#9333ea"),
}

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
class h_sigmoid(nn.Module):
    def __init__(self): super().__init__(); self.relu = nn.ReLU6(inplace=True)
    def forward(self, x): return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self): super().__init__(); self.sigmoid = h_sigmoid()
    def forward(self, x): return x * self.sigmoid(x)

class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, inp, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y   = torch.cat([x_h, x_w], dim=2)
        y   = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * self.sigmoid(self.conv_h(x_h)) * self.sigmoid(self.conv_w(x_w))

class HybridModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.effnet = timm.create_model('efficientnet_b3', pretrained=False)
        self.eff_n_features = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Identity()
        self.effnet.global_pool = nn.Identity()
        self.eff_mhsa = nn.MultiheadAttention(embed_dim=self.eff_n_features, num_heads=8, batch_first=True)
        self.swin = timm.create_model('swinv2_tiny_window8_256', pretrained=False)
        self.swin_n_features = self.swin.head.in_features
        self.swin.head = nn.Identity()
        self.swin_coord_att = CoordinateAttention(inp=self.swin_n_features)
        fusion_dim = self.eff_n_features + self.swin_n_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(fusion_dim, 512),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        eff_feat   = self.effnet.forward_features(x)
        b, c, h, w = eff_feat.shape
        eff_tokens = eff_feat.flatten(2).transpose(1, 2)
        eff_att, _ = self.eff_mhsa(eff_tokens, eff_tokens, eff_tokens)
        eff_out    = torch.mean(eff_att, dim=1)
        swin_feat  = self.swin.forward_features(x)
        if swin_feat.dim() == 4:
            swin_feat = swin_feat.permute(0, 3, 1, 2)
        swin_att = self.swin_coord_att(swin_feat)
        swin_out = torch.mean(swin_att.flatten(2), dim=2)
        return self.classifier(torch.cat((eff_out, swin_out), dim=1))

# ── Google Drive file ID for the model weights ───────────────
# Replace with your actual file ID from Google Drive share link:
# https://drive.google.com/file/d/FILE_ID/view  →  copy FILE_ID
GDRIVE_FILE_ID = "1up0E_dCI4ZcEWmT9Lu_xGvDq88LUj6Jt"

@st.cache_resource
def load_model(path):
    """Load weights, auto-downloading from Google Drive if needed."""
    if not os.path.exists(path):
        if GDRIVE_FILE_ID and GDRIVE_FILE_ID != "YOUR_GDRIVE_FILE_ID_HERE":
            try:
                import gdown
                st.info("Downloading model weights from Google Drive...")
                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                               path, quiet=False)
            except Exception as e:
                return None, f"Download failed: {e}"
        else:
            return None, (
                "Model file not found. Set GDRIVE_FILE_ID in app.py "
                "or place best_model.pth next to app.py."
            )

    if not os.path.exists(path):
        return None, f"Model still not found at {path!r} after download attempt."

    try:
        m = HybridModel(num_classes=5).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        return m, True
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, tensor, target_class=None):
        tensor = tensor.to(DEVICE).unsqueeze(0)
        tensor.requires_grad_()
        out = self.model(tensor)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, target_class].backward()
        pooled = self.gradients.mean(dim=[0, 2, 3])
        cam = self.activations[0].clone()
        for i in range(cam.shape[0]):
            cam[i] *= pooled[i]
        cam = cam.mean(0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()
        return cam, target_class

def make_overlay(pil_img, cam, alpha=0.45):
    img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    hm  = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    hm  = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm  = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * hm + (1 - alpha) * img)

infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(model, pil_img):
    tensor = infer_tf(pil_img)
    with torch.no_grad():
        out   = model(tensor.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs

def show_gradcam(model, pil_img, pred, alpha=0.45):
    try:
        target_layer = model.effnet.blocks[-1][-1].conv_pwl
        gc  = GradCAM(model, target_layer)
        cam, _ = gc.generate(infer_tf(pil_img), target_class=pred)
        return make_overlay(pil_img, cam, alpha)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def risk_score(pred, probs):
    weights = [0, 25, 50, 75, 100]
    base = weights[pred]
    conf = float(probs[pred])
    return int(base * 0.7 + conf * 100 * 0.3)

def make_gauge_html(score, color):
    dash = score * 0.628 * 1.0
    gap  = 62.8 - dash
    return f"""
    <div class='gauge-wrap'>
        <svg width='120' height='76' viewBox='0 0 120 76'>
            <path d='M12,70 A50,50 0 0,1 108,70' fill='none'
                  stroke='#e2e8f0' stroke-width='9' stroke-linecap='round'/>
            <path d='M12,70 A50,50 0 0,1 108,70' fill='none'
                  stroke='{color}' stroke-width='9'
                  stroke-dasharray='{dash:.1f} {gap:.1f}' stroke-linecap='round'
                  style='filter:drop-shadow(0 0 4px {color}88);'/>
            <text x='60' y='68' text-anchor='middle' font-size='19' font-weight='800'
                  font-family='JetBrains Mono,monospace' fill='{color}'>{score}</text>
        </svg>
        <div class='gauge-lbl'>Risk Score / 100</div>
    </div>"""

def prob_bars_html(probs, pred):
    bars = ""
    for i, (name, prob) in enumerate(zip(LABEL_NAMES, probs)):
        pct  = prob * 100
        bold = f"color:{GRADE_COLORS[i]};font-weight:700;" if i == pred else ""
        bars += f"""<div class='prow'>
            <div class='plbl' style='{bold}'>{name}</div>
            <div class='pbg'><div class='pfill' style='width:{pct:.1f}%;
                 background:linear-gradient(90deg,{GRADE_COLORS[i]}88,{GRADE_COLORS[i]});'></div></div>
            <div class='pval' style='{bold}'>{pct:.1f}%</div>
        </div>"""
    return bars

def make_qr(text):
    import qrcode
    qr = qrcode.QRCode(version=1, box_size=5, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1e40af", back_color="#f8fafc")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def sanitize(text):
    """Convert text to fpdf-safe latin-1 string, replacing all non-latin-1 chars."""
    t = str(text)
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2012': '-', '\u2212': '-',
        '\u201c': '"', '\u201d': '"',
        '\u2018': "'", '\u2019': "'",
        '\u2026': '...', '\u2022': '*', '\u2023': '*',
        '\u2192': '->', '\u2190': '<-', '\u21d2': '=>',
        '\u2265': '>=', '\u2264': '<=',
        '\u00b0': ' deg', '\u00d7': 'x',
        '\u2248': '~=', '\u2260': '!=',
        '\u2705': '[OK]', '\u2714': '[OK]', '\u26a0': '[!]',
        '\U0001f7e1': '[!]', '\U0001f7e0': '[!!]',
        '\U0001f534': '[!!!]', '\U0001f7e3': '[URGENT]',
    }
    for src, dst in replacements.items():
        t = t.replace(src, dst)
    # Final safety: strip anything still outside latin-1
    return t.encode('latin-1', errors='replace').decode('latin-1')

def make_pdf(patient_info, pred, probs, confidence, pil_img, overlay_img, notes):
    from fpdf import FPDF

    s = sanitize

    # ── Layout constants (never use width=0 in cells/multi_cell) ──
    L  = 12       # left margin mm
    R  = 12       # right margin mm
    T  = 20       # top margin mm
    PW = 210      # A4 page width
    CW = PW - L - R   # usable content width = 186 mm

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(37, 99, 235)
            self.rect(0, 0, PW, 16, 'F')
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(255, 255, 255)
            self.set_xy(L, 3)
            # Explicit width = CW — never 0 in header/footer
            self.cell(CW, 10,
                      'RetinaAI  |  Diabetic Retinopathy Screening Report',
                      ln=True)

        def footer(self):
            self.set_y(-12)
            self.set_x(L)
            self.set_font('Helvetica', '', 7)
            self.set_text_color(120, 120, 120)
            self.cell(CW, 10,
                      f'FOR RESEARCH USE ONLY  |  Not a substitute for clinical diagnosis  |  '
                      f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
                      f'  |  Page {self.page_no()}',
                      align='C')

    # ── IMPORTANT: set_margins BEFORE add_page ────────────────
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(L, T, R)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Reset to left margin — call before every block of output
    def lm():
        pdf.set_x(L)

    def section(title):
        pdf.ln(4)
        lm()
        pdf.set_fill_color(239, 246, 255)
        pdf.set_draw_color(191, 219, 254)
        pdf.set_text_color(30, 64, 175)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(CW, 7, f'  {title.upper()}', fill=True, border='LB', ln=True)
        pdf.ln(2)

    def field(label, value, w_label=55):
        lm()
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(w_label, 5.5, s(label) + ':', ln=False)
        pdf.set_font('Helvetica', 'B', 8.5)
        pdf.set_text_color(30, 41, 59)
        # Explicit remaining width — never 0
        pdf.cell(CW - w_label, 5.5, s(str(value)), ln=True)

    # ── Patient Information ───────────────────────────────────
    section('Patient Information')
    for label, val in [
        ('Patient Name',        patient_info.get('name',       'N/A')),
        ('Date of Birth',       patient_info.get('dob',        'N/A')),
        ('Age',                 f"{patient_info.get('age', 'N/A')} years"),
        ('Sex',                 patient_info.get('sex',        'N/A')),
        ('Diabetes Type',       patient_info.get('diab_type',  'N/A')),
        ('HbA1c (%)',           f"{patient_info.get('hba1c', 'N/A')}%"),
        ('Diabetes Duration',   f"{patient_info.get('duration', 'N/A')} years"),
        ('Eye Examined',        patient_info.get('eye',        'N/A')),
        ('Exam Date',           datetime.datetime.now().strftime('%Y-%m-%d')),
        ('Referring Physician', patient_info.get('physician',  'N/A')),
    ]:
        field(label, val)

    # ── AI Analysis Result ────────────────────────────────────
    section('AI Analysis Result')
    grade_rgb = [(22,163,74),(202,138,4),(234,88,12),(220,38,38),(147,51,234)]
    r, g, b = grade_rgb[pred]

    lm()
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(r, g, b)
    pdf.cell(CW, 12, s(f'Grade {pred}  -  {LABEL_NAMES[pred]}'), ln=True)

    lm()
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(CW, 5.5, s(GRADE_DESC[pred]))
    pdf.ln(2)

    field('Model Confidence', f'{confidence:.1f}%')
    field('Risk Score',       f'{risk_score(pred, probs)} / 100')

    # ── Recommendation banner ─────────────────────────────────
    rec_title, _ = RECOMMENDATIONS[pred]
    pdf.ln(3)
    lm()
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(CW, 8, f'  {s(rec_title)}', fill=True, ln=True)
    pdf.ln(2)

    # ── Class Probabilities ───────────────────────────────────
    section('Class Probabilities')
    LBL_W = 36    # label column
    BAR_W = 95    # progress bar
    PCT_W = 18    # percentage — total = 149, well within CW=186
    for i, (name, prob) in enumerate(zip(LABEL_NAMES, probs)):
        lm()
        r2, g2, b2 = grade_rgb[i]
        is_pred = (i == pred)
        pdf.set_font('Helvetica', 'B' if is_pred else '', 8)
        pdf.set_text_color(r2, g2, b2)
        pdf.cell(LBL_W, 6, name, ln=False)

        bar_x = pdf.get_x()
        bar_y = pdf.get_y() + 1.5
        pdf.set_fill_color(226, 232, 240)
        pdf.rect(bar_x, bar_y, BAR_W, 3, 'F')
        pdf.set_fill_color(r2, g2, b2)
        pdf.rect(bar_x, bar_y, BAR_W * prob, 3, 'F')

        pdf.set_x(bar_x + BAR_W + 3)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(PCT_W, 6, f'{prob * 100:.1f}%', ln=True)
    pdf.ln(2)

    # ── Clinical Remarks ──────────────────────────────────────
    # Rule: always call lm() before multi_cell — never mix cell()+multi_cell()
    section('Clinical Remarks')
    for remark in GRADE_REMARKS[pred]:
        lm()
        pdf.set_font('Helvetica', '', 8.5)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(CW, 5.5, s('- ' + remark))

    # ── Retinal Images ────────────────────────────────────────
    section('Retinal Images')
    tmp_orig    = '/tmp/retina_orig.png'
    tmp_overlay = '/tmp/retina_cam.png'
    pil_img.resize((IMG_SIZE, IMG_SIZE)).save(tmp_orig)

    IMG_W = 88
    lm()
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(100, 116, 139)

    if overlay_img is not None:
        Image.fromarray(overlay_img).save(tmp_overlay)
        pdf.cell(IMG_W + 4, 5, 'Original Fundus Image',       align='C', ln=False)
        pdf.cell(IMG_W + 4, 5, 'Attention Map (Grad-CAM)',    align='C', ln=True)
        y_img = pdf.get_y()
        pdf.image(tmp_orig,    x=L,              y=y_img, w=IMG_W, h=58)
        pdf.image(tmp_overlay, x=L + IMG_W + 6,  y=y_img, w=IMG_W, h=58)
    else:
        pdf.cell(CW, 5, 'Original Fundus Image', ln=True)
        pdf.image(tmp_orig, x=L, y=pdf.get_y(), w=IMG_W, h=58)
    pdf.ln(62)

    # ── Clinician Notes ───────────────────────────────────────
    if notes and notes.strip():
        section('Clinician Notes')
        lm()
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(CW, 5.5, s(notes))

    # ── Disclaimer ────────────────────────────────────────────
    pdf.ln(4)
    lm()
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(120, 113, 108)
    pdf.multi_cell(CW, 4.5,
        'DISCLAIMER: This report is generated by an AI screening tool for research and '
        'educational purposes only. It must not replace professional medical evaluation '
        'by a qualified ophthalmologist or diabetologist. Clinical decisions must be '
        'made by licensed healthcare professionals.',
        border=1)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "batch_queue" not in st.session_state:
    st.session_state.batch_queue = []   # list of {"name": str, "bytes": bytes}
if "last_single_name" not in st.session_state:
    st.session_state.last_single_name = None

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Patient info + settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sb-brand'>
      <div class='sb-logo'>👁</div>
      <div>
        <div class='sb-title'>RetinaAI</div>
        <div class='sb-sub'>DR Screening Platform</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-section'>⚙ Configuration</div>", unsafe_allow_html=True)
    model_path = st.text_input("Model path", value="best_model.pth")
    show_cam   = st.checkbox("Show Grad-CAM heatmap", value=True)
    alpha_val  = st.slider("Heatmap opacity", 0.2, 0.8, 0.45, 0.05, disabled=not show_cam)

    st.markdown("<div class='sb-section'>👤 Patient Information</div>", unsafe_allow_html=True)
    p_name     = st.text_input("Full name", placeholder="e.g. John Doe")
    p_dob      = st.text_input("Date of birth", placeholder="DD/MM/YYYY")
    p_age      = st.number_input("Age (years)", 0, 120, 50)
    p_sex      = st.selectbox("Sex", ["Male", "Female", "Other"])

    st.markdown("<div class='sb-section'>🏥 Clinical Data</div>", unsafe_allow_html=True)
    p_eye      = st.selectbox("Eye examined", ["Right Eye (OD)", "Left Eye (OS)", "Both"])
    p_dtype    = st.selectbox("Diabetes type", ["Type 1", "Type 2", "Gestational", "Unknown"])
    p_hba1c    = st.number_input("HbA1c (%)", 4.0, 20.0, 7.0, 0.1, format="%.1f")
    p_dur      = st.number_input("Diabetes duration (yrs)", 0, 60, 5)
    p_physician = st.text_input("Referring physician", placeholder="Dr. Smith")
    p_notes    = st.text_area("Clinician notes", placeholder="Additional observations…", height=80)

# ─────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='topbar'>
  <div class='logo-wrap'>
    <div class='logo-icon'>👁</div>
    <div>
      <div class='logo-text'>RetinaAI</div>
      <div class='logo-sub'>Diabetic Retinopathy Screening</div>
    </div>
  </div>
  <div class='topbar-right'>
    <span style='font-size:0.72rem;color:#94a3b8;'>
      <span class='topbar-dot'></span>AI Online
    </span>
    <span class='topbar-badge'>🩺 CLINICIAN MODE</span>
    <span class='topbar-badge' style='background:#f0fdf4;border-color:#86efac;color:#15803d;'>
      RESEARCH USE ONLY
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬  Single Analysis",
    "📂  Batch Analysis",
    "📋  Session History"
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — SINGLE ANALYSIS
# ═════════════════════════════════════════════════════════════
with tab1:

    model, loaded = load_model(model_path)
    if loaded is not True:
        st.error(f"⚠️ {loaded}")

    # Upload
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📂 Upload Fundus Image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop a retinal fundus image (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="visible"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded and loaded is True:
        img_bytes = uploaded.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Auto-add to batch queue (once per unique filename)
        existing_names = [b["name"] for b in st.session_state.batch_queue]
        if uploaded.name not in existing_names:
            st.session_state.batch_queue.append({
                "name": uploaded.name,
                "bytes": img_bytes
            })

        with st.spinner("Analysing retinal image…"):
            pred, probs = predict(model, pil_img)
            confidence  = float(probs[pred]) * 100
            rs          = risk_score(pred, probs)
            overlay     = show_gradcam(model, pil_img, pred, alpha_val) if show_cam else None

        patient_info = dict(
            name=p_name, dob=p_dob, age=p_age, sex=p_sex,
            eye=p_eye, diab_type=p_dtype, hba1c=p_hba1c,
            duration=p_dur, physician=p_physician
        )

        # ── Result banner ─────────────────────────────────────
        st.markdown(f"""
        <div class='rbanner {RB_CLS[pred]}'>
          <div>
            <div class='grade-sub'>Severity Grade</div>
            <div class='grade-num' style='color:{GRADE_COLORS[pred]};'>Grade {pred}</div>
            <div style='margin-top:8px;'>
              <span class='badge {BADGE_CLS[pred]}'>{GRADE_ICONS[pred]} {LABEL_NAMES[pred]}</span>
              &nbsp;
              <span style='font-size:0.72rem;color:#94a3b8;'>{p_eye}</span>
            </div>
          </div>
          <div style='display:flex;gap:1rem;margin-left:auto;align-items:center;'>
            <div style='background:rgba(255,255,255,0.7);border:1px solid #e2e8f0;border-radius:12px;
                        padding:0.8rem 1.2rem;text-align:center;backdrop-filter:blur(4px);'>
              <div style='font-size:1.9rem;font-weight:800;font-family:"JetBrains Mono",monospace;
                          color:{GRADE_COLORS[pred]};'>{confidence:.1f}%</div>
              <div style='font-size:0.6rem;color:#94a3b8;letter-spacing:0.1em;
                          text-transform:uppercase;margin-top:2px;'>Confidence</div>
            </div>
            <div>{make_gauge_html(rs, GRADE_COLORS[pred])}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < 60:
            st.markdown("""
            <div class='warn-box'>
              ⚠️ <div><b>Low confidence result</b> — The model is uncertain about this prediction.
              Manual specialist review is strongly recommended before any clinical decision.</div>
            </div>""", unsafe_allow_html=True)

        # ── Images + probs ────────────────────────────────────
        c1, c2, c3 = st.columns([1.1, 1.1, 1])
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🖼 Original Fundus</div>", unsafe_allow_html=True)
            st.image(pil_img.resize((IMG_SIZE, IMG_SIZE)), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🔥 Attention Map (Grad-CAM)</div>", unsafe_allow_html=True)
            if show_cam and overlay is not None:
                st.image(overlay, use_container_width=True)
                st.markdown(
                    "<div style='font-size:0.68rem;color:#94a3b8;text-align:center;margin-top:4px;'>"
                    "🔴 High relevance &nbsp;·&nbsp; 🔵 Low relevance</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='height:160px;display:flex;align-items:center;justify-content:center;"
                    "color:#cbd5e1;font-size:0.82rem;'>Heatmap disabled</div>",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📊 Grade Probabilities</div>", unsafe_allow_html=True)
            st.markdown(prob_bars_html(probs, pred), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Clinical description + remarks ────────────────────
        rec_title, rec_color = RECOMMENDATIONS[pred]
        remarks_html = "".join([
            f"<div class='remark-item'>"
            f"<div class='remark-dot' style='background:{GRADE_COLORS[pred]};'></div>"
            f"<div style='font-size:0.84rem;color:#475569;'>{r}</div>"
            f"</div>"
            for r in GRADE_REMARKS[pred]
        ])

        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>🩺 Clinical Assessment & Remarks</div>
          <div style='font-size:0.87rem;color:#475569;line-height:1.7;margin-bottom:1rem;
                      padding:0.75rem 1rem;background:#f8fafc;border-radius:8px;
                      border-left:3px solid {GRADE_COLORS[pred]};'>
            {GRADE_DESC[pred]}
          </div>
          <div class='remark-box {REMARK_CLS[pred]}'>
            <div class='remark-title' style='color:{rec_color};'>{rec_title}</div>
            <div class='remark-body'>{remarks_html}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Export: PDF + QR ──────────────────────────────────
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📤 Export Report</div>", unsafe_allow_html=True)

        col_pdf, col_qr, col_save = st.columns([1, 1, 1])

        with col_pdf:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF…"):
                    pdf_bytes = make_pdf(
                        patient_info, pred, probs,
                        confidence, pil_img, overlay, p_notes
                    )
                fname = (
                    f"RetinaAI_{(p_name or 'report').replace(' ', '_')}"
                    f"_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
                )
                st.download_button(
                    "⬇ Download PDF", pdf_bytes, fname,
                    "application/pdf", use_container_width=True
                )

        with col_qr:
            qr_text = (
                f"RetinaAI Screening Result\n"
                f"Patient: {p_name or 'N/A'}\n"
                f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n"
                f"Eye: {p_eye}\n"
                f"Grade: {pred} - {LABEL_NAMES[pred]}\n"
                f"Confidence: {confidence:.1f}%\n"
                f"Risk Score: {rs}/100\n"
                f"HbA1c: {p_hba1c}%  |  Diabetes: {p_dur} yrs\n"
                f"[FOR RESEARCH USE ONLY]"
            )
            qr_bytes = make_qr(qr_text)
            st.image(qr_bytes, width=110)
            st.download_button(
                "⬇ Save QR Code", qr_bytes,
                "result_qr.png", "image/png",
                use_container_width=True
            )

        with col_save:
            if st.button("💾 Save to History", use_container_width=True):
                st.session_state.history.append(dict(
                    date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    name=p_name or "Anonymous",
                    age=p_age, sex=p_sex,
                    eye=p_eye,
                    hba1c=p_hba1c,
                    grade=pred,
                    label=LABEL_NAMES[pred],
                    confidence=f"{confidence:.1f}%",
                    risk_score=rs,
                    image_file=uploaded.name
                ))
                st.success("✅ Saved to history!")

        st.markdown("</div>", unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class='disclaimer'>
          ⚠️ <b>Medical Disclaimer:</b> This tool is intended for research and educational purposes only.
          Results must not replace clinical evaluation by a qualified ophthalmologist or diabetologist.
        </div>""", unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style='text-align:center;padding:4rem 1rem;'>
          <div style='font-size:4rem;opacity:0.15;margin-bottom:1rem;'>🔬</div>
          <div style='font-size:0.88rem;color:#94a3b8;line-height:1.8;'>
            Fill in patient information in the sidebar,<br>
            then upload a retinal fundus image above.
          </div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ═════════════════════════════════════════════════════════════
with tab2:

    model_b, loaded_b = load_model(model_path)
    if loaded_b is not True:
        st.error(f"⚠️ {loaded_b}")

    # Show auto-added images from single analysis
    if st.session_state.batch_queue:
        st.markdown(
            f"<div style='font-size:0.82rem;color:#64748b;margin-bottom:0.5rem;'>"
            f"🔗 <b>{len(st.session_state.batch_queue)}</b> image(s) "
            f"automatically queued from Single Analysis.</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📂 Add More Images (Optional)</div>", unsafe_allow_html=True)
    extra_files = st.file_uploader(
        "Upload additional fundus images for batch processing",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="visible",
        key="batch_extra"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Merge extra files into queue
    if extra_files:
        existing_names = [b["name"] for b in st.session_state.batch_queue]
        for f in extra_files:
            if f.name not in existing_names:
                st.session_state.batch_queue.append({
                    "name": f.name,
                    "bytes": f.read()
                })

    # Clear queue button
    if st.session_state.batch_queue:
        col_run, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑 Clear Queue"):
                st.session_state.batch_queue = []
                st.rerun()
        with col_run:
            run_batch = st.button(
                f"▶ Run Batch Analysis ({len(st.session_state.batch_queue)} image(s))",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#94a3b8;font-size:0.88rem;'>
          Upload an image in Single Analysis or add images above to populate the batch queue.
        </div>""", unsafe_allow_html=True)
        run_batch = False

    if run_batch and loaded_b is True and st.session_state.batch_queue:
        results = []
        prog = st.progress(0, text="Analysing…")
        total = len(st.session_state.batch_queue)

        for i, item in enumerate(st.session_state.batch_queue):
            try:
                img  = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
                p, probs = predict(model_b, img)
                rs   = risk_score(p, probs)
                conf = float(probs[p]) * 100
                results.append(dict(
                    filename   = item["name"],
                    grade      = p,
                    label      = LABEL_NAMES[p],
                    confidence = f"{conf:.1f}%",
                    risk_score = rs,
                    recommendation = RECOMMENDATIONS[p][0],
                    prob_NoDR  = f"{probs[0]*100:.1f}%",
                    prob_Mild  = f"{probs[1]*100:.1f}%",
                    prob_Mod   = f"{probs[2]*100:.1f}%",
                    prob_Severe = f"{probs[3]*100:.1f}%",
                    prob_PDR   = f"{probs[4]*100:.1f}%",
                ))
            except Exception as e:
                results.append(dict(
                    filename=item["name"], grade="ERR", label=str(e),
                    confidence="-", risk_score="-",
                    recommendation="-",
                    prob_NoDR="-", prob_Mild="-", prob_Mod="-",
                    prob_Severe="-", prob_PDR="-"
                ))
            prog.progress((i + 1) / total, text=f"Analysed {i+1}/{total}")

        df = pd.DataFrame(results)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📊 Batch Results</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Export Results CSV", csv,
            f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv", use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Grade distribution
        valid = [r for r in results if r['grade'] != 'ERR']
        if valid:
            from collections import Counter
            dist = Counter(r['label'] for r in valid)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📈 Grade Distribution</div>", unsafe_allow_html=True)
            cols = st.columns(5)
            for i, name in enumerate(LABEL_NAMES):
                with cols[i]:
                    cnt = dist.get(name, 0)
                    pct = cnt / len(valid) * 100 if valid else 0
                    st.markdown(f"""
                    <div class='stat-card'>
                      <div class='stat-val' style='color:{GRADE_COLORS[i]};'>{cnt}</div>
                      <div class='stat-lbl'>{name}</div>
                      <div class='stat-pct' style='color:{GRADE_COLORS[i]};'>{pct:.0f}%</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ═════════════════════════════════════════════════════════════
with tab3:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📋 Session Analysis History</div>", unsafe_allow_html=True)

    if st.session_state.history:
        df_h = pd.DataFrame(st.session_state.history)
        st.dataframe(df_h, use_container_width=True, hide_index=True)

        c_exp, c_clr = st.columns([2, 1])
        with c_exp:
            csv_h = df_h.to_csv(index=False).encode()
            st.download_button(
                "⬇ Export History CSV", csv_h,
                f"history_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv", use_container_width=True
            )
        with c_clr:
            if st.button("🗑 Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        # Quick stats
        total = len(st.session_state.history)
        from collections import Counter
        grade_dist = Counter(r['label'] for r in st.session_state.history)
        referral_needed = sum(1 for r in st.session_state.history if r['grade'] >= 2)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📈 Session Statistics</div>", unsafe_allow_html=True)
        sc = st.columns(4)
        with sc[0]:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-val' style='color:#2563eb;'>{total}</div>
              <div class='stat-lbl'>Total Analyses</div>
            </div>""", unsafe_allow_html=True)
        with sc[1]:
            cnt0 = grade_dist.get("No DR", 0)
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-val' style='color:#16a34a;'>{cnt0}</div>
              <div class='stat-lbl'>No DR</div>
              <div class='stat-pct' style='color:#16a34a;'>{cnt0/total*100:.0f}%</div>
            </div>""", unsafe_allow_html=True)
        with sc[2]:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-val' style='color:#ea580c;'>{referral_needed}</div>
              <div class='stat-lbl'>Referral Needed</div>
              <div class='stat-pct' style='color:#ea580c;'>Grade ≥ 2</div>
            </div>""", unsafe_allow_html=True)
        with sc[3]:
            cnt4 = grade_dist.get("Proliferative", 0)
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-val' style='color:#9333ea;'>{cnt4}</div>
              <div class='stat-lbl'>Proliferative</div>
              <div class='stat-pct' style='color:#9333ea;'>Urgent cases</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:2.5rem;color:#94a3b8;font-size:0.85rem;'>
          No analyses saved yet. Use the <b>Save to History</b> button in Single Analysis.
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)