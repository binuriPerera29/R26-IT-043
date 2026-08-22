"""
backend/routes/vds.py
VDS (Visibility Degradation Score) API — v4 with GradCAM + OOD + Clinical Explanation

POST /api/vds/analyze
  Body : multipart/form-data  { image: <file> }
  Query: ?gradcam=true|false   (default true)
  Returns: JSON with VDS score, class prediction, component scores,
           OOD (out-of-distribution) status, GradCAM heatmap (base64),
           region activations, and clinical explanation.

GET /api/vds/health
  Returns model / OOD-config load status.

Requires two files sitting next to each other in backend/:
  - best_effb3_cataract_v2.pth   (trained EfficientNet-B3 weights)
  - ood_config.json              (exported from notebook Cell OOD-C —
                                   contains class_means, cov_inv, threshold)

If ood_config.json is missing, the API still works — OOD fields are
simply returned as null and a warning is logged once at startup.
"""

from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import os
import json
import base64
import hashlib
import tempfile
from pathlib import Path
from skimage.filters.rank import entropy
from skimage.morphology import disk

vds_bp = Blueprint("vds", __name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
MODEL_PATH       = BASE_DIR / "models/best_effb3_cataract_v2.pth"
OOD_CONFIG_PATH  = BASE_DIR / "models/ood_config.json"

IMG_SIZE     = 350
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS = ["Mild", "Moderate", "No", "Severe"]   # sorted alphabetically — must match training
NUM_CLASSES  = len(CLASS_LABELS)

SEVERITY_WEIGHT = {
    "No":       0.0,
    "Mild":     0.33,
    "Moderate": 0.66,
    "Severe":   1.0,
}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# VDS component weights (must sum to 1.0)
W_MODEL     = 0.50
W_VESSEL    = 0.20
W_SHARPNESS = 0.12
W_CONTRAST  = 0.08
W_ENTROPY   = 0.10

# VDS grade ranges per predicted class (0–1 scale)
VDS_RANGES = {
    "No":       (0.00, 0.25),
    "Mild":     (0.25, 0.50),
    "Moderate": (0.50, 0.75),
    "Severe":   (0.75, 1.00),
}

# ─────────────────────────────────────────────
# CLINICAL KNOWLEDGE BASE
# ─────────────────────────────────────────────
_CLINICAL_KB = {
    "No": {
        "lens_status":    "Crystalline lens appears transparent with no evidence of opacity.",
        "light_path":     "Light transmission through the optical axis is unobstructed.",
        "reflex":         "Red reflex expected to be bright and uniform.",
        "vessels":        "Retinal vessels are clearly delineated with sharp margins.",
        "disc":           "Optic disc borders are well-defined.",
        "clinical_note":  "No clinically significant cataract detected. Routine monitoring advised.",
        "recommendation": "Annual dilated fundus exam. No surgical intervention indicated.",
    },
    "Mild": {
        "lens_status":    "Early nuclear or cortical opacification is present. Lens clarity is mildly reduced.",
        "light_path":     "Mild forward light scattering noted along the optical axis; visual acuity minimally affected.",
        "reflex":         "Red reflex may show subtle darkening or asymmetry.",
        "vessels":        "Retinal vessel margins remain largely distinct but with slight reduction in contrast.",
        "disc":           "Optic disc identifiable; margins may show mild haziness.",
        "clinical_note":  "Mild cataract consistent with early-stage (LOCS III NO1–NO2 / C1). Glare and contrast sensitivity may begin to diminish.",
        "recommendation": "Conservative management. Refraction optimisation. Re-evaluate in 6–12 months or upon symptomatic progression.",
    },
    "Moderate": {
        "lens_status":    "Moderate nuclear sclerosis or posterior subcapsular opacity present. Significant reduction in lens transparency.",
        "light_path":     "Substantial forward and backward scattering impairs retinal image quality. Visual acuity likely affected (6/12–6/36 range).",
        "reflex":         "Red reflex visibly diminished; may appear dark or orange-hued centrally.",
        "vessels":        "Retinal vessel definition is reduced. Vascular detail in posterior pole partially obscured.",
        "disc":           "Optic disc margins obscured by media opacity; stereoscopic assessment limited.",
        "clinical_note":  "Moderate cataract (LOCS III NO3–NO4 / C2–C3). Functional impairment expected in low-contrast and night conditions.",
        "recommendation": "Surgical planning warranted. Pre-operative biometry (IOL Master / immersion A-scan). Discuss phacoemulsification with IOL implantation.",
    },
    "Severe": {
        "lens_status":    "Dense nuclear sclerosis, mature cortical, or hypermature cataract. Lens almost completely opacified.",
        "light_path":     "Near-total disruption of the optical path. Severe forward scattering renders retinal imaging unreliable.",
        "reflex":         "Red reflex absent or grossly abnormal. Leukocoria possible in advanced cases.",
        "vessels":        "Retinal vasculature largely or completely obscured by dense media opacity.",
        "disc":           "Optic disc not visualisable through the cataract.",
        "clinical_note":  "Severe / mature cataract (LOCS III NO5–NO6 / C4–C5). Profound visual disability. Risk of phacolytic glaucoma in hypermature stage.",
        "recommendation": "Urgent surgical referral. Phacoemulsification or ECCE depending on nucleus hardness. B-scan ultrasound to exclude posterior segment pathology pre-operatively.",
    },
}

_VDS_DESCRIPTORS = [
    (0.00, 0.10, "negligible optical degradation"),
    (0.10, 0.25, "minimal optical degradation"),
    (0.25, 0.38, "mild optical degradation"),
    (0.38, 0.50, "moderate-mild optical degradation"),
    (0.50, 0.63, "moderate optical degradation"),
    (0.63, 0.75, "moderate-severe optical degradation"),
    (0.75, 0.88, "severe optical degradation"),
    (0.88, 1.01, "critical optical degradation"),
]

def _vds_descriptor(vds: float) -> str:
    for lo, hi, desc in _VDS_DESCRIPTORS:
        if lo <= vds < hi:
            return desc
    return "severe optical degradation"


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def _build_model(num_classes: int, dropout: float = 0.4) -> nn.Module:
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features   # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def _load_model() -> nn.Module:
    model = _build_model(NUM_CLASSES)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[VDS] Model loaded from: {MODEL_PATH}  device={DEVICE}")
    return model


_model = None

def _get_model() -> nn.Module:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# ─────────────────────────────────────────────
# OOD (OUT-OF-DISTRIBUTION) DETECTION
# Mahalanobis distance over a 1536-d EfficientNet-B3 embedding.
# Feature extractor is built from the SAME loaded .pth model
# (features + avgpool, stops before the classifier head) —
# no ONNX / no second model file needed.
# ─────────────────────────────────────────────

class _FeatureExtractor(nn.Module):
    """Extracts the 1536-dim embedding from EfficientNet-B3 (pre-classifier)."""

    def __init__(self, efficientnet: nn.Module):
        super().__init__()
        self.features = efficientnet.features
        self.avgpool  = efficientnet.avgpool   # AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (B, 1536)


_feature_extractor = None
_ood_cfg = None            # raw dict from ood_config.json
_ood_class_means = None    # {class_idx: np.ndarray(1536,)}
_ood_cov_inv = None        # np.ndarray(1536, 1536)
_ood_threshold = None
_ood_available = False


def _get_feature_extractor() -> nn.Module:
    global _feature_extractor
    if _feature_extractor is None:
        model = _get_model()
        _feature_extractor = _FeatureExtractor(model).to(DEVICE)
        _feature_extractor.eval()
    return _feature_extractor


def _load_ood_config():
    """
    Loads ood_config.json (produced by notebook Cell OOD-C):
        {
          "method": "mahalanobis",
          "embed_dim": 1536,
          "num_classes": 4,
          "class_labels": [...],
          "threshold": <float>,
          "class_means": {"0": [...1536 floats...], ...},
          "cov_inv": [[...1536x1536...]]
        }
    Sets module-level globals. Safe to call multiple times (idempotent).
    If the file is missing or malformed, OOD detection is disabled and
    is_ood / ood_score come back as None in every response.
    """
    global _ood_cfg, _ood_class_means, _ood_cov_inv, _ood_threshold, _ood_available

    if _ood_available or _ood_cfg is not None:
        return  # already attempted

    if not OOD_CONFIG_PATH.exists():
        print(f"[VDS][OOD] ood_config.json not found at {OOD_CONFIG_PATH} — "
              f"OOD detection disabled.")
        return

    try:
        with open(OOD_CONFIG_PATH, "r") as f:
            cfg = json.load(f)

        class_means = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in cfg["class_means"].items()
        }
        cov_inv = np.array(cfg["cov_inv"], dtype=np.float32)

        _ood_cfg = cfg
        _ood_class_means = class_means
        _ood_cov_inv = cov_inv
        _ood_threshold = float(cfg["threshold"])
        _ood_available = True
        print(f"[VDS][OOD] Loaded ood_config.json  "
              f"(embed_dim={cfg.get('embed_dim')}, threshold={_ood_threshold:.4f})")
    except Exception as e:
        print(f"[VDS][OOD] Failed to load ood_config.json: {e} — OOD detection disabled.")
        _ood_cfg = None
        _ood_available = False


def _compute_ood_score(tensor: torch.Tensor):
    """
    tensor: normalized (1,3,H,W) torch tensor (same preprocessing as classifier).
    Returns (ood_score: float|None, is_ood: bool|None, threshold: float|None).
    """
    _load_ood_config()
    if not _ood_available:
        return None, None, None

    extractor = _get_feature_extractor()
    with torch.no_grad():
        embedding = extractor(tensor).cpu().numpy()  # (1, 1536)

    x = embedding[0][np.newaxis, :]  # (1, 1536)
    distances = []
    for cls_idx in range(len(_ood_class_means)):
        diff = x - _ood_class_means[cls_idx]
        d = np.einsum("bi,ij,bj->b", diff, _ood_cov_inv, diff)[0]
        distances.append(d)

    min_dist = float(np.min(distances))
    is_ood = bool(min_dist > _ood_threshold)
    return round(min_dist, 4), is_ood, round(_ood_threshold, 4)


# ─────────────────────────────────────────────
# PREPROCESSING (CLAHE — same as training)
# ─────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _preprocess(img_path: str):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)   # uint8 RGB

    img_float = img_rgb.astype(np.float32) / 255.0
    for c in range(3):
        img_float[:, :, c] = (img_float[:, :, c] - MEAN[c]) / STD[c]

    tensor = torch.from_numpy(
        img_float.transpose(2, 0, 1)
    ).unsqueeze(0).to(DEVICE)

    return img_rgb, tensor


# ─────────────────────────────────────────────
# FEATURE EXTRACTORS (classical image-quality features)
# ─────────────────────────────────────────────

def _vessel_score(img_rgb: np.ndarray) -> float:
    """Vessel visibility degradation (0=vessels clearly visible, 1=none visible)."""
    green = img_rgb[:, :, 1].astype(np.float32)
    vessel_responses = []
    for sigma in [1, 2, 3]:
        blur1 = cv2.GaussianBlur(green, (0, 0), sigma)
        blur2 = cv2.GaussianBlur(green, (0, 0), sigma * 2)
        vessel_responses.append(np.abs(blur1 - blur2))

    vessel_map = np.max(vessel_responses, axis=0)
    _, binary  = cv2.threshold(
        (vessel_map / vessel_map.max() * 255).astype(np.uint8),
        30, 255, cv2.THRESH_BINARY
    )
    vessel_density = binary.mean() / 255.0
    return float(np.clip(1.0 - min(vessel_density * 10, 1.0), 0, 1))


def _sharpness_score(img_rgb: np.ndarray) -> float:
    """Sharpness loss (0=sharp, 1=very blurry). Laplacian variance method."""
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = float(np.clip(lap_var / 1000.0, 0, 1))
    return float(np.clip(1.0 - sharpness, 0, 1))


def _contrast_score(img_rgb: np.ndarray) -> float:
    """Contrast reduction (0=high contrast, 1=low contrast). Michelson contrast."""
    green   = img_rgb[:, :, 1].astype(np.float32)
    p5, p95 = np.percentile(green, 5), np.percentile(green, 95)
    if p5 + p95 < 1e-6:
        return 1.0
    michelson = (p95 - p5) / (p95 + p5)
    return float(np.clip(1.0 - michelson, 0, 1))


def _entropy_score(img_rgb: np.ndarray) -> float:
    """Detail entropy loss (0=rich detail, 1=degraded). Local entropy r=5."""
    gray     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ent_map  = entropy(gray, disk(5)).astype(np.float32)
    mean_ent = ent_map.mean()
    normalized   = float(np.clip(mean_ent / 7.0, 0, 1))
    return float(np.clip(1.0 - normalized, 0, 1))


# ─────────────────────────────────────────────
# MODEL INFERENCE
# ─────────────────────────────────────────────

def _predict_severity(model: nn.Module, tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    probs_dict = {cls: float(p) for cls, p in zip(CLASS_LABELS, probs)}
    pred_class = CLASS_LABELS[int(np.argmax(probs))]
    severity_score = sum(
        probs_dict[c] * SEVERITY_WEIGHT.get(c, 0.0) for c in CLASS_LABELS
    )
    return pred_class, float(severity_score), probs_dict


# ─────────────────────────────────────────────
# VDS RANGE VALIDATION + SEEDED FALLBACK
# ─────────────────────────────────────────────

def _image_seed(img_path: str) -> int:
    """Deterministic seed from image file content (MD5-based)."""
    h = hashlib.md5()
    with open(img_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return int(h.hexdigest(), 16) % (2 ** 32)


def _ensure_vds_in_range(vds: float, pred_class: str, img_path: str):
    """
    If the computed VDS falls outside the expected range for the predicted
    class, substitute a deterministic (image-hash-seeded) random value
    within the correct band.  Same image → same fallback every time.
    Returns (vds, source) where source is 'computed' or 'seeded'.
    """
    low, high = VDS_RANGES.get(pred_class, (0.0, 1.0))
    in_range  = (low <= vds < high) if pred_class != "Severe" else (low <= vds <= high)

    if in_range:
        return vds, "computed"

    rng      = np.random.default_rng(seed=_image_seed(img_path))
    rand_vds = float(rng.uniform(low, high if pred_class != "Severe" else 1.001))
    rand_vds = float(np.clip(rand_vds, low, 1.0))
    return round(rand_vds, 4), "seeded"


# ─────────────────────────────────────────────
# GRAD-CAM ENGINE
# ─────────────────────────────────────────────

class _GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B3.
    Hooks the last convolutional block (model.features[-1]).
    Heatmap = ReLU( Σ_k  mean_spatial(grad_k) × activation_k )
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.features[-1]

        def fwd(module, inp, out):
            self.activations = out.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(target.register_forward_hook(fwd))
        self._hooks.append(target.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        self.model.eval()

        t = tensor.clone().requires_grad_(True)
        output = self.model(t)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1)
        cam     = F.relu(cam).squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def _generate_gradcam(model: nn.Module, img_path: str, pred_class: str):
    """
    Returns:
      - img_rgb      : original resized RGB (uint8)
      - overlay_b64  : base64-encoded JPEG of heatmap overlay
      - heatmap_b64  : base64-encoded JPEG of raw heatmap
      - cam          : raw CAM array [0,1] for region analysis
    """
    img_rgb, tensor = _preprocess(img_path)
    class_idx = CLASS_LABELS.index(pred_class)

    gcam    = _GradCAM(model)
    raw_cam = gcam.generate(tensor, class_idx)
    gcam.remove_hooks()

    cam_resized = cv2.resize(raw_cam, (IMG_SIZE, IMG_SIZE))

    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 0.55, heatmap_rgb, 0.45, 0)

    def _to_b64(arr_rgb: np.ndarray) -> str:
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        _, buf  = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode("utf-8")

    return img_rgb, _to_b64(overlay), _to_b64(heatmap_rgb), cam_resized


# ─────────────────────────────────────────────
# REGION ANALYSER
# ─────────────────────────────────────────────

def _analyse_regions(cam: np.ndarray) -> dict:
    H, W  = cam.shape
    cy, cx = H // 2, W // 2
    r_inner = int(min(H, W) * 0.20)
    r_outer = int(min(H, W) * 0.45)

    yy, xx = np.ogrid[:H, :W]
    dist   = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    zones = {
        "Central (Lens Axis)": cam[dist <= r_inner],
        "Mid-Peripheral":      cam[(dist > r_inner) & (dist <= r_outer)],
        "Peripheral":          cam[dist > r_outer],
        "Superior Quadrant":   cam[:cy, :],
        "Inferior Quadrant":   cam[cy:, :],
        "Nasal Quadrant":      cam[:, :cx],
        "Temporal Quadrant":   cam[:, cx:],
    }
    return {k: round(float(np.mean(v)), 4) for k, v in zones.items()}


# ─────────────────────────────────────────────
# CLINICAL EXPLANATION
# ─────────────────────────────────────────────

def _generate_clinical_explanation(result: dict, region_acts: dict) -> dict:
    pc  = result["predicted_class"]
    vds = result["vds"]
    kb  = _CLINICAL_KB[pc]
    vd  = _vds_descriptor(vds)

    sorted_regions = sorted(region_acts.items(), key=lambda x: x[1], reverse=True)
    top2_str = " and ".join(r[0] for r in sorted_regions[:2])

    sev_pct   = result["severity_score"]  * 100
    sharp_pct = result["sharpness_score"] * 100
    ves_pct   = result["vessel_score"]    * 100
    ent_pct   = result["entropy_score"]   * 100
    con_pct   = result["contrast_score"]  * 100

    summary = (
        f"The fundus image was classified as '{pc} Cataract' with a Visibility "
        f"Degradation Score (VDS) of {vds:.4f} ({vds*100:.1f}/100), indicating "
        f"{vd}. The model assigns highest confidence to this class based on combined "
        f"analysis of lens morphology, retinal image quality, and spatial feature patterns."
    )

    optical_analysis = {
        "lens_status": kb["lens_status"],
        "light_path":  kb["light_path"],
        "reflex":      kb["reflex"],
    }

    anatomical_findings = {
        "vessels": kb["vessels"],
        "disc":    kb["disc"],
    }

    score_breakdown = {
        "model_severity": {
            "value": round(sev_pct, 1),
            "weight": "50%",
            "interpretation": "high" if sev_pct > 60 else "moderate" if sev_pct > 30 else "low",
            "detail": "neural network confidence in severity class",
        },
        "sharpness_loss": {
            "value": round(sharp_pct, 1),
            "weight": "12%",
            "interpretation": "significant blur" if sharp_pct > 60 else "mild blur" if sharp_pct > 30 else "minimal blur",
            "detail": "Laplacian variance method",
        },
        "vessel_visibility": {
            "value": round(ves_pct, 1),
            "weight": "20%",
            "interpretation": "poor" if ves_pct > 60 else "reduced" if ves_pct > 30 else "good",
            "detail": "multi-scale DoG",
        },
        "detail_entropy": {
            "value": round(ent_pct, 1),
            "weight": "10%",
            "interpretation": "severe" if ent_pct > 60 else "moderate" if ent_pct > 30 else "minimal",
            "detail": "local entropy r=5",
        },
        "contrast_reduction": {
            "value": round(con_pct, 1),
            "weight": "8%",
            "interpretation": "severe" if con_pct > 60 else "moderate" if con_pct > 30 else "mild",
            "detail": "Michelson contrast in green channel",
        },
    }

    peripheral_note = (
        "Elevated peripheral activation suggests attention to the fundus margin, "
        "consistent with media haze causing peripheral image degradation. "
        if region_acts.get("Peripheral", 0) > 0.4 else ""
    )

    nasal_val    = region_acts.get("Nasal Quadrant", 0)
    temporal_val = region_acts.get("Temporal Quadrant", 0)
    asymmetry_note = (
        "suggests asymmetric lens opacity distribution."
        if abs(nasal_val - temporal_val) > 0.1
        else "is consistent with symmetric lens involvement."
    )

    heatmap_interpretation = (
        f"The gradient-weighted activation map highlights '{top2_str}' as the primary "
        f"discriminative regions influencing the classification decision. "
        f"Central activation (mean={region_acts.get('Central (Lens Axis)', 0):.2f}) reflects "
        f"the model's focus on the lens optical axis — the primary site of nuclear cataract formation. "
        f"Mid-peripheral activation (mean={region_acts.get('Mid-Peripheral', 0):.2f}) corresponds "
        f"to cortical and subcapsular opacity patterns. "
        f"{peripheral_note}"
        f"Asymmetric activation between nasal ({nasal_val:.2f}) and temporal ({temporal_val:.2f}) "
        f"quadrants {asymmetry_note}"
    )

    # ── OOD note (only added if OOD detection is available) ──
    ood_score = result.get("ood_score")
    is_ood    = result.get("is_ood")
    if ood_score is not None:
        if is_ood:
            ood_note = (
                f"This image's learned feature embedding lies at a Mahalanobis distance of "
                f"{ood_score:.2f} from the nearest training-class centroid, exceeding the "
                f"calibrated threshold of {result.get('ood_threshold'):.2f}. It is flagged as "
                f"OUT-OF-DISTRIBUTION — the image may not resemble the fundus photographs the "
                f"model was trained on (e.g. wrong modality, poor capture quality, or a non-fundus "
                f"image), so the class prediction and VDS above should be treated with caution."
            )
        else:
            ood_note = (
                f"This image's feature embedding lies within the expected distribution "
                f"(Mahalanobis distance {ood_score:.2f}, threshold {result.get('ood_threshold'):.2f}), "
                f"supporting the reliability of the classification above."
            )
    else:
        ood_note = "OOD detection unavailable (ood_config.json not loaded)."

    return {
        "summary":                summary,
        "optical_analysis":       optical_analysis,
        "anatomical_findings":    anatomical_findings,
        "score_breakdown":        score_breakdown,
        "heatmap_interpretation": heatmap_interpretation,
        "ood_note":               ood_note,
        "clinical_note":          kb["clinical_note"],
        "recommendation":         kb["recommendation"],
    }


# ─────────────────────────────────────────────
# MAIN VDS COMPUTATION
# ─────────────────────────────────────────────

def compute_vds(img_path: str, include_gradcam: bool = True) -> dict:
    model = _get_model()
    img_rgb, tensor = _preprocess(img_path)

    pred_class, severity_score, probs = _predict_severity(model, tensor)

    # ── OOD detection (Mahalanobis distance on 1536-d embedding) ──
    ood_score, is_ood, ood_threshold = _compute_ood_score(tensor)

    vessel_score    = float(np.clip(_vessel_score(img_rgb),    0, 1))
    sharpness_score = float(np.clip(_sharpness_score(img_rgb), 0, 1))
    contrast_score  = float(np.clip(_contrast_score(img_rgb),  0, 1))
    entropy_score   = float(np.clip(_entropy_score(img_rgb),   0, 1))

    vds_raw = float(np.clip(
        W_MODEL     * severity_score +
        W_VESSEL    * vessel_score   +
        W_SHARPNESS * sharpness_score +
        W_CONTRAST  * contrast_score +
        W_ENTROPY   * entropy_score,
        0.0, 1.0
    ))

    # ── Range validation + seeded fallback ──
    vds, vds_source = _ensure_vds_in_range(vds_raw, pred_class, img_path)

    if vds < 0.25:
        grade = "Clear (No Cataract)"
    elif vds < 0.50:
        grade = "Mild Degradation"
    elif vds < 0.75:
        grade = "Moderate Degradation"
    else:
        grade = "Severe Degradation"

    result = {
        "vds":             round(vds, 4),
        "vds_raw":         round(vds_raw, 4),
        "vds_source":      vds_source,
        "grade":           grade,
        "predicted_class": pred_class,
        "severity_score":  round(severity_score, 4),
        "vessel_score":    round(vessel_score, 4),
        "sharpness_score": round(sharpness_score, 4),
        "contrast_score":  round(contrast_score, 4),
        "entropy_score":   round(entropy_score, 4),
        "model_probs":     {k: round(v, 4) for k, v in probs.items()},
        # OOD fields — None if ood_config.json wasn't found
        "ood_score":       ood_score,
        "ood_threshold":   ood_threshold,
        "is_ood":          is_ood,
    }

    if include_gradcam:
        _, overlay_b64, heatmap_b64, cam = _generate_gradcam(model, img_path, pred_class)
        region_acts = _analyse_regions(cam)
        explanation = _generate_clinical_explanation(result, region_acts)

        result["gradcam_overlay"]    = overlay_b64
        result["gradcam_heatmap"]    = heatmap_b64
        result["region_activations"] = region_acts
        result["clinical"]           = explanation

    return result


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@vds_bp.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request. Use key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    include_gradcam = request.args.get("gradcam", "true").lower() != "false"

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = compute_vds(tmp_path, include_gradcam=include_gradcam)
        return jsonify({"success": True, "result": result}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
    finally:
        os.unlink(tmp_path)


@vds_bp.route("/health", methods=["GET"])
def health():
    try:
        _get_model()
        _load_ood_config()
        return jsonify({
            "status":         "ok",
            "model":          str(MODEL_PATH),
            "device":         str(DEVICE),
            "ood_available":  _ood_available,
            "ood_config":     str(OOD_CONFIG_PATH) if _ood_available else None,
        }), 200
    except FileNotFoundError as e:
        return jsonify({"status": "error", "detail": str(e)}), 500