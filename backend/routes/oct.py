"""
oct.py — OCT Disease Classification Route  (PyTorch + OOD Detection)

Matches the .pth inference path from the notebook:
  Cell 9-OOD-A : FeatureExtractor (EfficientNet backbone -> 1280-d embedding)
  Cell 9-OOD-B : Mahalanobis distance — per-class means + shared (tied)
                 covariance, threshold = MAX of validation-set distances
  Cell 9-OOD-C : Exports consumed here — labels.json / ood_config.json /
                 best_efficientnet_b0_oct.pth
  Cell 9       : Single-image prediction with OOD gating
  Cell 10      : Grad-CAM (true backward-hook version) + OOD

Runs on torch / torchvision directly against the .pth checkpoint.
No ONNX involved anywhere in this file.

Additionally (not in notebook): predictions with confidence below
CONFIDENCE_THRESHOLD are flagged and treated with the same caution as
OOD detections — separate status field, merged into the same warning.

Required packages: torch, torchvision, numpy, opencv-python, pillow, flask
"""

from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
import base64
import os
import json
import threading
from io import BytesIO

oct_bp = Blueprint("oct", __name__)

# ── CONFIG ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]   # ImageNet stats — EfficientNet pretrained
STD      = [0.229, 0.224, 0.225]
DROPOUT  = 0.4                     # must match build_efficientnet_b0 in the notebook

CONFIDENCE_THRESHOLD = 70.0        # % — below this, flag like an OOD case

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(env_var: str, default_filename: str) -> str:
    path = os.environ.get(env_var, default_filename)
    if not os.path.isabs(path):
        path = os.path.join(_backend_dir, path)
    return path


MODEL_PTH_PATH       = _resolve("MODEL_PATH_OCT_PTH", "models/best_efficientnet_b0_oct.pth")
LABELS_JSON_PATH     = _resolve("LABELS_JSON_PATH_OCT", "models/oct_labels.json")
OOD_CONFIG_JSON_PATH = _resolve("OOD_CONFIG_JSON_PATH_OCT", "models/oct_ood_config.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Medical explanation database (unchanged — not in notebook) ─────────────
MEDICAL_EXPLANATIONS = {
    "CNV": {
        "full_name": "Choroidal Neovascularization",
        "severity": "Severe",
        "severity_color": "#EF4444",
        "description": (
            "Choroidal Neovascularization (CNV) is the growth of new, abnormal blood vessels "
            "from the choroid layer beneath the retina. These fragile vessels can leak fluid and "
            "blood into the retinal layers, causing rapid and potentially irreversible vision loss "
            "if left untreated."
        ),
        "oct_findings": [
            "Subretinal or intraretinal fluid accumulation",
            "Hyperreflective membrane beneath the retinal pigment epithelium (RPE)",
            "Irregular RPE elevation (pigment epithelial detachment)",
            "Increased retinal thickness in the affected region",
            "Possible subretinal hemorrhage signal",
        ],
        "gradcam_interpretation": (
            "The Grad-CAM heatmap highlights the region of RPE disruption and subretinal membrane. "
            "Red/yellow areas indicate where the model detected abnormal vascular tissue or fluid "
            "accumulation. Focus is typically in the central macular region where CNV is most active."
        ),
        "causes": [
            "Age-related macular degeneration (AMD)",
            "High myopia",
            "Ocular histoplasmosis",
            "Trauma",
        ],
        "treatment": [
            "Anti-VEGF injections (Ranibizumab, Bevacizumab, Aflibercept)",
            "Photodynamic therapy (PDT)",
            "Laser photocoagulation (for extra-foveal CNV)",
        ],
        "urgency": "Urgent — Please consult a retinal specialist within 1–2 weeks.",
        "prognosis": "Early treatment with anti-VEGF therapy can stabilize or improve vision in most patients.",
    },
    "DME": {
        "full_name": "Diabetic Macular Edema",
        "severity": "Moderate–Severe",
        "severity_color": "#F97316",
        "description": (
            "Diabetic Macular Edema (DME) is a complication of diabetic retinopathy where fluid "
            "leaks from damaged retinal blood vessels into the macula — the central area responsible "
            "for sharp, detailed vision. DME is the leading cause of vision loss in working-age adults "
            "with diabetes."
        ),
        "oct_findings": [
            "Intraretinal cystoid spaces (cystic edema)",
            "Diffuse retinal thickening around the fovea",
            "Hard exudate deposits appearing as hyperreflective foci",
            "Subretinal fluid in some cases",
            "Disruption of the inner segment/outer segment (IS/OS) junction",
        ],
        "gradcam_interpretation": (
            "The Grad-CAM heatmap highlights areas of cystoid edema and retinal thickening. "
            "High-activation zones (red/orange) correspond to regions with intraretinal fluid "
            "pockets and disrupted retinal architecture. The foveal and perifoveal regions "
            "typically show the highest activation in DME."
        ),
        "causes": [
            "Uncontrolled diabetes mellitus (Type 1 or Type 2)",
            "Hypertension",
            "Dyslipidemia",
            "Prolonged hyperglycemia",
        ],
        "treatment": [
            "Anti-VEGF therapy (first-line)",
            "Intravitreal corticosteroids (Triamcinolone, Dexamethasone implant)",
            "Focal/grid laser photocoagulation",
            "Tight glycemic and blood pressure control",
        ],
        "urgency": "Moderate–Urgent — Ophthalmology referral recommended within 2–4 weeks.",
        "prognosis": "With prompt treatment and diabetes management, vision can be stabilized. Long-term outcomes depend on glycemic control.",
    },
    "DRUSEN": {
        "full_name": "Drusen (Age-Related Macular Degeneration — Dry AMD)",
        "severity": "Mild–Moderate",
        "severity_color": "#EAB308",
        "description": (
            "Drusen are yellow deposits of lipids, proteins, and cellular debris that accumulate "
            "beneath the retinal pigment epithelium (RPE). They are the hallmark of dry age-related "
            "macular degeneration (AMD) and represent early to intermediate disease. While drusen "
            "alone may not cause immediate vision loss, they signal elevated risk of progression "
            "to advanced AMD (geographic atrophy or wet AMD)."
        ),
        "oct_findings": [
            "Small dome-shaped RPE elevations (drusen deposits)",
            "Irregular RPE reflectivity and contour",
            "Variable size — small (<63µm), medium (63–124µm), or large (≥125µm)",
            "No subretinal fluid (distinguishes from wet AMD)",
            "Possible RPE atrophy in areas of drusen regression",
        ],
        "gradcam_interpretation": (
            "The Grad-CAM heatmap reveals the RPE layer irregularities associated with drusen deposits. "
            "The model focuses on areas of RPE undulation and elevation. Multiple small activation "
            "hotspots are typical as drusen are often distributed across the macula rather than "
            "concentrated in one spot."
        ),
        "causes": [
            "Aging (primary risk factor)",
            "Genetic predisposition (CFH, ARMS2 gene variants)",
            "Smoking",
            "Cardiovascular disease",
            "High dietary fat intake",
        ],
        "treatment": [
            "AREDS2 supplements (Vitamin C, E, Zinc, Lutein, Zeaxanthin) for intermediate AMD",
            "Lifestyle modifications (smoking cessation, UV protection)",
            "Regular monitoring (OCT every 6–12 months)",
            "No direct treatment for drusen — monitor for progression",
        ],
        "urgency": "Non-urgent — Schedule ophthalmology follow-up within 1–3 months.",
        "prognosis": "Most patients with drusen have good visual prognosis. Risk of progression to advanced AMD increases with larger and more numerous drusen.",
    },
    "NORMAL": {
        "full_name": "Normal Retina",
        "severity": "None",
        "severity_color": "#22C55E",
        "description": (
            "The OCT scan shows a normal retinal architecture with no signs of pathological changes. "
            "All retinal layers appear intact, with a smooth and regular surface profile. "
            "The foveal depression is preserved, and there are no signs of fluid, drusen, or "
            "structural abnormalities."
        ),
        "oct_findings": [
            "Smooth, regular retinal surface contour",
            "Well-defined foveal pit (foveal depression present)",
            "Distinct retinal layers: NFL, GCL, IPL, INL, OPL, ONL, IS/OS, RPE",
            "No intraretinal or subretinal fluid",
            "Uniform RPE reflectivity without deposits",
        ],
        "gradcam_interpretation": (
            "In a normal scan, Grad-CAM activations are typically diffuse and low-intensity across "
            "the retina, as there are no pathological focal features for the model to concentrate on. "
            "Mild activation may appear around the fovea — a normal anatomical landmark the model "
            "uses to orient the retinal layer structure."
        ),
        "causes": [],
        "treatment": [
            "Routine eye examination every 1–2 years",
            "Maintain healthy diet rich in antioxidants and omega-3 fatty acids",
            "Protect eyes from UV exposure",
            "Monitor for any new visual symptoms",
        ],
        "urgency": "Routine — No urgent action required. Continue regular eye check-ups.",
        "prognosis": "Excellent. Normal retinal findings indicate healthy macular function.",
    },
}


# ── Model definition (Cell 4 / Cell 9 build_efficientnet_b0) ───────────────
def build_efficientnet_b0(num_classes: int, dropout: float = DROPOUT) -> nn.Module:
    """Same head shape used at train time — weights=None since we load our own state_dict."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Cell 9-OOD-A: FeatureExtractor (1280-d embedding, pre-classifier) ─────
class FeatureExtractor(nn.Module):
    def __init__(self, efficientnet):
        super().__init__()
        self.features = efficientnet.features
        self.avgpool  = efficientnet.avgpool   # AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)             # (B, 1280)


# ── Cell 10: true backward-hook Grad-CAM ────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output   = self.model(input_tensor)
        probs    = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        if class_idx is None:
            class_idx = pred_idx
        output[0, class_idx].backward()

        weights = self.gradients[0].mean(dim=(1, 2))          # (C,)
        cam = torch.zeros(
            self.activations.shape[2:], dtype=torch.float32, device=self.activations.device
        )
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, pred_idx, probs.detach().cpu().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Singleton runtime state ─────────────────────────────────────────────────
_lock              = threading.Lock()
_model             = None   # loaded EfficientNet-B0, weights from .pth
_feature_extractor = None   # wraps _model.features + _model.avgpool
_class_labels      = None
_ood_threshold      = None
_ood_class_means    = None  # dict[int] -> (1280,) np.float32
_ood_cov_inv        = None  # (1280, 1280) np.float32


def get_runtime():
    """
    Lazily loads (once):
      - the EfficientNet-B0 model, weights from the .pth checkpoint
      - a FeatureExtractor sharing those weights (for OOD embeddings)
      - class labels (labels.json)
      - Mahalanobis OOD config (ood_config.json)
    """
    global _model, _feature_extractor, _class_labels
    global _ood_threshold, _ood_class_means, _ood_cov_inv

    if _model is not None:
        return

    with _lock:
        if _model is not None:
            return

        for label, path in (
            ("model.pth", MODEL_PTH_PATH),
            ("labels.json", LABELS_JSON_PATH),
            ("ood_config.json", OOD_CONFIG_JSON_PATH),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    f"Place your {label} in backend/ and set the matching *_PATH env var."
                )

        # ── labels.json  {"0": "CNV", "1": "DME", ...} ──
        with open(LABELS_JSON_PATH, "r") as f:
            labels_dict = json.load(f)
        _class_labels = [labels_dict[str(i)] for i in range(len(labels_dict))]

        # ── ood_config.json (Cell 9-OOD-B / 9-OOD-C) ──
        with open(OOD_CONFIG_JSON_PATH, "r") as f:
            ood_cfg = json.load(f)
        _ood_threshold = float(ood_cfg["threshold"])
        _ood_class_means = {int(k): np.array(v, dtype=np.float32) for k, v in ood_cfg["class_means"].items()}
        _ood_cov_inv = np.array(ood_cfg["cov_inv"], dtype=np.float32)

        # ── model + feature extractor ──
        model = build_efficientnet_b0(len(_class_labels))
        model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model

        _feature_extractor = FeatureExtractor(model).to(DEVICE)
        _feature_extractor.eval()

        print(f"[INFO] PyTorch runtime ready — device={DEVICE}  classes={_class_labels}  "
              f"ood_threshold={_ood_threshold:.4f}  confidence_threshold={CONFIDENCE_THRESHOLD}")


# ── PREPROCESSING (matches notebook val_transform exactly) ─────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def preprocess(pil_image: Image.Image):
    """
    Returns:
        tensor      : (1, 3, 224, 224) float32, on DEVICE — model input
        display_np  : (224, 224, 3) uint8 RGB — resized original, for display/overlay
    """
    tensor = _transform(pil_image).unsqueeze(0).to(DEVICE)
    display_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    return tensor, display_np


# ── Cell 9-OOD-B: Mahalanobis OOD scorer ────────────────────────────────────
def compute_ood_score(embedding: np.ndarray):
    """
    embedding : (1280,) numpy array
    Returns (min_mahal_distance: float, is_ood: bool)
    """
    x = embedding[np.newaxis, :]
    distances = []
    for cls_idx in range(len(_class_labels)):
        diff = x - _ood_class_means[cls_idx]
        d = np.einsum("bi,ij,bj->b", diff, _ood_cov_inv, diff)[0]
        distances.append(d)
    min_dist = float(np.min(distances))
    return min_dist, min_dist > _ood_threshold


# ── Cell 10: overlay heatmap (unchanged math) ───────────────────────────────
def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray,
                     alpha: float = 0.45, colormap=cv2.COLORMAP_JET):
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    if heatmap_rgb.shape[:2] != image_np.shape[:2]:
        heatmap_rgb = cv2.resize(
            heatmap_rgb,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    superimposed = cv2.addWeighted(image_np, 1 - alpha, heatmap_rgb, alpha, 0)
    return superimposed, heatmap_rgb


def np_to_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── PREDICT ENDPOINT ─────────────────────────────────────────────────────
@oct_bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        get_runtime()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    try:
        pil_image = Image.open(file.stream).convert("RGB")
        tensor, display_np = preprocess(pil_image)

        # Everything below touches shared state on _model (hooks + backward),
        # so the whole inference step — OOD embedding included — is serialized
        # to avoid two requests' Grad-CAM hooks clobbering each other.
        with _lock:
            with torch.no_grad():
                embedding = _feature_extractor(tensor).cpu().numpy()[0]  # (1280,)
            ood_score, is_ood = compute_ood_score(embedding)

            grad_cam = GradCAM(_model, target_layer=_model.features[-1])
            tensor_for_grad = tensor.clone().requires_grad_(True)
            cam, pred_idx, probs = grad_cam.generate(tensor_for_grad)
            grad_cam.remove_hooks()

        pred_label = _class_labels[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        is_low_confidence = confidence < CONFIDENCE_THRESHOLD

        class_probs = {
            _class_labels[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(_class_labels))
        }

        overlay, heatmap = overlay_heatmap(display_np, cam, alpha=0.45)

        medical_info = MEDICAL_EXPLANATIONS[pred_label]

        response = {
            "prediction": pred_label,
            "confidence": round(confidence, 2),
            "class_probabilities": class_probs,
            "ood_detection": {
                "score": round(ood_score, 4),
                "threshold": round(_ood_threshold, 4),
                "is_ood": bool(is_ood),
                "status": "Out-of-Distribution" if is_ood else "In-Distribution",
            },
            "confidence_check": {
                "confidence": round(confidence, 2),
                "threshold": CONFIDENCE_THRESHOLD,
                "is_low_confidence": bool(is_low_confidence),
                "status": "Low Confidence" if is_low_confidence else "Confident",
            },
            "images": {
                "original": np_to_b64(display_np),
                "heatmap": np_to_b64(heatmap),
                "overlay": np_to_b64(overlay),
            },
            "medical_explanation": {
                "full_name": medical_info["full_name"],
                "severity": medical_info["severity"],
                "severity_color": medical_info["severity_color"],
                "description": medical_info["description"],
                "oct_findings": medical_info["oct_findings"],
                "gradcam_interpretation": medical_info["gradcam_interpretation"],
                "causes": medical_info["causes"],
                "treatment": medical_info["treatment"],
                "urgency": medical_info["urgency"],
                "prognosis": medical_info["prognosis"],
            },
        }

        # ── Unified caution flag: OOD and low-confidence get the same treatment ──
        needs_caution = is_ood or is_low_confidence
        response["needs_caution"] = bool(needs_caution)

        if is_ood and is_low_confidence:
            response["warning"] = (
                "This prediction is both out-of-distribution and low-confidence "
                f"({confidence:.1f}% < {CONFIDENCE_THRESHOLD:.0f}%). Treat it with "
                "significant caution — it may not be a genuine OCT scan, may show "
                "pathology outside the four trained classes, or the model may simply "
                "be uncertain about this image."
            )
        elif is_ood:
            response["warning"] = (
                "This image's feature embedding falls outside the model's training "
                "distribution (Mahalanobis distance exceeds the calibrated threshold). "
                "Treat this prediction with caution — it may not be a genuine OCT scan "
                "or may show pathology outside the four trained classes."
            )
        elif is_low_confidence:
            response["warning"] = (
                f"Prediction confidence ({confidence:.1f}%) is below the "
                f"{CONFIDENCE_THRESHOLD:.0f}% threshold. Treat this prediction with "
                "caution — the model is not confident in this classification."
            )

        return jsonify(response)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500