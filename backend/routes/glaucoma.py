"""
Glaucoma Detection Blueprint  (with OOD detection)
────────────────────────────────────────────────────────────────
POST /api/glaucoma/predict   → classify fundus image (+ deterministic TTA + Grad-CAM + OOD)
GET  /api/glaucoma/classes   → return class names

CLASS ORDER (must match notebook): ["advanced", "early", "normal"]
  index 0 = advanced
  index 1 = early
  index 2 = normal

OOD DETECTION
─────────────
Ported from notebook cells OOD-A / OOD-B / OOD-C:
  • FeatureExtractor  → 1280-d embedding from model.backbone (timm EffNet-B0)
  • Per-class means + shared (tied) covariance, fitted on the TRAIN set
  • Mahalanobis distance = min over classes of (x-mean)^T * cov_inv * (x-mean)
  • threshold = 99th percentile of TRAIN distances (from ood_config.json)

An image is flagged as OOD ("ood") if EITHER:
  (a) mahalanobis_distance > ood_threshold   (genuinely unfamiliar image), OR
  (b) softmax confidence   < CONFIDENCE_THRESHOLD (0.70)  (model itself unsure)

Both cases return the SAME "ood" label / risk block, so the frontend only
needs to branch on `prediction.class_name == "ood"` (or `ood.is_ood`).

REQUIRED APP CONFIG (set these wherever you build the Flask app, alongside
MODEL / DEVICE / CLASS_NAMES / MODEL_PATH):
    app.config["OOD_CONFIG_PATH"] = "/path/to/ood_config.json"
If you don't set it, the blueprint will try to auto-locate
"ood_config.json" next to MODEL_PATH. If it still can't find it, OOD
detection via Mahalanobis distance is silently disabled (the low-confidence
rule still applies).

TTA (TEST-TIME AUGMENTATION) — DETERMINISTIC
─────────────────────────────────────────────
Previously, TTA used RandomCrop / RandomHorizontalFlip / ColorJitter, which
made repeated predictions on the SAME image return different confidence
scores (and occasionally different class_name near decision boundaries)
on every call, since nothing was seeded.

This version replaces that with classic deterministic 5-crop TTA:
  1. center crop
  2. top-left corner crop
  3. top-right corner crop
  4. bottom-left corner crop
  5. bottom-right corner crop
No randomness anywhere — the same input image always produces the exact
same crops, and therefore the exact same averaged probabilities, class_idx,
confidence, OOD result, and Grad-CAM target every time.
"""

import io
import os
import json
import base64
import traceback

import cv2
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify, current_app

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

glaucoma_bp = Blueprint("glaucoma", __name__)

IMG_SIZE = 224

# Below this softmax confidence, we no longer trust the class prediction and
# treat the image exactly like an OOD detection (same label + risk block).
CONFIDENCE_THRESHOLD = 0.70

_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
_NORMALIZE_STD  = [0.229, 0.224, 0.225]


# =============================================================================
#  Transforms — deterministic 5-crop TTA (no RandomCrop/Flip/Jitter)
# =============================================================================

def _base_transform():
    """Single deterministic pass — plain resize, no cropping."""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
    ])


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return TF.normalize(tensor, mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD)


def _deterministic_five_crop(image: Image.Image, crop_size: int = IMG_SIZE, upscale_pad: int = 32):
    """
    Resize to (crop_size + upscale_pad) on each side, then take the classic
    fixed 5 crops: center + 4 corners. Every crop position is fixed — no
    randomness — so the same input image always yields the same 5 tensors.

    Returns: list[torch.Tensor], each already normalized, shape (3, crop_size, crop_size)
    """
    resized = T.functional.resize(image, (crop_size + upscale_pad, crop_size + upscale_pad))
    # T.functional.five_crop returns (top-left, top-right, bottom-left, bottom-right, center)
    crops = T.functional.five_crop(resized, crop_size)

    tensors = []
    for crop in crops:
        t = T.functional.to_tensor(crop)
        t = _normalize_tensor(t)
        tensors.append(t)
    return tensors


# =============================================================================
#  Inference
# =============================================================================

def _predict_single(model, tensor, device):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def _predict_with_tta(model, image: Image.Image, device, tta_steps: int = 5):
    """
    Deterministic TTA: averages softmax probabilities over the plain resize
    plus up to 5 fixed crops (center + 4 corners). No randomness anywhere,
    so the same image always produces the exact same result.

    tta_steps controls how many of the fixed crops are used in addition to
    the base pass (capped at 5, the number of available deterministic crops).
    tta_steps <= 1 → base pass only (no cropping).
    """
    base_tf = _base_transform()
    all_probs = [_predict_single(model, base_tf(image), device)]

    if tta_steps > 1:
        n_crops = min(tta_steps - 1, 5)
        crop_tensors = _deterministic_five_crop(image)[:n_crops]
        for t in crop_tensors:
            all_probs.append(_predict_single(model, t, device))

    return np.mean(all_probs, axis=0)


# =============================================================================
#  OOD detection — Mahalanobis distance (ported from notebook cells OOD-A/B/C)
# =============================================================================

class FeatureExtractor(nn.Module):
    """
    Extracts the 1280-dim backbone embedding from GlaucomaEfficientNetB0.
    Stops before the classifier head — identical to the notebook's
    FeatureExtractor, used to fit / score the Mahalanobis distance.
    """
    def __init__(self, glaucoma_model):
        super().__init__()
        self.backbone = glaucoma_model.backbone   # timm EffNet-B0, pool included

    def forward(self, x):
        return self.backbone(x)   # (B, 1280) — already pooled by timm


# Module-level cache so we only read + parse ood_config.json once per process.
_OOD_STATE = {
    "loaded":      False,
    "enabled":     False,
    "threshold":   None,
    "class_means": None,   # {int cls_idx: np.array(1280,)}
    "cov_inv":     None,   # np.array(1280, 1280)
}


def _resolve_ood_config_path():
    path = current_app.config.get("OOD_CONFIG_PATH")
    if path and os.path.exists(path):
        return path

    # Fallback: look for glaucoma_ood_config.json next to MODEL_PATH, same
    # folder the notebook exports it into alongside the .pth checkpoint.
    # NOTE: app.py now hardcodes OOD_CONFIG_PATH directly, so this fallback
    # should rarely trigger — kept here only as a safety net.
    model_path = current_app.config.get("MODEL_PATH")
    if model_path:
        candidate = os.path.join(os.path.dirname(model_path), "glaucoma_ood_config.json")
        if os.path.exists(candidate):
            return candidate

    return None


def _load_ood_config():
    """Lazily loads + caches the Mahalanobis OOD config (class means +
    shared covariance inverse + threshold) exported by the notebook."""
    if _OOD_STATE["loaded"]:
        return _OOD_STATE

    _OOD_STATE["loaded"] = True  # mark attempted regardless of outcome

    path = _resolve_ood_config_path()
    if not path:
        print("[OOD] ood_config.json not found — Mahalanobis OOD check disabled "
              "(low-confidence rule still applies). Set app.config['OOD_CONFIG_PATH'].")
        return _OOD_STATE

    try:
        with open(path, "r") as f:
            cfg = json.load(f)

        _OOD_STATE["threshold"]   = float(cfg["threshold"])
        _OOD_STATE["class_means"] = {
            int(k): np.array(v, dtype=np.float32) for k, v in cfg["class_means"].items()
        }
        _OOD_STATE["cov_inv"]  = np.array(cfg["cov_inv"], dtype=np.float32)
        _OOD_STATE["enabled"]  = True
        print(f"[OOD] Loaded config from {path} (threshold={_OOD_STATE['threshold']:.4f})")
    except Exception:
        traceback.print_exc()
        print(f"[OOD] Failed to load/parse {path} — Mahalanobis OOD check disabled.")

    return _OOD_STATE


def _get_feature_extractor(model, device):
    """Builds (and caches on the app config) the FeatureExtractor wrapping
    the already-loaded classification model's backbone."""
    fx = current_app.config.get("_OOD_FEATURE_EXTRACTOR")
    if fx is None:
        fx = FeatureExtractor(model).to(device)
        fx.eval()
        current_app.config["_OOD_FEATURE_EXTRACTOR"] = fx
    return fx


def _compute_ood_score(feature_extractor, tensor, device, ood_state):
    """
    tensor: (1, 3, H, W), NOT yet on device.
    Returns (min_mahalanobis_distance, is_ood_bool).
    """
    with torch.no_grad():
        emb = feature_extractor(tensor.to(device)).cpu().numpy()[0]  # (1280,)

    x = emb[np.newaxis, :]
    distances = []
    for cls_idx, mean in ood_state["class_means"].items():
        diff = x - mean
        d = np.einsum('bi,ij,bj->b', diff, ood_state["cov_inv"], diff)[0]
        distances.append(d)

    min_dist = float(np.min(distances))
    is_ood   = min_dist > ood_state["threshold"]
    return min_dist, is_ood


# =============================================================================
#  Grad-CAM engine — ported from notebook (GradCAM class)
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for timm EfficientNet-B0.
    Target layer: model.backbone.conv_head (7×7 spatial map at IMG_SIZE=224).
    """

    def __init__(self, model, target_layer_name: str = "conv_head"):
        self.model = model
        self.model.eval()
        self._gradients   = None
        self._activations = None

        target_layer = getattr(model.backbone, target_layer_name, None)
        if target_layer is None:
            named        = dict(model.backbone.named_modules())
            target_layer = named.get("bn2", list(named.values())[-3])

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:],
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max - cam_min > 1e-8 else np.zeros_like(cam)

        return cam, class_idx, probs.squeeze().cpu().detach().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _run_gradcam(model, pil_image: Image.Image, device, pred_idx: int):
    """
    Run Grad-CAM for the predicted class.
    Returns a dict with base64 images and stats, or None on failure.
    """
    try:
        transform = _base_transform()
        display_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        tensor     = transform(pil_image).unsqueeze(0).to(device)

        gradcam = GradCAM(model, target_layer_name="conv_head")
        heatmap, _, _ = gradcam(tensor, class_idx=pred_idx)
        gradcam.remove_hooks()

        # Colourmap + overlay (exact notebook helpers)
        heatmap_u8   = np.uint8(255 * heatmap)
        heatmap_bgr  = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        alpha    = 0.50
        overlay  = np.clip(
            (1 - alpha) * display_np.astype(np.float32) +
            alpha       * heatmap_rgb.astype(np.float32),
            0, 255
        ).astype(np.uint8)

        # Bounding box around high-activation region (threshold 0.75)
        overlay_boxed = overlay.copy()
        ys, xs        = np.where(heatmap > 0.75)
        box           = None
        if len(ys) > 0:
            box = {
                "x":      int(xs.min()),
                "y":      int(ys.min()),
                "width":  int(xs.max() - xs.min()),
                "height": int(ys.max() - ys.min()),
            }
            cv2.rectangle(
                overlay_boxed,
                (box["x"], box["y"]),
                (box["x"] + box["width"], box["y"] + box["height"]),
                (255, 255, 255), 2
            )

        def _to_b64(arr):
            pil = Image.fromarray(arr)
            pil.thumbnail((512, 512))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        return {
            "target_layer":        "backbone.conv_head",
            "activation_peak":     round(float(heatmap.max())  * 100, 2),
            "activation_mean":     round(float(heatmap.mean()) * 100, 2),
            "high_activation_box": box,
            "images": {
                "heatmap":     _to_b64(heatmap_rgb),
                "overlay":     _to_b64(overlay),
                "overlay_box": _to_b64(overlay_boxed),
            },
        }

    except Exception:
        traceback.print_exc()
        return None


# =============================================================================
#  Risk mapping
# =============================================================================

RISK_MAP = {
    "advanced": {
        "level":          "High",
        "color":          "#ef4444",
        "recommendation": (
            "Advanced glaucoma signs detected. Seek immediate consultation with an "
            "ophthalmologist. Early treatment is critical to prevent permanent vision loss."
        ),
    },
    "early": {
        "level":          "Moderate",
        "color":          "#f59e0b",
        "recommendation": (
            "Early glaucoma signs detected. Schedule a comprehensive eye exam with an "
            "ophthalmologist within 1–3 months. Monitor intraocular pressure regularly."
        ),
    },
    "normal": {
        "level":          "Low",
        "color":          "#22c55e",
        "recommendation": (
            "No signs of glaucoma detected. Maintain regular annual eye check-ups "
            "and protect your eyes from UV exposure."
        ),
    },
    # Used both for genuine out-of-distribution images (flagged by the
    # Mahalanobis check) AND for low-confidence predictions (<70%).
    "ood": {
        "level":          "Uncertain",
        "color":          "#6b7280",
        "recommendation": (
            "This image could not be reliably classified as advanced, early, or "
            "normal glaucoma. This can happen with poor image quality, an unusual "
            "or non-standard fundus photo, or an image that isn't a fundus "
            "photograph at all. Please re-capture a clear, well-centered fundus "
            "image, or have it reviewed manually by an ophthalmologist."
        ),
    },
}


# =============================================================================
#  Routes
# =============================================================================

@glaucoma_bp.route("/predict", methods=["POST"])
def predict():
    model       = current_app.config.get("MODEL")
    device      = current_app.config["DEVICE"]
    class_names = current_app.config["CLASS_NAMES"]   # ["advanced", "early", "normal"]

    if model is None:
        return jsonify({"error": "Model not loaded. Check MODEL_PATH in backend."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided. POST with key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot read image. Use PNG or JPEG."}), 400

    tta_steps       = int(request.form.get("tta_steps", 5))
    include_gradcam = request.form.get("gradcam", "true").lower() != "false"

    try:
        # ── Step 1: Deterministic TTA classification ──────────────────────────
        probs      = _predict_with_tta(model, image, device, tta_steps)
        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        class_name = class_names[class_idx]

        # ── Step 2: OOD check — Mahalanobis distance on clean (non-TTA) embedding ─
        ood_state = _load_ood_config()

        mahal_distance      = None
        ood_threshold        = ood_state.get("threshold")
        is_ood_mahalanobis  = False

        if ood_state.get("enabled"):
            feature_extractor = _get_feature_extractor(model, device)
            clean_tensor      = _base_transform()(image).unsqueeze(0)
            mahal_distance, is_ood_mahalanobis = _compute_ood_score(
                feature_extractor, clean_tensor, device, ood_state
            )

        # Rule requested: confidence below 70% is treated exactly like OOD.
        is_low_confidence = confidence < CONFIDENCE_THRESHOLD
        is_ood            = is_ood_mahalanobis or is_low_confidence

        if is_ood_mahalanobis and is_low_confidence:
            ood_reason = "mahalanobis_and_low_confidence"
        elif is_ood_mahalanobis:
            ood_reason = "mahalanobis_distance"
        elif is_low_confidence:
            ood_reason = "low_confidence"
        else:
            ood_reason = None

        if is_ood:
            display_class_name = "ood"
            display_class_idx  = -1
            risk = RISK_MAP["ood"].copy()
        else:
            display_class_name = class_name
            display_class_idx  = class_idx
            risk = RISK_MAP.get(class_name, RISK_MAP["normal"]).copy()

        # ── Step 3: Grad-CAM (still shown for transparency, even if OOD) ──────
        gradcam_result = None
        if include_gradcam:
            gradcam_result = _run_gradcam(model, image, device, pred_idx=class_idx)

        # ── Step 4: Thumbnail for frontend preview ────────────────────────────
        thumb = image.copy()
        thumb.thumbnail((256, 256))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "prediction": {
                "class_index":    display_class_idx,
                "class_name":     display_class_name,   # "ood" if flagged, else real class
                "raw_class_name": class_name,            # what the classifier head predicted, regardless of OOD
                "confidence":     round(confidence * 100, 2),
                "probabilities": {
                    class_names[i]: round(float(probs[i]) * 100, 2)
                    for i in range(len(class_names))
                },
            },
            "ood": {
                "is_ood":                is_ood,
                "reason":                ood_reason,
                "mahalanobis_distance":  round(mahal_distance, 4) if mahal_distance is not None else None,
                "mahalanobis_threshold": round(ood_threshold, 4) if ood_threshold is not None else None,
                "confidence_threshold":  CONFIDENCE_THRESHOLD * 100,
                "mahalanobis_check_enabled": ood_state.get("enabled", False),
            },
            "risk":      risk,
            "thumbnail": f"data:image/jpeg;base64,{thumb_b64}",
            "tta_steps": tta_steps,
            "gradcam":   gradcam_result,   # None if include_gradcam=false or error
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@glaucoma_bp.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": current_app.config["CLASS_NAMES"]})