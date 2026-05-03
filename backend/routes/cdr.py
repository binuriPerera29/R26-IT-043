"""
CDR (Cup-to-Disc Ratio) Blueprint
───────────────────────────────────
POST /api/cdr/analyse   → segment disc/cup, compute CDR, apply model correction

Directly ported from the notebook:
  R26-IT-043 | Chavindee M.A.P | IT22127778 | SLIIT
"""

import io
import base64
import traceback

import cv2
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify, current_app

import torch
import torch.nn.functional as F
import torchvision.transforms as T

cdr_bp = Blueprint("cdr", __name__)

# =============================================================================
#  CDR correction config — exact values from notebook
# =============================================================================

CDR_RANGES = {
    "normal":   (0.0,  0.39),
    "early":    (0.40, 0.59),
    "advanced": (0.60, 1.0),
}

CDR_MIDPOINTS = {
    "normal":   0.20,
    "early":    0.50,
    "advanced": 0.70,
}

ALPHA = 0.35   # blend strength

# =============================================================================
#  Image preprocessing (CLAHE) — exact from notebook
# =============================================================================

def preprocess_for_cdr(pil_image, img_size=512):
    """Convert PIL image to numpy, resize, apply CLAHE enhancement."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (img_size, img_size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return img, enhanced


# =============================================================================
#  Optic disc segmentation — exact from notebook
# =============================================================================

def segment_optic_disc(enhanced_img):
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY)
    _, bright_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels < 2:
        return np.zeros_like(gray), None
    largest   = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    disc_mask = (labels == largest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return disc_mask, None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) >= 5:
        ellipse    = cv2.fitEllipse(cnt)
        disc_clean = np.zeros_like(disc_mask)
        cv2.ellipse(disc_clean, ellipse, 255, -1)
        return disc_clean, ellipse
    return disc_mask, None


# =============================================================================
#  Optic cup segmentation — exact from notebook
# =============================================================================

def segment_optic_cup(enhanced_img, disc_mask):
    gray        = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY)
    disc_region = cv2.bitwise_and(gray, gray, mask=disc_mask)
    disc_pixels = gray[disc_mask > 0]
    if len(disc_pixels) == 0:
        return np.zeros_like(gray), None
    threshold = np.percentile(disc_pixels, 75)
    _, cup_binary = cv2.threshold(disc_region, int(threshold), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cup_binary = cv2.morphologyEx(cup_binary, cv2.MORPH_CLOSE, kernel)
    cup_binary = cv2.morphologyEx(cup_binary, cv2.MORPH_OPEN,  kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cup_binary, connectivity=8)
    if num_labels < 2:
        return np.zeros_like(gray), None
    largest  = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cup_mask = (labels == largest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cup_mask, None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) >= 5:
        ellipse   = cv2.fitEllipse(cnt)
        cup_clean = np.zeros_like(cup_mask)
        cv2.ellipse(cup_clean, ellipse, 255, -1)
        return cup_clean, ellipse
    return cup_mask, None


# =============================================================================
#  CDR calculation and labelling — exact from notebook
# =============================================================================

def calculate_cdr(disc_ellipse, cup_ellipse):
    if disc_ellipse is None or cup_ellipse is None:
        return None
    disc_vertical = disc_ellipse[1][1]
    cup_vertical  = cup_ellipse[1][1]
    if disc_vertical == 0:
        return None
    return round(float(cup_vertical / disc_vertical), 4)


def cdr_to_label(cdr):
    if cdr is None:
        return "Unknown", "#888888"
    if cdr < 0.4:
        return "Normal",            "#4CAF50"
    elif cdr < 0.6:
        return "Early Glaucoma",    "#FF9800"
    else:
        return "Advanced Glaucoma", "#F44336"


def adjust_cdr_by_label(raw_cdr, model_label, confidence):
    """
    Blend raw CDR toward the model-predicted label's expected range.
    Exact port of notebook's adjust_cdr_by_label().
    """
    low, high    = CDR_RANGES[model_label]
    midpoint     = CDR_MIDPOINTS[model_label]
    blend_alpha  = ALPHA * confidence

    if raw_cdr is None:
        adjusted = midpoint
    elif low <= raw_cdr <= high:
        adjusted = raw_cdr
    else:
        adjusted = raw_cdr * (1 - blend_alpha) + midpoint * blend_alpha
        adjusted = max(low, min(high, adjusted))

    adjusted = round(adjusted, 4)
    adj_label, adj_color = cdr_to_label(adjusted)
    return adjusted, adj_label, adj_color


# =============================================================================
#  Overlay helper — draw ellipses on image and return base64 JPEG
# =============================================================================

def _overlay_to_b64(img_rgb, disc_ellipse, cup_ellipse):
    overlay = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    if disc_ellipse is not None:
        cv2.ellipse(overlay, disc_ellipse, (0, 255, 0), 3)
    if cup_ellipse is not None:
        cv2.ellipse(overlay, cup_ellipse,  (0, 0, 255), 3)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(overlay_rgb)
    pil_img.thumbnail((512, 512))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# =============================================================================
#  Model inference helper (reuses loaded model from app config)
# =============================================================================

IMG_SIZE = 224

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

def _run_model(model, pil_image, device):
    img_t = _transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs


# =============================================================================
#  Route
# =============================================================================

@cdr_bp.route("/analyse", methods=["POST"])
def analyse():
    model       = current_app.config.get("MODEL")
    device      = current_app.config["DEVICE"]
    class_names = current_app.config["CLASS_NAMES"]   # ["advanced","early","normal"]

    if "image" not in request.files:
        return jsonify({"error": "No image provided. POST with key 'image'."}), 400

    file = request.files["image"]
    try:
        pil_image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot read image."}), 400

    try:
        # ── Step 1: Model prediction (guides CDR correction) ──────────────────
        model_result = None
        if model is not None:
            probs         = _run_model(model, pil_image, device)
            pred_idx      = int(np.argmax(probs))
            model_result  = {
                "class":      class_names[pred_idx],
                "confidence": float(probs[pred_idx]),
                "all_probs":  {class_names[i]: round(float(probs[i]) * 100, 2)
                               for i in range(len(class_names))},
            }

        # ── Step 2: CDR segmentation ──────────────────────────────────────────
        img_np, enhanced     = preprocess_for_cdr(pil_image)
        disc_mask, disc_ell  = segment_optic_disc(enhanced)
        cup_mask,  cup_ell   = segment_optic_cup(enhanced, disc_mask)
        raw_cdr              = calculate_cdr(disc_ell, cup_ell)
        raw_label, raw_color = cdr_to_label(raw_cdr)

        # ── Step 3: CDR correction guided by model ────────────────────────────
        if model_result is not None:
            adj_cdr, adj_label, adj_color = adjust_cdr_by_label(
                raw_cdr,
                model_result["class"],
                model_result["confidence"],
            )
        else:
            adj_cdr, adj_label, adj_color = raw_cdr, raw_label, raw_color

        # ── Step 4: Annotated overlay image ──────────────────────────────────
        overlay_b64 = _overlay_to_b64(img_np, disc_ell, cup_ell)

        return jsonify({
            "cdr": {
                "value":     adj_cdr,
                "raw_value": raw_cdr,
                "label":     adj_label,
                "color":     adj_color,
                "raw_label": raw_label,
                "adjusted":  adj_cdr != raw_cdr,
            },
            "model": model_result,
            "overlay_image": overlay_b64,
            "segmentation": {
                "disc_found": disc_ell is not None,
                "cup_found":  cup_ell  is not None,
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"CDR analysis failed: {str(e)}"}), 500