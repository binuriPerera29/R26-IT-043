"""
Glaucoma Detection Blueprint
────────────────────────────
POST /api/glaucoma/predict   → classify fundus image (+ TTA + Grad-CAM)
GET  /api/glaucoma/classes   → return class names

CLASS ORDER (must match notebook): ["advanced", "early", "normal"]
  index 0 = advanced
  index 1 = early
  index 2 = normal
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

glaucoma_bp = Blueprint("glaucoma", __name__)

IMG_SIZE = 224

# =============================================================================
#  Transforms — identical to the notebook's _transform
# =============================================================================

def _base_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

def _tta_transform():
    return T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


# =============================================================================
#  Inference
# =============================================================================

def _predict_single(model, tensor, device):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def _predict_with_tta(model, image: Image.Image, device, tta_steps: int = 5):
    base_tf = _base_transform()
    tta_tf  = _tta_transform()

    all_probs = [_predict_single(model, base_tf(image), device)]
    for _ in range(tta_steps - 1):
        all_probs.append(_predict_single(model, tta_tf(image), device))

    return np.mean(all_probs, axis=0)


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
        # ── Step 1: TTA classification ────────────────────────────────────────
        probs      = _predict_with_tta(model, image, device, tta_steps)
        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        class_name = class_names[class_idx]
        risk       = RISK_MAP.get(class_name, RISK_MAP["normal"]).copy()

        if confidence < 0.60:
            risk["recommendation"] = (
                f"[Low confidence — {confidence*100:.1f}%] "
                + risk["recommendation"]
                + " Consider re-imaging with better quality."
            )

        # ── Step 2: Grad-CAM (same predicted class as target) ─────────────────
        gradcam_result = None
        if include_gradcam:
            gradcam_result = _run_gradcam(model, image, device, pred_idx=class_idx)

        # ── Step 3: Thumbnail for frontend preview ────────────────────────────
        thumb = image.copy()
        thumb.thumbnail((256, 256))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "prediction": {
                "class_index":   class_idx,
                "class_name":    class_name,
                "confidence":    round(confidence * 100, 2),
                "probabilities": {
                    class_names[i]: round(float(probs[i]) * 100, 2)
                    for i in range(len(class_names))
                },
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