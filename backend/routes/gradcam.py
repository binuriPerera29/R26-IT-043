"""
Grad-CAM Blueprint
──────────────────
POST /api/gradcam/analyse   → run Grad-CAM on a fundus image, return heatmap + overlay

Ported directly from the notebook:
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

gradcam_bp = Blueprint("gradcam", __name__)

# =============================================================================
#  Config — exact values from notebook
# =============================================================================

GRADCAM_CONFIG = {
    "IMG_SIZE":    224,
    "CLASS_NAMES": ["advanced", "early", "normal"],
    "CLASS_COLORS": {
        "advanced": "#F44336",
        "early":    "#FF9800",
        "normal":   "#4CAF50",
    },
    "COLORMAP": cv2.COLORMAP_JET,
    "ALPHA":    0.50,
}

# =============================================================================
#  GradCAM engine — exact port of notebook class
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for timm EfficientNet-B0.

    Target layer: model.backbone.conv_head
      — the 1×1 conv that projects from MBConv blocks to 1280-d features,
        giving a 7×7 spatial map at IMG_SIZE=224.
    """

    def __init__(self, model, target_layer_name: str = "conv_head"):
        self.model = model
        self.model.eval()

        self._gradients   = None
        self._activations = None

        target_layer = getattr(model.backbone, target_layer_name, None)
        if target_layer is None:
            named = dict(model.backbone.named_modules())
            target_layer = named.get("bn2", list(named.values())[-3])

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            input_tensor : (1, 3, H, W) normalised, on device
            class_idx    : target class; None → argmax of predictions
        Returns:
            heatmap (H, W) float32 [0,1], class_idx (int), probs (np.ndarray)
        """
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:],
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, probs.squeeze().cpu().detach().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# =============================================================================
#  Helpers — exact ports of notebook helpers
# =============================================================================

_transform = T.Compose([
    T.Resize((GRADCAM_CONFIG["IMG_SIZE"], GRADCAM_CONFIG["IMG_SIZE"])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def _pil_to_tensor_and_display(pil_image, device):
    """Returns (normalised_tensor, uint8_RGB_numpy_for_display)."""
    img_size   = GRADCAM_CONFIG["IMG_SIZE"]
    display_np = np.array(pil_image.resize((img_size, img_size), Image.BILINEAR))
    tensor     = _transform(pil_image).unsqueeze(0).to(device)
    return tensor, display_np


def _apply_colormap(heatmap_01: np.ndarray) -> np.ndarray:
    """float [0,1] heatmap → RGB uint8 via OpenCV colormap (JET)."""
    heatmap_u8  = np.uint8(255 * heatmap_01)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, GRADCAM_CONFIG["COLORMAP"])
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def _blend_overlay(original_rgb: np.ndarray,
                   heatmap_rgb:  np.ndarray,
                   alpha: float = 0.50) -> np.ndarray:
    """Alpha-blend heatmap onto original image."""
    blended = (1 - alpha) * original_rgb.astype(np.float32) + \
              alpha       * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _ndarray_to_b64jpeg(arr: np.ndarray, max_size: int = 512) -> str:
    """Encode RGB uint8 numpy array as base64 JPEG data-URI."""
    pil = Image.fromarray(arr)
    pil.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _detect_high_activation_box(heatmap: np.ndarray, threshold: float = 0.75):
    """Return bounding box [x,y,w,h] of pixels above threshold, or None."""
    ys, xs = np.where(heatmap > threshold)
    if len(ys) == 0:
        return None
    return {
        "x":      int(xs.min()),
        "y":      int(ys.min()),
        "width":  int(xs.max() - xs.min()),
        "height": int(ys.max() - ys.min()),
    }


# =============================================================================
#  Route
# =============================================================================

@gradcam_bp.route("/analyse", methods=["POST"])
def analyse():
    model       = current_app.config.get("MODEL")
    device      = current_app.config["DEVICE"]
    class_names = current_app.config["CLASS_NAMES"]   # ["advanced","early","normal"]

    if model is None:
        return jsonify({"error": "Model not loaded. Check MODEL_PATH in backend."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided. POST with key 'image'."}), 400

    file = request.files["image"]
    try:
        pil_image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot read image."}), 400

    # Optional: force a specific target class (0=advanced, 1=early, 2=normal)
    force_class = request.form.get("class_idx")
    class_idx   = int(force_class) if force_class is not None else None

    try:
        tensor, display_np = _pil_to_tensor_and_display(pil_image, device)

        # Create a fresh GradCAM engine per request (thread-safe, no shared hooks)
        gradcam = GradCAM(model, target_layer_name="conv_head")
        heatmap, pred_idx, probs = gradcam(tensor, class_idx=class_idx)
        gradcam.remove_hooks()

        pred_class = class_names[pred_idx]
        heatmap_rgb = _apply_colormap(heatmap)
        overlay     = _blend_overlay(display_np, heatmap_rgb, GRADCAM_CONFIG["ALPHA"])

        # Draw high-activation bounding box on overlay copy
        overlay_boxed = overlay.copy()
        box = _detect_high_activation_box(heatmap, threshold=0.75)
        if box:
            cv2.rectangle(
                overlay_boxed,
                (box["x"], box["y"]),
                (box["x"] + box["width"], box["y"] + box["height"]),
                (255, 255, 255), 2
            )

        return jsonify({
            "prediction": {
                "class_index":   pred_idx,
                "class_name":    pred_class,
                "confidence":    round(float(probs[pred_idx]) * 100, 2),
                "probabilities": {
                    class_names[i]: round(float(probs[i]) * 100, 2)
                    for i in range(len(class_names))
                },
            },
            "gradcam": {
                "target_layer":     "backbone.conv_head",
                "activation_peak":  round(float(heatmap.max())  * 100, 2),
                "activation_mean":  round(float(heatmap.mean()) * 100, 2),
                "high_activation_box": box,
            },
            "images": {
                "original":    _ndarray_to_b64jpeg(display_np),
                "heatmap":     _ndarray_to_b64jpeg(heatmap_rgb),
                "overlay":     _ndarray_to_b64jpeg(overlay),
                "overlay_box": _ndarray_to_b64jpeg(overlay_boxed),
            },
            "class_colors": GRADCAM_CONFIG["CLASS_COLORS"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Grad-CAM analysis failed: {str(e)}"}), 500