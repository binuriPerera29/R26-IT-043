"""
oct.py — OCT Disease Classification Route

Notebook cells reproduced exactly:
  Cell 1  : CLAHE only at dataset-prep time — NOT at inference
  Cell 2  : CLASS_LABELS, IMG_SIZE=224, MEAN, STD, DEVICE
  Cell 4  : build_efficientnet_b0() — efficientnet_b0, classifier[1]=Linear(1280,4), dropout=0.4
  Cell 9  : predict() — Image.open.convert('RGB') → Resize(224) → ToTensor → Normalize → torch.no_grad
  Cell 10 : GradCAM(model, model.features[-1]), overlay_heatmap(alpha=0.45, COLORMAP_JET)
            preprocess_image: pil.resize(224) → transform → tensor.requires_grad_(True)
            display_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))

Notebook saves checkpoint as plain state_dict:
  torch.save(model.state_dict(), best_model_path)   ← Cell 6
So load_state_dict receives it directly — no wrapper key.
"""

from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import base64
import os
from io import BytesIO

oct_bp = Blueprint("oct", __name__)

# ── Cell 2: CONFIG ────────────────────────────────────────────────────────
CLASS_LABELS = ["CNV", "DME", "DRUSEN", "NORMAL"]   # matches train_dataset.class_to_idx order
IMG_SIZE     = 224
MEAN         = [0.485, 0.456, 0.406]                 # ImageNet stats — EfficientNet pretrained
STD          = [0.229, 0.224, 0.225]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH   = os.environ.get("MODEL_PATH_OCT", "best_efficientnet_b0_oct.pth")

# ── Medical explanation database (added for web — not in notebook) ─────────
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


# ── Cell 4 & Cell 9 & Cell 10: MODEL ARCHITECTURE ────────────────────────
# Notebook Cell 4 builds with pretrained weights, but saves only state_dict.
# Cell 9 & 10 reload with weights=None then load_state_dict — we do the same.
def build_efficientnet_b0(num_classes: int = 4, dropout: float = 0.4) -> nn.Module:
    """
    Exact match of notebook Cell 4 / Cell 9 / Cell 10:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features   # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features   # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Singleton loader ──────────────────────────────────────────────────────
_model = None

def get_model() -> nn.Module:
    global _model
    if _model is None:
        model_path = MODEL_PATH
        if not os.path.isabs(model_path):
            # resolve relative to backend/ directory
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path  = os.path.join(backend_dir, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Place your .pth file in backend/ and set MODEL_PATH in .env"
            )

        _model = build_efficientnet_b0(num_classes=len(CLASS_LABELS))

        # Notebook Cell 6 saves: torch.save(model.state_dict(), path)
        # So the checkpoint IS the state_dict directly — no wrapper key.
        # We also handle wrapped checkpoints (model_state_dict key) just in case.
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        _model.load_state_dict(state)
        _model.to(DEVICE)
        _model.eval()
        print(f"[INFO] EfficientNet-B0 loaded → {model_path}  (device: {DEVICE})")
    return _model


# ── Cell 9 & Cell 10: INFERENCE TRANSFORM ────────────────────────────────
# Notebook val_transform (Cell 3) = inference transform used in Cell 9 & Cell 10:
#   Resize(224,224) → ToTensor → Normalize(MEAN, STD)
# NO CLAHE at inference. CLAHE was dataset-prep only (Cell 1 save_with_clahe).
def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ── Cell 10: GRAD-CAM ────────────────────────────────────────────────────
class GradCAM:
    """
    Exact match of notebook Cell 10 GradCAM class.
    Target layer : model.features[-1]  (last MBConv block, ~7×7 spatial map)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Notebook Cell 10 generate() logic — line for line:
          1. zero_grad
          2. forward → softmax → pred_idx
          3. backward on target class score
          4. weights = gradients[0].mean(dim=(1,2))
          5. cam = zeros on activations.device   ← key device fix from notebook
          6. weighted sum of activation maps
          7. ReLU → interpolate → cpu → normalise [0,1]
        """
        self.model.zero_grad()

        output   = self.model(input_tensor)
        probs    = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

        if class_idx is None:
            class_idx = pred_idx

        # Backward on the target class score
        output[0, class_idx].backward()

        # α^c_k = Global Average Pool of gradients  (shape: C,)
        weights = self.gradients[0].mean(dim=(1, 2))

        # Weighted sum — initialised ON THE SAME DEVICE as activations
        # (this is the explicit cuda/cpu fix noted in notebook Cell 10)
        cam = torch.zeros(
            self.activations.shape[2:],
            dtype=torch.float32,
            device=self.activations.device,
        )
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        # ReLU: keep only positive influences
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Move to CPU after all GPU ops (notebook: cam = cam.squeeze().cpu().numpy())
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, pred_idx, probs.cpu().detach().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Cell 10: OVERLAY HEATMAP ─────────────────────────────────────────────
def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.45, colormap=cv2.COLORMAP_JET):
    """
    Exact match of notebook Cell 10 overlay_heatmap():
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(image_np, 1 - alpha, heatmap_rgb, alpha, 0)
    alpha=0.45, colormap=COLORMAP_JET — same defaults as notebook.
    """
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


# ── PREDICT ENDPOINT (Cell 9 + Cell 10 combined) ─────────────────────────
@oct_bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    try:
        # ── Cell 9 & Cell 10: Image.open(path).convert('RGB') ──
        pil_image = Image.open(file.stream).convert("RGB")

        # ── Cell 10: preprocess_image() ──
        #   display_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))
        #   tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        #   tensor.requires_grad_(True)
        transform  = get_val_transform()
        display_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))  # uint8 RGB for display
        tensor     = transform(pil_image).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)   # required for backward through the graph

        # ── Cell 10: GradCAM on model.features[-1] ──
        grad_cam             = GradCAM(model, target_layer=model.features[-1])
        cam, pred_idx, probs = grad_cam.generate(tensor)
        overlay, heatmap     = overlay_heatmap(display_np, cam, alpha=0.45)
        grad_cam.remove_hooks()

        # ── Cell 9: results ──
        pred_label  = CLASS_LABELS[pred_idx]
        confidence  = float(probs[pred_idx]) * 100

        # class probabilities dict  e.g. {"CNV": 97.3, "DME": 1.2, ...}
        class_probs = {
            CLASS_LABELS[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_LABELS))
        }

        medical_info = MEDICAL_EXPLANATIONS[pred_label]

        return jsonify({
            "prediction":          pred_label,
            "confidence":          round(confidence, 2),
            "class_probabilities": class_probs,
            "images": {
                "original": np_to_b64(display_np),   # resized original
                "heatmap":  np_to_b64(heatmap),       # JET colourmap heatmap
                "overlay":  np_to_b64(overlay),       # blended overlay (alpha=0.45)
            },
            "medical_explanation": {
                "full_name":              medical_info["full_name"],
                "severity":               medical_info["severity"],
                "severity_color":         medical_info["severity_color"],
                "description":            medical_info["description"],
                "oct_findings":           medical_info["oct_findings"],
                "gradcam_interpretation": medical_info["gradcam_interpretation"],
                "causes":                 medical_info["causes"],
                "treatment":              medical_info["treatment"],
                "urgency":                medical_info["urgency"],
                "prognosis":              medical_info["prognosis"],
            },
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500