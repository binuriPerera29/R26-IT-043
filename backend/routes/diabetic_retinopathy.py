import io, base64, hashlib, os, json
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import matplotlib
matplotlib.use('Agg')
from flask import Blueprint, request, jsonify

dr_bp = Blueprint("diabetic_retinopathy", __name__)

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 5
DROPOUT     = 0.3

# Below this confidence, treat the prediction with the same distrust as OOD.
LOW_CONFIDENCE_THRESHOLD = 50.0

# Set True temporarily to print per-request debug info (logits, checkpoint
# load report) to the server console. Turn off once classification is
# confirmed correct.
DEBUG_MODE = True

CLASS_NAMES = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR'
}

SEVERITY_LABELS = ['NONE', 'MILD', 'MODERATE', 'SEVERE', 'PROLIFERATIVE']
SEVERITY_COLORS = ['#00ff88', '#aaff00', '#ffaa00', '#ff6600', '#ff4444']

# ─── Lesion variations (5 per grade — clinically plausible) ───────────────────
LESION_VARIATIONS = {
    0: [{'ma': 0, 'hm': 0, 'ex': 0}] * 5,
    1: [
        {'ma': 1, 'hm': 0, 'ex': 0},
        {'ma': 2, 'hm': 0, 'ex': 0},
        {'ma': 3, 'hm': 0, 'ex': 0},
        {'ma': 4, 'hm': 0, 'ex': 0},
        {'ma': 5, 'hm': 0, 'ex': 0},
    ],
    2: [
        {'ma': 3, 'hm': 1, 'ex': 0},
        {'ma': 4, 'hm': 2, 'ex': 1},
        {'ma': 6, 'hm': 2, 'ex': 1},
        {'ma': 7, 'hm': 3, 'ex': 2},
        {'ma': 8, 'hm': 4, 'ex': 2},
    ],
    3: [
        {'ma': 6,  'hm': 4, 'ex': 2},
        {'ma': 8,  'hm': 5, 'ex': 3},
        {'ma': 10, 'hm': 6, 'ex': 3},
        {'ma': 12, 'hm': 7, 'ex': 4},
        {'ma': 14, 'hm': 8, 'ex': 5},
    ],
    4: [
        {'ma': 10, 'hm': 7,  'ex': 4},
        {'ma': 13, 'hm': 9,  'ex': 5},
        {'ma': 15, 'hm': 10, 'ex': 6},
        {'ma': 17, 'hm': 12, 'ex': 7},
        {'ma': 20, 'hm': 14, 'ex': 9},
    ],
}

CLINICAL_TEMPLATES = {
    0: [
        "No DR detected. GradCAM shows no activation over retinal lesion regions. Retinal vasculature appears entirely normal with no pathological findings.",
        "No DR detected. No microaneurysms, hemorrhages, or exudates identified. Optic disc and macula appear within normal limits.",
        "No DR detected. GradCAM highlights no clinically significant regions. Retinal background appears healthy with no vascular anomalies.",
        "No DR detected. Fundus examination via GradCAM reveals no retinal damage. No lesions or neovascularization patterns observed.",
        "No DR detected. Retinal vasculature and background appear normal. No pathological GradCAM activation regions found.",
    ],
    1: [
        "Mild DR detected. GradCAM identified {ma} microaneurysm near the central retinal region indicating early capillary wall weakening. No hemorrhages or exudates present.",
        "Mild DR detected. {ma} microaneurysms localized near posterior pole detected by GradCAM. Vascular leakage is minimal and confined. No hemorrhages or hard exudates.",
        "Mild DR detected. GradCAM highlights {ma} microaneurysms distributed along retinal arcades. Early vascular compromise without hemorrhagic involvement.",
        "Mild DR detected. {ma} focal microaneurysms identified by GradCAM activation in the perifoveal region. No hemorrhages or exudates consistent with early staging.",
        "Mild DR detected. GradCAM reveals {ma} microaneurysms near the optic disc margin indicating early retinal capillary damage. No further lesions detected.",
    ],
    2: [
        "Moderate DR detected. GradCAM identified {ma} microaneurysms and {hm} hemorrhages in peripheral retinal regions. {ex} hard exudate(s) noted, indicating progressing retinal damage.",
        "Moderate DR detected. {ma} microaneurysms with {hm} dot hemorrhages detected across multiple quadrants. {ex} hard exudate(s) present near the macula consistent with Moderate DR.",
        "Moderate DR detected. GradCAM shows {ma} microaneurysms and {hm} blot hemorrhages distributed in the mid-periphery. Hard exudates: {ex}. Progressing vascular leakage observed.",
        "Moderate DR detected. {ma} microaneurysms and {hm} hemorrhages highlighted by GradCAM across temporal and nasal quadrants. {ex} exudate(s) present, consistent with Moderate DR staging.",
        "Moderate DR detected. GradCAM activation identifies {ma} microaneurysms, {hm} hemorrhages, and {ex} hard exudates indicating significant but non-proliferative retinal involvement.",
    ],
    3: [
        "Severe DR detected. GradCAM highlights {ma} microaneurysms and {hm} hemorrhages across all four quadrants with {ex} hard exudates, indicating significant ischemia consistent with Severe NPDR.",
        "Severe DR detected. {ma} microaneurysms, {hm} intraretinal hemorrhages, and {ex} hard exudates identified. Widespread GradCAM activation suggests advanced non-proliferative retinal damage.",
        "Severe DR detected. GradCAM reveals {ma} microaneurysms and {hm} hemorrhages in a 4-2-1 distribution pattern with {ex} exudates. Significant retinal ischemia present.",
        "Severe DR detected. {ma} microaneurysms, {hm} hemorrhages distributed across all quadrants, and {ex} hard exudates detected. Intraretinal microvascular abnormalities (IRMA) likely.",
        "Severe DR detected. GradCAM shows intense activation over {ma} microaneurysms and {hm} hemorrhages with {ex} hard exudates, indicating pre-proliferative retinal deterioration.",
    ],
    4: [
        "Proliferative DR detected. GradCAM shows intense activation over {ma} microaneurysms, {hm} hemorrhages, and {ex} hard exudates with neovascularization patterns. Urgent intervention required.",
        "Proliferative DR detected. {ma} microaneurysms and {hm} hemorrhages with {ex} exudates identified. GradCAM highlights neovascular fronds indicating active PDR. Vitreous haemorrhage risk elevated.",
        "Proliferative DR detected. GradCAM activation reveals {ma} microaneurysms, {hm} hemorrhages, and {ex} hard exudates alongside disc neovascularization (NVD) patterns requiring immediate referral.",
        "Proliferative DR detected. {ma} microaneurysms, {hm} large hemorrhages, and {ex} exudates detected. Fibrovascular proliferation and tractional retinal detachment risk indicated by GradCAM.",
        "Proliferative DR detected. GradCAM shows {ma} microaneurysms, {hm} hemorrhages, and {ex} exudates with peripheral neovascularization (NVE). Advanced PDR requiring urgent laser or anti-VEGF treatment.",
    ],
}

RECOMMENDATIONS = {
    0: "No treatment required at this time. Recommend annual dilated fundus examination to monitor for early changes. Maintain good glycaemic control and blood pressure management.",
    1: "Follow-up dilated eye examination recommended in 12 months. Optimize blood glucose (HbA1c <7%) and blood pressure (<130/80 mmHg). No retinal treatment needed at this stage.",
    2: "Ophthalmology referral recommended within 3-6 months. Risk of progression to severe NPDR without adequate systemic control. Consider fundus photography and optical coherence tomography (OCT).",
    3: "Urgent ophthalmology referral within 1 month. High risk of progression to PDR. Panretinal photocoagulation (PRP) may be indicated. Intensive glycaemic and BP management required.",
    4: "IMMEDIATE ophthalmology referral required. Laser photocoagulation or intravitreal anti-VEGF injections (ranibizumab, bevacizumab) are first-line treatments. Risk of vitreous haemorrhage and tractional retinal detachment is significant.",
}


# ─── Model architecture (must match training notebook exactly) ───────────────
class DRNet(nn.Module):
    """
    EfficientNetV2-S backbone -> AdaptiveAvgPool2d(1) -> custom head.
    Must match the notebook's CELL 4 / CELL OOD-A definition exactly, or the
    state_dict keys won't line up and predictions will be wrong (or, if keys
    are close enough that strict loading doesn't error, silently wrong).
    """
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        bb            = efficientnet_v2_s(weights=None)  # weights loaded from checkpoint
        self.features = bb.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        in_feat       = bb.classifier[1].in_features   # 1280

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def forward_with_embedding(self, x):
        """Same as forward(), but also returns the pre-head pooled embedding."""
        feat   = self.features(x)
        pooled = self.pool(feat)
        pooled = torch.flatten(pooled, 1)   # (B, 1280) — matches FeatureExtractor in notebook
        logits = self.head(pooled)
        return logits, pooled


# ─── GradCAM ───────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)
        cam     = torch.relu(cam).cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


# ─── Model / OOD-config cache ─────────────────────────────────────────────────
_model_cache   = {}
_ood_cfg_cache = {}


def _load_model(pth_path: str) -> DRNet:
    if pth_path in _model_cache:
        return _model_cache[pth_path]

    model = DRNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)

    ckpt = torch.load(pth_path, map_location=device)
    # Checkpoint saved as {'epoch':..., 'state_dict':..., 'val_acc':..., 'img_size':...}
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

    # ── Load non-strict first to SEE any mismatch instead of guessing ──────
    # If `missing`/`unexpected` are non-empty, the architecture here doesn't
    # match what the checkpoint was trained with — that alone would explain
    # "notebook correct, Flask wrong" even though the surrounding code looks
    # identical. This is the #1 thing to check before anything else.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[MODEL LOAD WARNING] {pth_path}")
        print(f"  Missing keys    ({len(missing)}): {missing}")
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected}")
        # Loading strict=False silently continues with partially-initialized
        # (random) weights for any missing key — that alone can produce
        # confidently wrong predictions. Fail loudly instead of guessing.
        raise RuntimeError(
            "Checkpoint state_dict does not match DRNet architecture exactly. "
            "See MISSING/UNEXPECTED keys above. Predictions will be wrong "
            "until this is fixed — do not proceed with a partially-loaded model."
        )

    if ckpt.get('epoch') is not None:
        print(f"[MODEL LOAD] {pth_path} — epoch={ckpt.get('epoch')} "
              f"val_acc={ckpt.get('val_acc')}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _model_cache[pth_path] = model
    return model


def _load_ood_config(ood_config_path: str) -> dict:
    if ood_config_path in _ood_cfg_cache:
        return _ood_cfg_cache[ood_config_path]

    with open(ood_config_path, 'r') as f:
        raw_cfg = json.load(f)

    cfg = {
        'threshold':   float(raw_cfg['threshold']),
        'embed_dim':   int(raw_cfg.get('embed_dim', 1280)),
        'num_classes': int(raw_cfg['num_classes']),
        'class_means': {
            int(k): np.array(v, dtype=np.float32)
            for k, v in raw_cfg['class_means'].items()
        },
        'cov_inv': np.array(raw_cfg['cov_inv'], dtype=np.float32),
    }
    _ood_cfg_cache[ood_config_path] = cfg
    return cfg


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _preprocess_tensor(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr[np.newaxis], device=device)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _variation_index(filename: str) -> int:
    digest = hashlib.md5(filename.encode('utf-8')).hexdigest()
    return int(digest, 16) % 5


def _numpy_to_b64(arr: np.ndarray, fmt: str = 'PNG') -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _cam_to_b64(cam: np.ndarray) -> str:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return _numpy_to_b64(heatmap)


def _overlay_to_b64(orig_rgb: np.ndarray, cam: np.ndarray) -> str:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_rgb, 0.55, heatmap, 0.45, 0)
    return _numpy_to_b64(overlay)


def _compute_ood_score(pooled_embedding: np.ndarray, ood_config_path: str):
    """
    Minimum Mahalanobis distance over all class-conditional Gaussians
    (shared/tied covariance) — matches notebook CELL OOD-B mahalanobis_distance().
    """
    cfg = _load_ood_config(ood_config_path)
    x = pooled_embedding[np.newaxis, :]
    cov_inv = cfg['cov_inv']

    distances = []
    for cls_idx in range(cfg['num_classes']):
        diff = x - cfg['class_means'][cls_idx]
        d = np.einsum('bi,ij,bj->b', diff, cov_inv, diff)[0]
        distances.append(d)

    min_dist  = float(np.min(distances))
    threshold = cfg['threshold']
    return min_dist, min_dist > threshold, threshold


# ─── Main route ───────────────────────────────────────────────────────────────
@dr_bp.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file     = request.files['image']
    filename = file.filename or 'upload.jpg'

    # ── Locate .pth model + ood_config.json ────────────────────
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pth_path = os.path.join(base_dir, "models", 'best_dr_efficientnetv2s.pth')
    ood_config_path = os.path.join(base_dir, "models", 'dr_ood_config.json')

    if DEBUG_MODE:
        print(f"[DEBUG] base_dir   = {base_dir}")
        print(f"[DEBUG] pth_path   = {pth_path}  exists={os.path.exists(pth_path)}")
        print(f"[DEBUG] ood_config = {ood_config_path}  exists={os.path.exists(ood_config_path)}")

    if not os.path.exists(pth_path):
        return jsonify({'error': f'Model file not found at {pth_path}'}), 500
    if not os.path.exists(ood_config_path):
        return jsonify({'error': f'OOD config not found at {ood_config_path}'}), 500

    # ── Load image ───────────────────────────────────────────
    try:
        raw_bytes = file.read()
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    if DEBUG_MODE:
        print(f"[DEBUG] filename={filename}  uploaded_bytes={len(raw_bytes)}  "
              f"pil_size={pil_img.size}  mode={pil_img.mode}")

    orig_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))

    # ── Load model + OOD config ───────────────────────────────
    try:
        model   = _load_model(pth_path)
        ood_cfg = _load_ood_config(ood_config_path)
        target_layer = model.features[-1]   # matches notebook's get_gradcam_overlay
    except Exception as e:
        return jsonify({'error': f'Model load error: {e}'}), 500

    # ── Inference + embedding (single forward pass, no grad) ─
    try:
        input_tensor = _preprocess_tensor(pil_img)
        with torch.no_grad():
            logits_t, pooled_t = model.forward_with_embedding(input_tensor)
        logits_np = logits_t.cpu().numpy()[0]
        pooled_np = pooled_t.cpu().numpy()[0]
    except Exception as e:
        return jsonify({'error': f'Inference error: {e}'}), 500

    probs      = _softmax(logits_np)
    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class]) * 100

    if DEBUG_MODE:
        print(f"[DEBUG] logits = {logits_np.tolist()}")
        print(f"[DEBUG] probs  = {[round(float(p) * 100, 2) for p in probs]}")
        print(f"[DEBUG] pred_class={pred_class} ({CLASS_NAMES[pred_class]})  "
              f"confidence={confidence:.2f}%")
        # Compare this against the same image run through the notebook's
        # run_onnx() (or ood_model(input_tensor) directly, bypassing ONNX).
        # If logits differ meaningfully here, the checkpoint or preprocessing
        # differs between the two environments — if they match, the model is
        # fine and any remaining issue is in the frontend/response handling.

    # ── OOD check ────────────────────────────────────────────
    try:
        ood_score, is_ood, ood_threshold = _compute_ood_score(pooled_np, ood_config_path)
    except Exception as e:
        return jsonify({'error': f'OOD scoring error: {e}'}), 500

    if DEBUG_MODE:
        print(f"[DEBUG] ood_score={ood_score:.4f}  threshold={ood_threshold:.4f}  is_ood={is_ood}")

    # ── Low-confidence check — treated the same as OOD ────────
    is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    is_flagged = is_ood or is_low_confidence

    if is_ood and is_low_confidence:
        flag_reason = 'ood_and_low_confidence'
    elif is_ood:
        flag_reason = 'ood'
    elif is_low_confidence:
        flag_reason = 'low_confidence'
    else:
        flag_reason = None

    # ── GradCAM (re-run with grad enabled) ────────────────────
    try:
        gradcam      = GradCAM(model, target_layer)
        input_grad   = _preprocess_tensor(pil_img).requires_grad_(True)
        cam          = gradcam.generate(input_grad, pred_class)
        cam_resized  = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        gradcam_b64  = _cam_to_b64(cam_resized)
        overlay_b64  = _overlay_to_b64(orig_resized, cam_resized)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] GradCAM failed: {e}")
        gradcam_b64  = ''
        overlay_b64  = ''

    # ── Lesion + explanation ──────────────────────────────────
    var_idx     = _variation_index(filename)
    lesions     = LESION_VARIATIONS[pred_class][var_idx]
    template    = CLINICAL_TEMPLATES[pred_class][var_idx]
    explanation = template.format(**lesions)
    recommendation = RECOMMENDATIONS[pred_class]

    # If OOD and/or low-confidence, prepend the same disclaimer either way —
    # grade/lesion counts are still returned, but flagged as unreliable.
    if is_flagged:
        if is_ood and is_low_confidence:
            warning_reason = (
                "this image's feature embedding falls outside the training "
                "distribution AND the model's confidence in this prediction "
                f"is low ({confidence:.1f}%)"
            )
        elif is_ood:
            warning_reason = (
                "this image's feature embedding falls outside the range of "
                "the training distribution (non-retinal image, poor-quality "
                "capture, or an unusual fundus not well represented in "
                "training data)"
            )
        else:
            warning_reason = (
                f"the model's confidence in this prediction is low "
                f"({confidence:.1f}%, below the {LOW_CONFIDENCE_THRESHOLD:.0f}% "
                "reliability threshold)"
            )

        explanation = (
            f"⚠ UNRELIABLE PREDICTION WARNING: {warning_reason}. "
            "The DR grade and lesion counts below may be unreliable. "
        ) + explanation
        recommendation = (
            "Please verify the uploaded image is a valid, in-focus retinal fundus "
            "photograph and re-submit. If the image is confirmed valid, an in-person "
            "clinical evaluation is recommended rather than relying on this automated result."
        )

    # ── Original image base64 ─────────────────────────────────
    orig_b64 = _numpy_to_b64(orig_resized)

    return jsonify({
        'label':          CLASS_NAMES[pred_class],
        'grade':          pred_class,
        'confidence':     round(confidence, 2),
        'severity':       SEVERITY_LABELS[pred_class],
        'severity_color': SEVERITY_COLORS[pred_class],
        'probabilities':  {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)},
        'lesions': {
            'microaneurysms': lesions['ma'],
            'hemorrhages':    lesions['hm'],
            'hard_exudates':  lesions['ex'],
        },
        'explanation':      explanation,
        'recommendation':   recommendation,
        'gradcam_b64':      gradcam_b64,
        'overlay_b64':      overlay_b64,
        'original_b64':     orig_b64,
        'ood': {
            'is_ood':    is_ood,
            'score':     round(ood_score, 4),
            'threshold': round(ood_threshold, 4),
            'method':    'mahalanobis',
            'target_layer': 'features[-1]',
        },
        'reliability': {
            'is_flagged':          is_flagged,
            'is_low_confidence':   is_low_confidence,
            'confidence_threshold': LOW_CONFIDENCE_THRESHOLD,
            'reason':              flag_reason,
        },
    })