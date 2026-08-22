"""
=============================================================================
  Glaucoma Detection API — Flask Backend
  Project: R26-IT-043 | Chavindee M.A.P | IT22127778 | SLIIT
=============================================================================
"""

import os
import traceback
from datetime import datetime

import torch
import torch.nn as nn
from flask import Flask, jsonify
from flask_cors import CORS
import timm


# =============================================================================
#  EXACT model architecture from notebook — must match checkpoint to load correctly
# =============================================================================

class GlaucomaEfficientNetB0(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False,
            num_classes=0, global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.45),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# =============================================================================
#  PROJECT ROOT — auto-derived, no hardcoded absolute path
#  ───────────────────────────────────────────────────────
#  This resolves to the folder app.py itself lives in, no matter what drive,
#  username, or parent folder path it's copied/cloned into. So when you share
#  this project (zip, git clone, another PC, a server) it works immediately —
#  nothing to edit.
#
#  Assumes this layout (matches your project):
#      backend/
#        app.py                     ← this file
#        models/
#          best_model_b0_clean.pth
#          glaucoma_ood_config.json
#        routes/
#          glaucoma.py
#          cdr.py
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH      = os.path.join(PROJECT_ROOT, "models", "best_model_b0_clean.pth")
OOD_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "glaucoma_ood_config.json")


# =============================================================================
#  App factory
# =============================================================================

def create_app():
    app = Flask(__name__)
    CORS(app, origins="*")  # tighten in production

    app.config["MODEL"]       = None
    app.config["DEVICE"]      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.config["NUM_CLASSES"] = 3

    # EXACT class order from notebook: CLASS_NAMES = ["advanced", "early", "normal"]
    app.config["CLASS_NAMES"] = ["advanced", "early", "normal"]

    app.config["MODEL_PATH"]      = MODEL_PATH
    app.config["OOD_CONFIG_PATH"] = OOD_CONFIG_PATH

    # Debug visibility — print exactly what the process is looking for at boot.
    print(f"[BOOT] MODEL_PATH        = {MODEL_PATH}")
    print(f"[BOOT] MODEL_PATH exists = {os.path.exists(MODEL_PATH)}")
    print(f"[BOOT] OOD_CONFIG_PATH   = {OOD_CONFIG_PATH}")
    print(f"[BOOT] OOD_CONFIG exists = {os.path.exists(OOD_CONFIG_PATH)}")

    _load_model(app)

    # Blueprints
    from routes.glaucoma import glaucoma_bp
    app.register_blueprint(glaucoma_bp, url_prefix="/api/glaucoma")

    from routes.cdr import cdr_bp
    app.register_blueprint(cdr_bp, url_prefix="/api/cdr")

    # Future modules:
    # from routes.module3 import module3_bp
    # app.register_blueprint(module3_bp, url_prefix="/api/module3")

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status":            "ok",
            "device":            str(app.config["DEVICE"]),
            "model_loaded":      app.config["MODEL"] is not None,
            "model_path":        app.config["MODEL_PATH"],
            "model_path_exists": os.path.exists(app.config["MODEL_PATH"]),
            "ood_config_path":   app.config["OOD_CONFIG_PATH"],
            "ood_config_found":  os.path.exists(app.config["OOD_CONFIG_PATH"]),
            "timestamp":         datetime.utcnow().isoformat()
        })

    return app


# =============================================================================
#  Model loader
# =============================================================================

def _load_model(app):
    path   = app.config["MODEL_PATH"]
    device = app.config["DEVICE"]

    if not os.path.exists(path):
        print(f"[WARN] Model file not found at '{path}'. Prediction will be unavailable.")
        return

    try:
        model      = GlaucomaEfficientNetB0(num_classes=3).to(device)
        checkpoint = torch.load(path, map_location=device)

        # Resolve the state dict from whatever the notebook saved
        if isinstance(checkpoint, dict):
            for key in ("model_state", "model_state_dict", "state_dict"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"[INFO] Found weights under checkpoint key: '{key}'")
                    break
            else:
                state_dict = checkpoint   # checkpoint is the state_dict itself
                print("[INFO] Checkpoint is a raw state_dict (no wrapper key)")
        else:
            state_dict = checkpoint

        # strict=True so any architecture mismatch is caught loudly, not silently
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        app.config["MODEL"] = model
        print(f"[INFO] Model loaded from '{path}' on {device}")
        print(f"[INFO] Class order: {app.config['CLASS_NAMES']}")

        # Sanity log so it's obvious at boot whether OOD detection will work
        ood_path = app.config["OOD_CONFIG_PATH"]
        if os.path.exists(ood_path):
            print(f"[INFO] OOD config found at '{ood_path}' — Mahalanobis OOD check enabled")
        else:
            print(f"[WARN] OOD config NOT found at '{ood_path}' — "
                  f"Mahalanobis OOD check disabled (low-confidence-only rule still applies)")

    except RuntimeError as e:
        print(f"[ERROR] State dict mismatch — check architecture:\n{e}")
        traceback.print_exc()
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        traceback.print_exc()


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5004)