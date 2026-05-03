"""
app.py — Flask entry point for EYE OCT Analysis System
Model : EfficientNet-B0 fine-tuned on OCT2017 (Cell 4 of notebook)
"""

import os
from flask import Flask, request, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from routes.oct import oct_bp


def create_app() -> Flask:
    app = Flask(__name__)

    # Allow all origins — belt-and-suspenders approach
    CORS(app)

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.before_request
    def handle_options():
        if request.method == "OPTIONS":
            res = Response()
            res.headers["Access-Control-Allow-Origin"]  = "*"
            res.headers["Access-Control-Allow-Headers"] = "Content-Type"
            res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            return res, 200

    app.register_blueprint(oct_bp, url_prefix="/api")

    @app.route("/api/health", methods=["GET"])
    def health():
        model_path = os.environ.get("MODEL_PATH", "best_dr_efficientnetv2s.pth")
        return {
            "status": "ok",
            "model_architecture": "EfficientNet-B0",
            "model_file": model_path,
            "classes": ["CNV", "DME", "DRUSEN", "NORMAL"],
            "img_size": 224,
        }, 200

    return app


if __name__ == "__main__":
    port       = int(os.environ.get("PORT", 5000))
    model_path = os.environ.get("MODEL_PATH", "best_dr_efficientnetv2s.pth")
    app        = create_app()
    print(f"[EYE Backend] Running on http://localhost:{port}")
    print(f"[EYE Backend] Architecture : EfficientNet-B0 (notebook Cell 4)")
    print(f"[EYE Backend] Model file   : {model_path}")
    print(f"[EYE Backend] Classes      : CNV | DME | DRUSEN | NORMAL")
    print(f"[EYE Backend] Grad-CAM     : model.features[-1] (notebook Cell 10)")
    app.run(host="0.0.0.0", port=port, debug=True)