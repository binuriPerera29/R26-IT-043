"""
backend/routes/__init__.py
Registers all route blueprints with the Flask app.
"""

from flask import Flask
from .vds import vds_bp
# from .glaucoma import glaucoma_bp   # uncomment when ready


def register_routes(app: Flask) -> None:
    app.register_blueprint(vds_bp, url_prefix="/api/vds")
    # app.register_blueprint(glaucoma_bp, url_prefix="/api/glaucoma")
