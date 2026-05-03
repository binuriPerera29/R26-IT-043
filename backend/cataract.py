"""
backend/app.py
Flask application entry point.
"""

from flask import Flask
from flask_cors import CORS

from routes import register_routes


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)                        # allow React dev server (localhost:5173)
    register_routes(app)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5002, debug=True)