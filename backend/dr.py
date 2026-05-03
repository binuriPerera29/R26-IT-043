"""
app.py  —  Flask backend for Diabetic Retinopathy Fundus Analyser
Run:  python app.py
"""
import os
from flask import Flask
from flask_cors import CORS
from routes.diabetic_retinopathy import dr_bp
from routes.dr_info               import dr_info_bp


def create_app():
    app = Flask(__name__)

    # Allow ALL origins (React dev server on 5173, or any other port)
    CORS(app, resources={r"/*": {"origins": "*"}},
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "OPTIONS"])

    app.register_blueprint(dr_bp,      url_prefix='/api')
    app.register_blueprint(dr_info_bp, url_prefix='/api')

    @app.route('/api/health', methods=['GET'])
    def health():
        return {
            'status': 'ok',
            'model':  'DRNet — EfficientNetV2-S',
            'task':   'Diabetic Retinopathy Classification (5-class)',
        }

    # Catch-all 404 with helpful message
    @app.errorhandler(404)
    def not_found(e):
        return {
            'error': '404 Not Found',
            'available_routes': ['/api/health', '/api/analyze', '/api/grades']
        }, 404

    return app


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app  = create_app()
    print("=" * 60)
    print("  Diabetic Retinopathy Analysis API")
    print("  EfficientNetV2-S | 5-Class DR | GradCAM")
    print(f"  http://localhost:{port}")
    print(f"  POST http://localhost:{port}/api/analyze")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=True)