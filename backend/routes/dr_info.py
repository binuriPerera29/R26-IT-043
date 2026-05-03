"""
dr_info.py — Diabetic Retinopathy informational routes
"""
from flask import Blueprint, jsonify

dr_info_bp = Blueprint('dr_info', __name__)

DR_GRADES = {
    0: {"name": "No DR",            "color": "#00ff88", "action": "Annual screening recommended."},
    1: {"name": "Mild DR",          "color": "#aaff00", "action": "Follow-up in 12 months."},
    2: {"name": "Moderate DR",      "color": "#ffaa00", "action": "Referral within 3–6 months."},
    3: {"name": "Severe DR",        "color": "#ff6600", "action": "Urgent referral within 1 month."},
    4: {"name": "Proliferative DR", "color": "#ff4444", "action": "Immediate ophthalmology referral."},
}

@dr_info_bp.route('/grades', methods=['GET'])
def get_grades():
    return jsonify(DR_GRADES)
