from flask import Blueprint, render_template, session, redirect, url_for
from app.models.user import User
from datetime import datetime

profile_bp = Blueprint("profile_bp", __name__)


@profile_bp.route("/profile")
def profile():

    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user = User.query.get(session["user_id"])

    if user.user_type == "doctor":
        return render_template(
            "doctor_profile.html",
            doctor=user.doctor
        )

    elif user.user_type == "patient":
        return render_template(
        "patient_profile.html",
        patient=user.patient,
        now=datetime.now()
)

    return redirect(url_for("auth.login"))