from flask import Blueprint, render_template, session, redirect, url_for
from app.models.user import User

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/dashboard")
def dashboard():

    # check login
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    # get logged in user
    user = User.query.get(session["user_id"])

    doctor = None
    patient = None

    # detect user type
    if user.user_type == "doctor":
        doctor = user.doctor

    if user.user_type == "patient":
        patient = user.patient

    return render_template(
        "dashboard.html",
        user=user,
        doctor=doctor,
        patient=patient
    )