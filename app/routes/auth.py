from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from app import db
from app.models.user import User, Doctor, Patient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

auth_bp = Blueprint("auth", __name__)


## ---------------- REGISTER ----------------
@auth_bp.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        full_name = request.form.get("full_name")
        email = request.form.get("email")
        password = request.form.get("password")
        user_type = request.form.get("user_type")
        phone = request.form.get("phone")  # get phone once

        # check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("auth.login"))

        hashed_password = generate_password_hash(password)

        new_user = User(
            full_name=full_name,
            email=email,
            password=hashed_password,
            user_type=user_type
        )

        db.session.add(new_user)
        db.session.commit()

        # ---------------- DOCTOR ----------------
        if user_type == "doctor":

            doctor = Doctor(
                user_id=new_user.id,
                license_number=request.form.get("license_number"),
                specialization=request.form.get("specialization"),
                hospital_name=request.form.get("hospital_name"),
                medical_degree=request.form.get("medical_degree"),
                experience=request.form.get("experience"),
                phone=phone   # use stored phone variable
            )

            db.session.add(doctor)

        # ---------------- PATIENT ----------------
        elif user_type == "patient":

            dob = request.form.get("dob")

            patient = Patient(
                user_id=new_user.id,
                dob=datetime.strptime(dob, "%Y-%m-%d") if dob else None,
                gender=request.form.get("gender"),
                blood_type=request.form.get("blood_type"),
                phone=phone,   # use stored phone variable
                allergies=request.form.get("allergies"),
                address=request.form.get("address")
            )

            db.session.add(patient)

        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("auth.login"))

    return render_template("registration.html")

# ---------------- LOGIN ----------------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("Account not found. Please register first.", "danger")
            return redirect(url_for("auth.register"))

        if not check_password_hash(user.password, password):
            flash("Incorrect password. Try again.", "danger")
            return redirect(url_for("auth.login"))

        # session creation
        session["user_id"] = user.id
        session["user_type"] = user.user_type
        session["user_name"] = user.full_name

        flash(f"Welcome back {user.full_name}!", "success")

        # role based dashboard
        if user.user_type == "doctor":
            return redirect(url_for("home_bp.home"))

        if user.user_type == "patient":
            return redirect(url_for("home_bp.home"))

    return render_template("login.html")


# ---------------- LOGOUT ----------------
@auth_bp.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.", "info")

    return redirect(url_for("auth.login"))