from flask import Blueprint, request, render_template

home_bp = Blueprint("home_bp", __name__)

@home_bp.route("/", methods= ["GET"])
def home():
    return render_template("index.html")