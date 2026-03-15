from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.secret_key = "healthchain_secret"

    app.config.from_object(Config)

    db.init_app(app)
    migrate.init_app(app, db)

    # import models
    from app.models.user import User, Patient, Doctor

    # register blueprints
    from app.routes.index import home_bp
    from app.routes.risk_analysis import risk_bp 
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.profile import profile_bp


    app.register_blueprint(risk_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(profile_bp)

    return app