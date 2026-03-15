from app import create_app
import os

app = create_app()

if __name__ == "__main__":
    # debug=True is optional; Flask detects from FLASK_ENV
    app.run(debug=True)
