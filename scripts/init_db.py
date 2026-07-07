import os
import sys

# Ensure project root is on sys.path so we can import app when running this script directly
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, db, User, Prediction


def init_db():
    with app.app_context():
        # Ensure the instance folder exists (Flask uses app.instance_path)
        try:
            os.makedirs(app.instance_path, exist_ok=True)
        except Exception:
            # Fallback: create an 'instance' directory next to project
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            os.makedirs(os.path.join(base, 'instance'), exist_ok=True)

        # Create all database tables
        db.create_all()
        print("Database initialized successfully!")

if __name__ == '__main__':
    init_db()