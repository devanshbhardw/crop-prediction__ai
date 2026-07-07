import os, sys
# Ensure project root is importable
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, db, User
from werkzeug.security import generate_password_hash

ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@example.com')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'adminpass')


def seed_admin():
    with app.app_context():
        user = User.query.filter_by(email=ADMIN_EMAIL).first()
        if user:
            print(f"Admin user already exists: {user.email}")
            return
        # Use a secure PBKDF2-based hashing algorithm
        hashed = generate_password_hash(ADMIN_PASSWORD, method='pbkdf2:sha256')
        admin = User(username=ADMIN_USERNAME, email=ADMIN_EMAIL, password=hashed)
        db.session.add(admin)
        db.session.commit()
        print(f"Created admin user: {ADMIN_EMAIL} (username: {ADMIN_USERNAME})")


if __name__ == '__main__':
    seed_admin()
