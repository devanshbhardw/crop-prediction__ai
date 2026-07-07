# Crop Yield Prediction Web App

This repository contains a Flask-based crop yield prediction application with AI-powered insights, weather integration, and a web dashboard.

## Features

- Crop yield prediction using saved machine learning models
- AI recommendation endpoint with optional OpenAI reasoning
- Weather integration via Open-Meteo APIs
- User login, profile, and prediction history
- Database-backed Flask application using SQLite
- Frontend templates, charts, and theme styling

## Project Structure

```
/ (project root)
├── app.py                 # Main Flask application
├── data/                  # Dataset storage
├── instance/              # SQLite database and instance files
├── migrations/            # Alembic DB migration scripts
├── models/                # Saved ML model artifacts
├── notebooks/             # Data exploration notebooks
├── scripts/               # Helper scripts
├── src/                   # ML and preprocessing code
├── static/                # CSS, JS, images
├── templates/             # Flask Jinja2 templates
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Prerequisites

- Python 3.10+
- `venv` or other virtual environment tool
- Internet access for API calls (Open-Meteo, NASA, optional OpenAI)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the SQLite database:
   ```bash
   cd /var/www/html/PRoject/crop
   /var/www/html/PRoject/crop/venv/bin/python - <<'PY'
from app import app, db
with app.app_context():
    db.create_all()
PY
   ```

4. Create a first user if needed, or clear browser session cookies if a stale login cookie is present.

## Environment Variables

Create a `.env` file or export these variables before running the app:

```bash
SECRET_KEY='your-secret-key'
OPENAI_API_KEY='sk-...'
```

- `SECRET_KEY`: Flask secret key for sessions and CSRF protection
- `OPENAI_API_KEY`: Optional API key for OpenAI-powered reasoning

## Running the App

Start the Flask app from the project root:

```bash
/var/www/html/PRoject/crop/venv/bin/python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Testing

Run the test suite with:

```bash
/var/www/html/PRoject/crop/venv/bin/python -m pytest tests
```

## Notes

- The app uses `instance/site.db` for SQLite storage.
- If you see a `no such table: user` error, it means the database schema was not created or your session is trying to load a missing user. Run the database initialization command above and/or clear your browser cookies.
- OpenAI integration is optional; AI recommendations work without it, but the natural-language reasoning panel requires `OPENAI_API_KEY`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push your branch
5. Open a pull request

## License

This project is licensed under the MIT License.# crop-prediction__ai
# crop-prediction__ai
