from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
from src.preprocessing.data_processor import prepare_features
import pandas as pd
import requests
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError

# Optional OpenAI import; only used if API key is present
try:
    import openai
except Exception:
    openai = None

load_dotenv()

# configure openai if key present
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or None
if OPENAI_API_KEY and openai is not None:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        pass

# helper to call OpenAI for enriched reasoning (optional)
def _call_openai_for_reasoning(context: dict) -> str:
    if not OPENAI_API_KEY or openai is None:
        return ''
    try:
        prompt = "You are an expert agronomist. Given inputs, recommend a primary and alternative crop, explain reasoning concisely. Return a short explanation."
        # build a user message with structured context
        user_msg = json.dumps(context, default=str, indent=2)
        resp = openai.ChatCompletion.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context:\n{user_msg}"}
            ],
            max_tokens=300,
            temperature=0.2
        )
        text = resp['choices'][0]['message']['content'].strip()
        return text
    except Exception:
        # Return a visible message rather than failing silently so the UI can show status
        try:
            return f'OpenAI call failed (see server logs)'
        except Exception:
            return 'OpenAI call failed'

def export_database_to_txt():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = "database_export.txt"
    
    with open(output_file, 'w') as f:
        # Write header with timestamp
        f.write(f"CropPredict Database Export\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write("="*50 + "\n\n")

        # Export Users
        f.write("USERS\n")
        f.write("-"*50 + "\n")
        users = User.query.all()
        if users:
            for user in users:
                f.write(f"User ID: {user.id}\n")
                f.write(f"Username: {user.username}\n")
                f.write(f"Email: {user.email}\n")
                f.write(f"Full Name: {user.full_name or 'Not set'}\n")
                f.write(f"Organization: {user.organization or 'Not set'}\n")
                f.write(f"Job Title: {user.job_title or 'Not set'}\n")
                f.write(f"Number of Predictions: {len(user.predictions)}\n")
                f.write("-"*30 + "\n")
        else:
            f.write("No users found in database.\n")
        f.write("\n")

        # Export Predictions
        f.write("PREDICTIONS\n")
        f.write("-"*50 + "\n")
        predictions = Prediction.query.all()
        if predictions:
            for pred in predictions:
                f.write(f"Prediction ID: {pred.id}\n")
                f.write(f"Date: {pred.date}\n")
                f.write(f"Crop: {pred.crop}\n")
                f.write(f"Yield Value: {pred.yield_value:.2f} tons/hectare\n")
                f.write("Environmental Parameters:\n")
                f.write(f"  Temperature: {pred.temperature}°C\n")
                f.write(f"  Humidity: {pred.humidity}%\n")
                f.write(f"  Rainfall: {pred.rainfall}mm\n")
                f.write("Soil Parameters:\n")
                f.write(f"  Nitrogen: {pred.nitrogen} kg/ha\n")
                f.write(f"  Phosphorus: {pred.phosphorus} kg/ha\n")
                f.write(f"  Potassium: {pred.potassium} kg/ha\n")
                f.write(f"  pH: {pred.ph}\n")
                f.write(f"Made by: {pred.author.username}\n")
                f.write("-"*30 + "\n")
        else:
            f.write("No predictions found in database.\n")
        f.write("\n")

        # Export Contact Submissions
        f.write("CONTACT SUBMISSIONS\n")
        f.write("-"*50 + "\n")
        contacts = Contact.query.all()
        if contacts:
            for contact in contacts:
                f.write(f"Contact ID: {contact.id}\n")
                f.write(f"Name: {contact.name}\n")
                f.write(f"Email: {contact.email}\n")
                f.write(f"Phone: {contact.phone}\n")
                f.write(f"Date: {contact.date}\n")
                f.write(f"Message:\n{contact.message}\n")
                f.write("-"*30 + "\n")
        else:
            f.write("No contact submissions found in database.\n")

app = Flask(__name__, 
            static_url_path='/static',
            static_folder='static')
# Ensure instance folder is used for the SQLite DB so the DB file is kept with the project
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
os.makedirs(instance_path, exist_ok=True)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
# Use a file-based SQLite DB inside the project's instance directory
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    predictions = db.relationship('Prediction', backref='author', lazy=True)
    
    # Additional profile fields
    full_name = db.Column(db.String(100))
    organization = db.Column(db.String(100))
    job_title = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    bio = db.Column(db.Text)
    expertise = db.Column(db.String(200))
    website = db.Column(db.String(200))
    profile_image = db.Column(db.String(200), server_default='default.jpg')
    date_joined = db.Column(db.DateTime, nullable=False, server_default=db.text('CURRENT_TIMESTAMP'))

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    crop = db.Column(db.String(100), nullable=False)
    yield_value = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Environmental parameters
    temperature = db.Column(db.Float, nullable=False, default=0.0)
    humidity = db.Column(db.Float, nullable=False, default=0.0)
    rainfall = db.Column(db.Float, nullable=False, default=0.0)
    
    # Soil parameters
    nitrogen = db.Column(db.Float, nullable=False, default=0.0)
    phosphorus = db.Column(db.Float, nullable=False, default=0.0)
    potassium = db.Column(db.Float, nullable=False, default=0.0)
    ph = db.Column(db.Float, nullable=False, default=7.0)

    def __repr__(self):
        return f"Prediction('{self.crop}', '{self.yield_value}', '{self.date}')"
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'crop': self.crop,
            'yield_value': self.yield_value,
            'user_id': self.user_id,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'rainfall': self.rainfall,
            'nitrogen': self.nitrogen,
            'phosphorus': self.phosphorus,
            'potassium': self.potassium,
            'ph': self.ph
        }

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"Contact('{self.name}', '{self.email}', '{self.date}')"
@event.listens_for(Contact, 'after_update')
@event.listens_for(Contact, 'after_delete')
def export_on_change(mapper, connection, target):
    """Export database content whenever there's a change"""
    export_database_to_txt()

@login_manager.user_loader
def load_user(user_id):
    # Use Session.get instead of the legacy Query.get() to avoid SQLAlchemy 2.0
    # deprecation warnings. `db.session` is the active Session for Flask-SQLAlchemy.
    return db.session.get(User, int(user_id))

class ContactForm(FlaskForm):
    name = StringField('Name', validators=[
        DataRequired(),
        Length(min=2, max=100)
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    phone = StringField('Phone Number', validators=[
        DataRequired(),
        Length(min=10, max=20)
    ])
    message = StringField('Message', validators=[
        DataRequired(),
        Length(min=10, max=1000)
    ])
    submit = SubmitField('Send Message')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=2, max=20)
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=6)
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password')
    ])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username is already taken.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email is already registered.')


class UpdateProfileForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=2, max=20)
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    full_name = StringField('Full Name', validators=[
        Length(max=100)
    ])
    organization = StringField('Organization/Farm Name', validators=[
        Length(max=100)
    ])
    job_title = StringField('Job Title', validators=[
        Length(max=100)
    ])
    phone = StringField('Phone Number', validators=[
        Length(max=20)
    ])
    address = StringField('Address', validators=[
        Length(max=200)
    ])
    bio = StringField('Bio/About', validators=[
        Length(max=500)
    ])
    expertise = StringField('Areas of Expertise', validators=[
        Length(max=200)
    ])
    website = StringField('Website', validators=[
        Length(max=200)
    ])
    submit = SubmitField('Update Profile')

    def __init__(self, original_username, original_email, *args, **kwargs):
        super(UpdateProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username
        self.original_email = original_email

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('Username is already taken.')

    def validate_email(self, email):
        if email.data != self.original_email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('Email is already registered.')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        # Use a secure PBKDF2-based hashing algorithm
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=hashed_password
        )
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/')
def home():
    return render_template('home.html', title='Home')

@app.route('/dashboard')
@login_required
def dashboard():
    from datetime import datetime, timedelta
    
    # Get filter period from query params (default to 'all')
    period = request.args.get('period', 'all')
    
    # Base query
    query = Prediction.query.filter_by(author=current_user)
    
    # Current time for calculations
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    
    # Get recent predictions count (last 7 days)
    recent_predictions_count = Prediction.query.filter_by(author=current_user)\
        .filter(Prediction.date >= week_ago).count()
    
    # Apply time period filter to main query
    if period == 'week':
        query = query.filter(Prediction.date >= week_ago)
    elif period == 'month':
        query = query.filter(Prediction.date >= now - timedelta(days=30))
    
    # Get predictions ordered by date
    predictions = query.order_by(Prediction.date.desc()).all()
    
    # Convert predictions to dictionaries for JSON serialization
    predictions_list = [pred.to_dict() for pred in predictions]
    
    return render_template(
        'dashboard.html',
        title='Dashboard',
        predictions=predictions_list,
        recent_predictions_count=recent_predictions_count
    )


@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/reset-db')
def reset_db():
    """Temporary route to reset the database - remove in production!"""
    db.drop_all()
    db.create_all()
    flash('Database has been reset!', 'success')
    return redirect(url_for('home'))


@app.route('/ai-insights')
@login_required
def ai_insights():
    return render_template('ai_insights.html', title='AI Insights')


def _weather_description(code):
    descriptions = {
        0: 'Clear sky',
        1: 'Mainly clear',
        2: 'Partly cloudy',
        3: 'Overcast',
        45: 'Fog',
        48: 'Depositing rime fog',
        51: 'Light drizzle',
        53: 'Moderate drizzle',
        55: 'Dense drizzle',
        61: 'Slight rain',
        63: 'Moderate rain',
        65: 'Heavy rain',
        71: 'Slight snow',
        73: 'Moderate snow',
        75: 'Heavy snow',
        95: 'Thunderstorm',
        96: 'Thunderstorm with hail',
        99: 'Thunderstorm with heavy hail'
    }
    return descriptions.get(code, 'Weather update')


def _weather_icon(code):
    if code in [0, 1]:
        return '☀️'
    if code in [2, 3]:
        return '☁️'
    if code in [45, 48]:
        return '🌫️'
    if code in [51, 53, 55, 61, 63, 65]:
        return '🌧️'
    if code in [71, 73, 75]:
        return '❄️'
    if code in [95, 96, 99]:
        return '⛈️'
    return '🌦️'


def _get_weather_payload(location):
    geocode_url = (
        'https://geocoding-api.open-meteo.com/v1/search'
        f'?name={location}&count=1&language=en&format=json'
    )
    geocode_response = requests.get(geocode_url, timeout=10)
    geocode_response.raise_for_status()
    geocode_data = geocode_response.json().get('results', [])
    if not geocode_data:
        raise ValueError('Location not found')

    location_data = geocode_data[0]
    latitude = location_data.get('latitude')
    longitude = location_data.get('longitude')

    today = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = today

    forecast_url = (
        'https://api.open-meteo.com/v1/forecast'
        f'?latitude={latitude}&longitude={longitude}'
        '&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code'
        '&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_mean'
        '&forecast_days=7&timezone=auto'
    )
    forecast_response = requests.get(forecast_url, timeout=10)
    forecast_response.raise_for_status()
    forecast_data = forecast_response.json()

    archive_url = (
        'https://archive-api.open-meteo.com/v1/archive'
        f'?latitude={latitude}&longitude={longitude}'
        f'&start_date={start_date}&end_date={end_date}'
        '&daily=temperature_2m_mean,precipitation_sum,weather_code'
        '&timezone=auto'
    )
    archive_response = requests.get(archive_url, timeout=10)
    archive_response.raise_for_status()
    archive_data = archive_response.json()

    forecast_days = forecast_data.get('daily', {})
    history_days = archive_data.get('daily', {})

    current = forecast_data.get('current', {})
    # safe accessor in case arrays are shorter/missing than 'time'
    def _safe_get(arr, idx):
        if not isinstance(arr, (list, tuple)):
            return None
        return arr[idx] if idx < len(arr) else None

    forecast_items = []
    for index, date in enumerate(forecast_days.get('time', [])[:7]):
        code = _safe_get(forecast_days.get('weather_code', []), index)
        forecast_items.append({
            'date': date,
            'max_temp': _safe_get(forecast_days.get('temperature_2m_max', []), index),
            'min_temp': _safe_get(forecast_days.get('temperature_2m_min', []), index),
            'chance_of_rain': _safe_get(forecast_days.get('precipitation_probability_mean', []), index),
            'weather_code': code,
            'description': _weather_description(code),
            'icon': _weather_icon(code)
        })

    history_items = []
    for index, date in enumerate(history_days.get('time', [])[:7]):
        code = _safe_get(history_days.get('weather_code', []), index)
        history_items.append({
            'date': date,
            'avg_temp': _safe_get(history_days.get('temperature_2m_mean', []), index),
            'precipitation': _safe_get(history_days.get('precipitation_sum', []), index),
            'weather_code': code,
            'description': _weather_description(code),
            'icon': _weather_icon(code)
        })

    return {
        'location': {
            'name': location_data.get('name'),
            'country': location_data.get('country'),
            'latitude': latitude,
            'longitude': longitude
        },
        'current': {
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'wind_speed': current.get('wind_speed_10m'),
            'weather_code': current.get('weather_code'),
            'description': _weather_description(current.get('weather_code')),
            'icon': _weather_icon(current.get('weather_code'))
        },
        'forecast': forecast_items,
        'history': history_items
    }


@app.route('/api/weather')
def weather_api():
    location = request.args.get('location', 'New York').strip() or 'New York'
    try:
        payload = _get_weather_payload(location)
        return jsonify(payload)
    except requests.RequestException as exc:
        return jsonify({'error': str(exc)}), 502
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 404


@app.route('/api/ai/recommendations', methods=['POST'])
@login_required
def ai_recommendations():
    """Simple AI-style recommendations endpoint.

    Accepts JSON body with fields like: location, area, ph, nitrogen, phosphorus,
    potassium, rainfall, season, irrigation (yes/no), preferred_crops (list)
    Returns a JSON with primary and alternative crop recommendations, expected
    yields, confidence scores and short reasoning.
    """
    data = request.get_json() or {}
    # basic required fields
    location = data.get('location', '').strip() or None
    area = float(data.get('area') or 0)

    # soil and environment
    ph = data.get('ph')
    nitrogen = data.get('nitrogen')
    phosphorus = data.get('phosphorus')
    potassium = data.get('potassium')
    rainfall = data.get('rainfall')
    season = data.get('season') or 'general'
    irrigation = data.get('irrigation') in [True, 'true', 'yes', '1']
    preferred = data.get('preferred_crops') or []
    # Optional: evaluate a specific crop (user-provided) instead of full recommendation
    requested_crop = (data.get('crop') or data.get('requested_crop') or '').strip() or None
    prefer_requested = data.get('prefer_requested') in [True, 'true', 'yes', '1']
    use_openai = data.get('use_openai') in [True, 'true', 'yes', '1']

    try:
        ph = float(ph) if ph is not None else None
    except Exception:
        ph = None
    try:
        nitrogen = float(nitrogen) if nitrogen is not None else None
    except Exception:
        nitrogen = None
    try:
        phosphorus = float(phosphorus) if phosphorus is not None else None
    except Exception:
        phosphorus = None
    try:
        potassium = float(potassium) if potassium is not None else None
    except Exception:
        potassium = None
    try:
        rainfall = float(rainfall) if rainfall is not None else None
    except Exception:
        rainfall = None

    # If location is provided, enrich with geocode + recent archive rainfall and air quality
    location_info = None
    pollution_risk = False
    pollution_level = None
    if location:
        try:
            geocode_url = (
                'https://geocoding-api.open-meteo.com/v1/search'
                f'?name={location}&count=1&language=en&format=json'
            )
            gresp = requests.get(geocode_url, timeout=8)
            gresp.raise_for_status()
            gres = gresp.json().get('results') or []
            if gres:
                locd = gres[0]
                lat = locd.get('latitude')
                lon = locd.get('longitude')
                location_info = {'name': locd.get('name'), 'country': locd.get('country'), 'latitude': lat, 'longitude': lon}

                # get last 365 days archive precipitation sum
                end_date = datetime.utcnow().strftime('%Y-%m-%d')
                start_date = (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d')
                archive_url = (
                    'https://archive-api.open-meteo.com/v1/archive'
                    f'?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}'
                    '&daily=precipitation_sum&timezone=auto'
                )
                aresp = requests.get(archive_url, timeout=10)
                aresp.raise_for_status()
                ad = aresp.json().get('daily', {})
                precip_list = ad.get('precipitation_sum') or []
                total_precip = sum([float(x) for x in precip_list]) if precip_list else None
                if total_precip is not None:
                    # approximate annual rainfall
                    rainfall = total_precip

                # check basic air quality (pm2_5) via Open-Meteo air-quality API
                try:
                    aq_url = (
                        'https://air-quality-api.open-meteo.com/v1/air-quality'
                        f'?latitude={lat}&longitude={lon}&hourly=pm2_5&timezone=auto'
                    )
                    aqresp = requests.get(aq_url, timeout=8)
                    aqresp.raise_for_status()
                    aqj = aqresp.json()
                    hourly = aqj.get('hourly', {})
                    pm25 = hourly.get('pm2_5') or []
                    if pm25:
                        # compute rough average
                        pm25_vals = [float(x) for x in pm25 if x is not None]
                        if pm25_vals:
                            pollution_level = sum(pm25_vals) / len(pm25_vals)
                            # threshold (µg/m3) - WHO guideline ~5, practical threshold set higher
                            if pollution_level > 25:
                                pollution_risk = True
                except Exception:
                    # ignore air-quality failures
                    pollution_level = None
        except Exception:
            # geocode/archive failure - ignore and continue with provided inputs
            location_info = None

    # Simple heuristic rules for demo purposes
    candidates = []
    # Example heuristics -- expand as needed
    if ph is not None:
        if 6.0 <= ph <= 7.5:
            candidates.append(('Wheat', 0.9))
            candidates.append(('Barley', 0.75))
        elif ph < 6.0:
            candidates.append(('Rice', 0.85))
            candidates.append(('Maize', 0.6))
        else:
            candidates.append(('Maize', 0.85))
            candidates.append(('Sorghum', 0.6))
    else:
        candidates.extend([('Maize', 0.7), ('Wheat', 0.65)])

    # rainfall and irrigation adjustments
    if rainfall is not None:
        if rainfall < 300:  # low rainfall annual
            candidates = [(c, s - 0.15 if c in ['Rice'] else s + 0.05) for c, s in candidates]
        elif rainfall > 800:
            candidates = [(c, s + 0.1 if c in ['Rice'] else s) for c, s in candidates]

    if irrigation:
        candidates = [(c, min(1.0, s + 0.08)) for c, s in candidates]

    # prefer user's preferred crops
    if preferred:
        preferred = [p.lower() for p in preferred]
        for i, (c, s) in enumerate(candidates):
            if c.lower() in preferred:
                candidates[i] = (c, min(1.0, s + 0.12))

    # sort by score
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    primary = candidates[0] if candidates else ('Maize', 0.6)
    alternative = candidates[1] if len(candidates) > 1 else None

    # crude yield estimates based on crop and area
    def estimate_yield(crop, area_val):
        base = {
            'Wheat': 4.8,
            'Maize': 6.5,
            'Rice': 5.0,
            'Barley': 3.8,
            'Sorghum': 3.5
        }.get(crop, 4.0)
        # adjust by score and area
        return round(base * (1 + (0.2 if area_val and area_val > 10 else 0)) , 2)

    result = {
        'location': location or 'unspecified',
        'area': area,
        'location_info': location_info,
        'pollution_risk': pollution_risk,
        'pollution_level_pm2_5': round(pollution_level, 2) if pollution_level is not None else None,
        'primary': {
            'crop': primary[0],
            'confidence': round(primary[1]*100, 1),
            'expected_yield_t_per_ha': estimate_yield(primary[0], area)
        },
        'alternative': None,
        'reasoning': [],
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    # If the user asked to evaluate a specific crop, compute its score/yield
    if requested_crop:
        # try to find in candidates case-insensitively
        found = None
        for c, s in candidates:
            if c.lower() == requested_crop.lower():
                found = (c, s)
                break
        if not found:
            # If not present, create a baseline score for the requested crop
            # baseline depends on crop family; simple heuristic
            baseline_score = 0.5
            if requested_crop.lower() in ['wheat', 'maize', 'rice']:
                baseline_score = 0.65
            found = (requested_crop.title(), baseline_score)

        result['requested'] = {
            'crop': found[0],
            'confidence': round(found[1] * 100, 1),
            'expected_yield_t_per_ha': estimate_yield(found[0], area)
        }

    if alternative:
        result['alternative'] = {
            'crop': alternative[0],
            'confidence': round(alternative[1]*100, 1),
            'expected_yield_t_per_ha': estimate_yield(alternative[0], area)
        }

    # populate reasoning
    if ph is not None:
        result['reasoning'].append(f"Soil pH {ph} influences suitability; recommended {result['primary']['crop']}")
    if rainfall is not None:
        result['reasoning'].append(f"Annual rainfall {rainfall} mm considered in selection")
    if irrigation:
        result['reasoning'].append('Irrigation available increases viable crop options')
    if preferred:
        result['reasoning'].append('User preferred crops were given and boosted in ranking')

    # If pollution risk is detected, reduce irrigation-related boosts and warn
    if pollution_risk:
        result['reasoning'].append('High local PM2.5 indicates pollution risk — irrigation and water quality may be unsuitable')
        # demote crops that depend heavily on irrigation (simple heuristic)
        for idx, (cname, score) in enumerate(candidates):
            candidates[idx] = (cname, max(0.0, score - 0.15))
        # recompute primary/alternative if requested
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        result['primary'] = {
            'crop': candidates[0][0],
            'confidence': round(candidates[0][1]*100, 1),
            'expected_yield_t_per_ha': estimate_yield(candidates[0][0], area)
        }
        if len(candidates) > 1:
            result['alternative'] = {
                'crop': candidates[1][0],
                'confidence': round(candidates[1][1]*100, 1),
                'expected_yield_t_per_ha': estimate_yield(candidates[1][0], area)
            }

    result['confidence_overall'] = round(result['primary']['confidence'] * 0.9, 1)

    # If user prefers the requested crop, boost its score and recompute
    if requested_crop and prefer_requested:
        # boost requested crop in candidates
        boosted = False
        for i, (c, s) in enumerate(candidates):
            if c.lower() == requested_crop.lower():
                candidates[i] = (c, min(1.0, s + 0.25))
                boosted = True
                break
        if not boosted:
            candidates.append((requested_crop.title(), 0.6 + 0.25))
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        result['primary'] = {
            'crop': candidates[0][0],
            'confidence': round(candidates[0][1]*100, 1),
            'expected_yield_t_per_ha': estimate_yield(candidates[0][0], area)
        }
        if len(candidates) > 1:
            result['alternative'] = {
                'crop': candidates[1][0],
                'confidence': round(candidates[1][1]*100, 1),
                'expected_yield_t_per_ha': estimate_yield(candidates[1][0], area)
            }

    # Optionally call OpenAI to produce a natural-language reasoning summary
    if use_openai:
        # prepare context
        ctx = {
            'input': {
                'location': location,
                'area': area,
                'ph': ph,
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'rainfall': rainfall,
                'irrigation': irrigation,
                'preferred': preferred,
                'location_info': location_info,
                'pollution_risk': pollution_risk,
                'pollution_level_pm2_5': pollution_level
            },
            'candidates': candidates,
            'primary': result.get('primary'),
            'alternative': result.get('alternative')
        }
        # ensure OpenAI key and client available
        if not (OPENAI_API_KEY and openai is not None):
            result['nl_reasoning'] = 'OpenAI API key not configured on server; enable OpenAI to get enriched reasoning.'
            result['openai_available'] = False
        else:
            nl = _call_openai_for_reasoning(ctx)
            if nl:
                result['nl_reasoning'] = nl
                result['openai_available'] = True

    return jsonify(result)


@app.route('/weather')
@login_required
def weather():
    return render_template('weather.html', title='Weather Integration')


@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html', title='Data Analytics')


@app.route('/predict')
@login_required
def predict_page():
    return render_template('predict.html', title='Predict')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = UpdateProfileForm(current_user.username, current_user.email)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.full_name = form.full_name.data
        current_user.organization = form.organization.data
        current_user.job_title = form.job_title.data
        current_user.phone = form.phone.data
        current_user.address = form.address.data
        current_user.bio = form.bio.data
        current_user.expertise = form.expertise.data
        current_user.website = form.website.data
        
        db.session.commit()
        flash('Your profile has been updated successfully!', 'success')
        return redirect(url_for('profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
        form.full_name.data = current_user.full_name
        form.organization.data = current_user.organization
        form.job_title.data = current_user.job_title
        form.phone.data = current_user.phone
        form.address.data = current_user.address
        form.bio.data = current_user.bio
        form.expertise.data = current_user.expertise
        form.website.data = current_user.website
    
    return render_template('profile.html', title='Profile', form=form)


@app.route('/prediction/<int:prediction_id>/delete', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Ensure users can only delete their own predictions
    if prediction.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    try:
        db.session.delete(prediction)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html', title='Settings')


# Load the trained model
model = None
scaler = None
encoders = None
feature_columns = None
feature_defaults = None


def load_model():
    global model, scaler, encoders, feature_columns, feature_defaults
    model_path = os.path.join('models', 'crop_yield_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    encoders_path = os.path.join('models', 'encoders.pkl')
    feat_path = os.path.join('models', 'feature_columns.pkl')
    defaults_path = os.path.join('models', 'feature_defaults.pkl')

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
    if os.path.exists(feat_path):
        feature_columns = joblib.load(feat_path)
    if os.path.exists(defaults_path):
        feature_defaults = joblib.load(defaults_path)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json() or {}

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Get the crop type from the input data
        crop_type = data.get('crop', 'unknown')

        # Ensure all expected feature columns are present (fill missing with defaults)
        if feature_columns is not None:
            for col in feature_columns:
                if col not in input_df.columns:
                    if feature_defaults is not None and col in feature_defaults:
                        input_df[col] = feature_defaults[col]
                    else:
                        input_df[col] = 0 if col in ['temperature', 'humidity', 'ph', 'rainfall', 'nitrogen', 'phosphorus', 'potassium'] else ''
            input_df = input_df.reindex(columns=feature_columns)

        # Preprocess input data using saved scaler/encoders
        processed_data, _, _ = prepare_features(input_df, scaler=scaler, encoders=encoders, fit=False)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Store prediction in database
        new_prediction = Prediction(
            crop=crop_type,
            yield_value=float(prediction),
            author=current_user
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'stored_prediction': new_prediction.to_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/fetch_nasa', methods=['POST'])
def fetch_nasa():
    """Simple fetcher for NASA POWER daily data. Expects JSON with lat, lon, start, end.
    Returns saved CSV path and a small preview.
    """
    try:
        payload = request.json or {}
        lat = float(payload.get('lat'))
        lon = float(payload.get('lon'))
        start = payload.get('start')  # YYYYMMDD
        end = payload.get('end')

        if not (start and end):
            return jsonify({'success': False, 'error': 'start and end dates are required (YYYYMMDD)'}), 400

        BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'start': start,
            'end': end,
            'latitude': lat,
            'longitude': lon,
            'parameters': 'T2M, T2M_MIN, T2M_MAX, RH2M, PRECTOT',
            'format': 'CSV'
        }

        resp = requests.get(BASE, params=params, timeout=30)
        resp.raise_for_status()

        os.makedirs('data', exist_ok=True)
        filename = f"nasa_power_{lat}_{lon}_{start}_{end}.csv"
        path = os.path.join('data', filename)
        with open(path, 'w') as f:
            f.write(resp.text)

        # Return small preview
        df = pd.read_csv(path)
        preview = df.head(10).to_dict(orient='records')

        return jsonify({'success': True, 'file': path, 'preview': preview})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        contact_entry = Contact(
            name=form.name.data,
            email=form.email.data,
            phone=form.phone.data,
            message=form.message.data
        )
        db.session.add(contact_entry)
        db.session.commit()
        flash('Your message has been sent successfully! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html', form=form)


def local_planner(query, extras=None):
    """Fallback planner when OpenAI API key is not available. Returns a small list of suggestions.
    This ensures the endpoint is useful for demos/tests without external API access.
    """
    # Provide a default suggestion around Bengaluru, India for demo purposes
    return [
        {
            'source': 'nasa_power',
            'name': 'nasa_power_by_location',
            'description': 'Daily NASA POWER for requested lat/lon and date range',
            'params': {
                'lat': extras.get('lat', 12.9716) if extras else 12.9716,
                'lon': extras.get('lon', 77.5946) if extras else 77.5946,
                'start': extras.get('start', '20230101') if extras else '20230101',
                'end': extras.get('end', '20231231') if extras else '20231231'
            }
        }
    ]


def call_openai_planner(query):
    """Call OpenAI to request a structured data-fetch plan. The function-calling schema is recommended but
    we implement a simple chat/completion request here that expects JSON in the assistant reply."""
    if openai is None or not OPENAI_API_KEY:
        return None

    try:
        prompt = (
            "You are a helpful assistant that suggests data sources to fetch crop-weather related data. "
            "Given a user query, return a JSON array of suggestions where each suggestion has fields: 'source' (e.g. 'nasa_power'), "
            "'name', 'description', and 'params' (an object with the parameters needed to fetch, such as lat, lon, start, end). "
            "Only suggest well-known, safe public data sources like 'nasa_power'."
            "Respond with only the JSON array."
            "\nUser query: " + query
        )

        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.0
        )

        assistant_text = resp['choices'][0]['message']['content']
        # attempt to parse JSON from assistant_text
        suggestions = json.loads(assistant_text)
        return suggestions
    except Exception as e:
        app.logger.error(f"OpenAI planner error: {e}")
        return None


@app.route('/fetch_via_chatgpt', methods=['POST'])
def fetch_via_chatgpt():
    """Endpoint to accept a user query, ask OpenAI for a data fetch plan (if available), and return a list of safe suggestions.
    If OpenAI key is not configured, uses a local fallback planner to return a demo suggestion.

    Request JSON: { 'query': 'get daily weather for my farm', 'lat': 12.9, 'lon': 77.6, 'start': '20230101', 'end': '20231231' }
    Response JSON: { 'success': True, 'suggestions': [ {source,name,description,params}, ... ] }
    """
    try:
        payload = request.json or {}
        query = payload.get('query', '')
        extras = {k: payload.get(k) for k in ('lat', 'lon', 'start', 'end') if payload.get(k) is not None}

        suggestions = None
        if OPENAI_API_KEY and openai is not None:
            suggestions = call_openai_planner(query)

        if not suggestions:
            # fallback planner for demos and tests
            suggestions = local_planner(query, extras=extras)

        # Strict whitelist: only keep safe sources
        allowed = ['nasa_power']
        filtered = [s for s in suggestions if s.get('source') in allowed]

        return jsonify({'success': True, 'suggestions': filtered})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Load the model before starting the server
    load_model()
    app.run(debug=True)
