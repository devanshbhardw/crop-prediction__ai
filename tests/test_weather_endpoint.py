import json
from unittest.mock import Mock, patch

from app import app, db


def test_weather_api_returns_current_forecast_and_history():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client, app.app_context():
        db.create_all()

        geocode_response = Mock()
        geocode_response.status_code = 200
        geocode_response.json.return_value = {
            'results': [
                {'name': 'Berlin', 'latitude': 52.52, 'longitude': 13.41, 'country_code': 'DE'}
            ]
        }

        forecast_response = Mock()
        forecast_response.status_code = 200
        forecast_response.json.return_value = {
            'current': {
                'temperature_2m': 2.4,
                'relative_humidity_2m': 84,
                'wind_speed_10m': 11.9,
                'weather_code': 1
            },
            'daily': {
                'time': ['2026-07-07', '2026-07-08'],
                'temperature_2m_max': [24.0, 26.0],
                'temperature_2m_min': [16.0, 17.0],
                'precipitation_probability_mean': [20.0, 10.0],
                'weather_code': [1, 2]
            }
        }

        archive_response = Mock()
        archive_response.status_code = 200
        archive_response.json.return_value = {
            'daily': {
                'time': ['2026-07-01', '2026-07-02'],
                'temperature_2m_mean': [18.0, 19.0],
                'precipitation_sum': [4.0, 0.0]
            }
        }

        with patch('app.requests.get', side_effect=[geocode_response, forecast_response, archive_response]):
            response = client.get('/api/weather?location=Berlin')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['location']['name'] == 'Berlin'
        assert data['current']['temperature'] == 2.4
        assert data['current']['humidity'] == 84
        assert data['forecast'][0]['date'] == '2026-07-07'
        assert data['history'][0]['date'] == '2026-07-01'

        db.session.remove()
        db.drop_all()
