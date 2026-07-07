import os
import json
import pytest
from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


def test_fetch_via_chatgpt_fallback(client):
    # This test exercises /fetch_via_chatgpt when OPENAI_API_KEY is not set.
    # It should return at least one suggestion (the local planner fallback).

    payload = {
        'query': 'daily weather for my farm',
        'lat': 12.9716,
        'lon': 77.5946,
        'start': '20230101',
        'end': '20230131'
    }

    rv = client.post('/fetch_via_chatgpt', data=json.dumps(payload), content_type='application/json')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['success'] is True
    assert isinstance(data.get('suggestions'), list)
    assert len(data['suggestions']) >= 1
    s = data['suggestions'][0]
    assert 'source' in s and s['source'] == 'nasa_power'
    assert 'params' in s
    assert 'lat' in s['params'] and 'lon' in s['params']
