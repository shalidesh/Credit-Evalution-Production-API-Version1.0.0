
import requests


ENDPOINT = "http://127.0.0.1:5000"


def test_check_main_route():
    response=requests.get(ENDPOINT)
    assert response.status_code == 200


def test_check_suzuki():

    payload={
        "yom":"2018",
        "manufacture":"suzuki",
        "model":"WAGON R STINGRAY SAFETY",
        "engine_capacity":650,
        "fuel":"HYBRID",
        "transmission":"AUTOMATIC"
    }
    response=requests.post(f"{ENDPOINT}/predict",json=payload)
    assert response.status_code == 200
    assert 'results' in response.json() and response.json()['results'] is not None

def test_check_toyota():

    payload={
        "yom":"2018",
        "manufacture":"toyota",
        "model":"PREMIO",
        "engine_capacity":1500,
        "fuel":"PETROL",
        "transmission":"AUTOMATIC"
    }
    response=requests.post(f"{ENDPOINT}/predict",json=payload)
    assert response.status_code == 200
    assert 'results' in response.json() and response.json()['results'] is not None


def test_check_nissan():

    payload={
        "yom":"2018",
        "manufacture":"nissan",
        "model":"Sunny",
        "engine_capacity":1295,
        "fuel":"Petrol",
        "transmission":"Automatic"
    }
    response=requests.post(f"{ENDPOINT}/predict",json=payload)
    assert response.status_code == 200
    assert 'results' in response.json() and response.json()['results'] is not None

def test_check_honda():

    payload={
        "yom":"2018",
        "manufacture":"honda",
        "model":"Civic",
        "engine_capacity":1500,
        "fuel":"Petrol",
        "transmission":"Automatic"
    }
    response=requests.post(f"{ENDPOINT}/predict",json=payload)
    assert response.status_code == 200
    assert 'results' in response.json() and response.json()['results'] is not None



