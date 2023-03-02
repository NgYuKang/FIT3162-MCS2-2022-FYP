
from app import app
import pytest
import io

import sys
import os
os.path.join("..")

@pytest.fixture()
def client():
    app.testing = True
    return app.test_client()


@pytest.fixture()
def runner():
    return app.test_cli_runner()


def test_main_page(client):
    response = client.get("/")
    assert b"Upload your photo here and get it classified as either sports or non sports!" in response.data


def test_valid_image(client):
    response = client.post("/", data={
        "image": open("test/random.JPG", "rb")
    }, follow_redirects=True)
    res = response.data.decode()
    assert "Result: " in res


def test_invalid_file(client):
    response = client.post("/", data={
        "image": open("README.md", "rb")
    }, follow_redirects=True)
    res = response.data.decode()
    assert "Invalid file type was uploaded!" in res


def test_empty_file(client):
    response = client.post("/", data={
        "image": (io.BytesIO(b"this is a test"), '')
    }, follow_redirects=True)
    res = response.data.decode()
    assert "No image was uploaded!" in res


def test_corrupt_image(client):
    response = client.post("/", data={
        "image": open("test/corrupt.JPG", "rb")
    }, follow_redirects=True)
    res = response.data.decode()
    assert "Unexpected error has occurred. It may be due to the image being corrupted." in res

# if __name__ == '__main__':
#     test_main_page(client())
