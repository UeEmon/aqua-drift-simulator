from fastapi.testclient import TestClient

from aqua_drift.api import app


def test_health_and_configuration_round_trip() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        config = client.get("/api/config").json()
        config["max_slant_range_yd"] = 9000.0
        config["source"]["shared_recognition_bias_hz"] = 1.25
        updated = client.put("/api/config", json=config)

        assert updated.status_code == 200
        assert updated.json()["max_slant_range_yd"] == 9000.0
        assert updated.json()["source"]["shared_recognition_bias_hz"] == 1.25
