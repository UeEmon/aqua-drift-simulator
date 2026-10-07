from fastapi.testclient import TestClient

from aqua_drift.api import app


def test_health_and_configuration_round_trip() -> None:
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

        config = client.get("/api/config").json()
        config["max_slant_range_yd"] = 9000.0
        config["source"]["shared_recognition_bias_hz"] = 1.25
        config["estimator"]["particle_count"] = 2000
        updated = client.put("/api/config", json=config)

        assert updated.status_code == 200
        assert updated.json()["max_slant_range_yd"] == 9000.0
        assert updated.json()["source"]["shared_recognition_bias_hz"] == 1.25


def test_estimator_feed_contains_no_truth() -> None:
    with TestClient(app) as client:
        client.post("/internal/clock", json={"tick": 5})
        position = {"latitude": 35.0, "longitude": 140.0, "depth_ft": 100.0}
        client.post(
            "/internal/observer",
            json={"observer_id": "feed-test", "tick": 5, "position": position},
        )
        batch = {
            "tick": 5,
            "observations": [
                {
                    "observer_id": "feed-test",
                    "tick": 5,
                    "observer_position": position,
                    "detected": True,
                    "observed_frequency_hz": 400.1,
                    "recognized_frequency_hz": 400.0,
                }
            ],
            "truth": [
                {
                    "observer_id": "feed-test",
                    "tick": 5,
                    "slant_range_yd": 1234.0,
                    "relative_speed_kt": 3.0,
                    "relative_radial_speed_kt": 1.0,
                }
            ],
        }
        assert client.post("/internal/doppler", json=batch).status_code == 200
        feed = client.get("/internal/estimator-feed", params={"after_tick": 0}).json()
        assert feed["batches"][-1]["truth"] == []
        text = str(feed)
        assert "1234" not in text
        assert "source_frequency_hz" not in text
        assert "shared_recognition_bias_hz" not in text


def test_observer_placement_queue() -> None:
    with TestClient(app) as client:
        placement = {"position": {"latitude": 35.1, "longitude": 140.1, "depth_ft": 300.0}}
        # with the layer (設標者, default) an operator placement becomes an approved drop task
        assert client.post("/api/observers/placements", json=placement).status_code == 200
        tasks = client.get("/api/snapshot").json()["deployment"]["tasks"]
        assert tasks[-1]["status"] == "APPROVED" and tasks[-1]["source"] == "manual"
        # without the layer the next observer container takes the placement at once
        config = client.get("/api/config").json()
        config["layer"]["enabled"] = False
        assert client.put("/api/config", json=config).status_code == 200
        assert client.post("/api/observers/placements", json=placement).status_code == 200
        assigned = client.get(
            "/internal/observer/assignment", params={"observer_id": "placed-1"}
        ).json()
        assert assigned["latitude"] == 35.1
        assert assigned["depth_ft"] == 300.0
        config["layer"]["enabled"] = True
        assert client.put("/api/config", json=config).status_code == 200


def test_drop_reschedule_and_cancel_endpoints() -> None:
    with TestClient(app) as client:
        config = client.get("/api/config").json()
        assert config["layer"]["preferred_turn"] == "left" and config["layer"]["paused"] is False
        placement = {"position": {"latitude": 35.2, "longitude": 140.2, "depth_ft": 200.0}, "planned_tick": 99999}
        assert client.post("/api/observers/placements", json=placement).status_code == 200
        task = client.get("/api/snapshot").json()["deployment"]["tasks"][-1]
        assert task["planned_tick"] == 99999
        moved = client.post("/api/drops/reschedule", json={"task_id": task["task_id"], "planned_tick": None})
        assert moved.status_code == 200 and moved.json()["planned_tick"] is None
        assert client.post("/api/drops/reschedule", json={"task_id": 123456, "planned_tick": 5}).status_code == 404
        cancelled = client.post("/api/drops/cancel", json={"task_ids": [task["task_id"]]}).json()
        assert [t["status"] for t in cancelled] == ["CANCELLED"]
        config["layer"]["preferred_turn"] = "right"
        config["layer"]["paused"] = True
        saved = client.put("/api/config", json=config).json()
        assert saved["layer"]["preferred_turn"] == "right" and saved["layer"]["paused"] is True
        config["layer"]["preferred_turn"] = "left"
        config["layer"]["paused"] = False
        assert client.put("/api/config", json=config).status_code == 200


def test_estimation_control_endpoints() -> None:
    with TestClient(app) as client:
        stopped = client.post("/api/estimation/stop").json()
        assert stopped["running"] is False
        assert client.get("/internal/estimator-feed").json()["estimation"]["running"] is False
        started = client.post("/api/estimation/start").json()
        assert started["running"] is True
        assert started["run_id"] > stopped["run_id"] - 1
        assert client.get("/api/snapshot").json()["estimation"]["running"] is True
        assert client.post("/api/reset", params={"replace_observers": "false"}).status_code == 200


def test_standby_assignment_and_deploy_now_without_estimate() -> None:
    with TestClient(app) as client:
        statuses = [
            client.get("/internal/observer/assignment", params={"observer_id": f"sb-{i}"}).status_code
            for i in range(8)
        ]
        assert 204 in statuses  # beyond deployment.initial_count -> standby
        client.post("/api/estimation/stop")
        client.post("/api/estimation/start")  # clears estimates
        assert client.post("/api/deployment/now").status_code == 409
        feed = client.get("/internal/deployment-feed").json()
        assert "target" not in feed and "truth" not in str(feed).lower()


def test_websocket_stream_protocol_v2() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            message = ws.receive_json()
        assert message["v"] == 2
        for key in ("t", "gen", "ctl", "obs", "est", "cfg"):
            assert key in message  # first message of a connection is complete
