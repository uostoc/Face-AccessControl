import importlib.util
import unittest

from src.config import DEFAULT_CONFIG
from src.storage.database import Database


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class AdminApiTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from src.admin.app import create_app

        self.db = Database(":memory:")
        config = {
            **DEFAULT_CONFIG,
            "storage": {
                **DEFAULT_CONFIG["storage"],
                "database_path": ":memory:",
            },
        }
        app = create_app(config)
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_person_create_and_list(self):
        payload = {
            "person_id": "STU1",
            "name": "Alice",
            "role": "student",
            "department": "Test",
            "status": "active",
        }
        create_response = self.client.post("/api/persons", json=payload)
        self.assertEqual(create_response.status_code, 200)

        list_response = self.client.get("/api/persons")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()[0]["person_id"], "STU1")

    def test_security_event_review(self):
        self.client.app.state.db.add_security_event(
            event_type="stranger",
            camera_id="camera-1",
            location="gate",
            confidence=0.2,
        )

        events_response = self.client.get("/api/security-events")
        event_id = events_response.json()[0]["id"]
        response = self.client.put(
            f"/api/security-events/{event_id}/review",
            json={"review_status": "confirmed", "reviewer": "admin", "review_comment": "ok"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["review_status"], "confirmed")


if __name__ == "__main__":
    unittest.main()
