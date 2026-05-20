import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from src.config import DEFAULT_CONFIG
from src.face.recognizer import DetectedFace
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
                "registered_face_dir": "data/test_registered_faces",
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

    def test_face_image_upload_missing_person_returns_404(self):
        response = self.client.post(
            "/api/persons/MISSING/face-images",
            data={"angle": "front"},
            files={"image": ("face.jpg", b"fake-image", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 404)

    def test_face_image_upload_success_writes_embedding_metadata(self):
        person_payload = {
            "person_id": "STU2",
            "name": "Bob",
            "role": "student",
            "department": "Test",
            "status": "active",
        }
        self.client.post("/api/persons", json=person_payload)

        class FakeRecognizer:
            def __init__(self, *args, **kwargs):
                pass

            def extract_from_image(self, image_path):
                self.last_image_path = Path(image_path)
                return DetectedFace(
                    bbox=(0, 0, 100, 100),
                    embedding=[1.0, 0.0],
                    quality_score=0.98,
                )

        with patch("src.admin.app.InsightFaceRecognizer", FakeRecognizer):
            response = self.client.post(
                "/api/persons/STU2/face-images",
                data={"angle": "front"},
                files={"image": ("face.jpg", b"fake-image", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["person_id"], "STU2")
        self.assertEqual(body["angle"], "front")
        self.assertEqual(body["quality_score"], 0.98)

        embeddings_response = self.client.get("/api/persons/STU2/embeddings")
        self.assertEqual(embeddings_response.status_code, 200)
        embeddings = embeddings_response.json()
        self.assertEqual(len(embeddings), 1)
        self.assertNotIn("embedding", embeddings[0])
        self.assertEqual(embeddings[0]["angle"], "front")


if __name__ == "__main__":
    unittest.main()
