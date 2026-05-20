import unittest

from src.storage.database import Database, Person


class DatabaseTest(unittest.TestCase):
    def test_person_and_embedding_round_trip(self):
        db = Database(":memory:")
        db.init_schema()
        db.upsert_person(Person(person_id="STU1", name="Alice"))
        db.add_embedding("STU1", "alice.jpg", [1.0, 0.0], "front", 0.99, "test")

        records = db.load_active_embeddings()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].person_id, "STU1")
        self.assertEqual(records[0].person_name, "Alice")
        self.assertEqual(records[0].embedding, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
