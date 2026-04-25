import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import pilot_security
from pilot_security import (
    authenticate_user,
    create_user,
    hash_password,
    init_pilot_database,
    list_audit_logs,
    save_supplier_upload,
    verify_password,
)


class PilotSecurityTests(unittest.TestCase):
    def test_password_hash_verification(self):
        stored = hash_password("correct horse battery staple")

        self.assertTrue(verify_password("correct horse battery staple", stored))
        self.assertFalse(verify_password("wrong password", stored))

    def test_user_creation_authentication_and_upload_audit(self):
        test_db = Path("tests/.tmp_pilot_security.db")
        for suffix in ["", "-wal", "-shm"]:
            path = Path(f"{test_db}{suffix}")
            if path.exists():
                path.unlink()
        try:
            with patch.object(pilot_security, "DB_PATH", test_db):
                init_pilot_database()
                admin = authenticate_user("admin", "ChangeMe123!")
                self.assertIsNotNone(admin)
                self.assertEqual(admin["role"], "admin")

                create_user("analyst1", "StrongPass123", "analyst", admin)
                analyst = authenticate_user("analyst1", "StrongPass123")
                self.assertIsNotNone(analyst)
                self.assertEqual(analyst["role"], "analyst")

                upload_id = save_supplier_upload(
                    "analyst1",
                    "suppliers.csv",
                    pd.DataFrame([{"supplier_name": "Apex", "annual_spend": 1000.0}]),
                )

                self.assertGreater(upload_id, 0)
                actions = {row["action"] for row in list_audit_logs()}
                self.assertIn("system.seed_admin", actions)
                self.assertIn("admin.create_user", actions)
        finally:
            for suffix in ["", "-wal", "-shm"]:
                path = Path(f"{test_db}{suffix}")
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    unittest.main()
