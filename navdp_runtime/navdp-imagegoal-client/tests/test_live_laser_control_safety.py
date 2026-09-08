import unittest

from utils_tasks.live_laser_trajectory import live_test_max_v
from utils_tasks.wheeled_client_core import control_stop_reason


class LiveLaserControlSafetyTests(unittest.TestCase):
    def base_stop_inputs(self):
        return {
            "now": 10.0,
            "enable_control": True,
            "arrival_blocked": False,
            "trajectory_ready": True,
            "last_frame_time": 9.9,
            "last_odom_time": 9.9,
            "last_plan_time": 9.9,
            "last_scan_time": 9.9,
            "frame_timeout": 1.0,
            "odom_timeout": 0.5,
            "plan_timeout": 1.0,
            "scan_timeout": 0.25,
        }

    def test_fresh_scan_allows_control(self):
        self.assertIsNone(control_stop_reason(**self.base_stop_inputs()))

    def test_missing_or_stale_scan_stops_control(self):
        missing = self.base_stop_inputs()
        missing["last_scan_time"] = None
        self.assertEqual(control_stop_reason(**missing), "scan_missing")

        stale = self.base_stop_inputs()
        stale["last_scan_time"] = 9.7
        self.assertEqual(control_stop_reason(**stale), "scan_stale")

    def test_existing_callers_can_omit_scan_gate(self):
        values = self.base_stop_inputs()
        values.pop("last_scan_time")
        values.pop("scan_timeout")
        self.assertIsNone(control_stop_reason(**values))

    def test_live_test_speed_is_hard_capped_at_point_one(self):
        self.assertEqual(live_test_max_v(0.2), 0.1)
        self.assertEqual(live_test_max_v(0.08), 0.08)

    def test_live_test_speed_rejects_invalid_values(self):
        for requested in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    live_test_max_v(requested)


if __name__ == "__main__":
    unittest.main()
