import ast
import unittest
from pathlib import Path


class LiveLaserClientWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def method_source(self, name):
        method = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return ast.get_source_segment(self.source, method)

    def test_planning_densifies_then_adjusts_with_latest_scan(self):
        planning = self.method_source("_planning_loop")

        dense_at = planning.index("Mpc_controller.make_ref_denser(")
        adjust_at = planning.index("prepare_live_laser_trajectory(")
        self.assertLess(dense_at, adjust_at)
        self.assertIn("latest_scan = self.latest_scan", planning)
        self.assertIn("scan=latest_scan", planning)
        self.assertIn("laser_config=self.laser_map_config", planning)
        self.assertIn("if critic_safe and laser_result.safe:", planning)

    def test_mpc_receives_adjusted_dense_reference_without_resampling(self):
        planning = self.method_source("_planning_loop")

        self.assertIn("ref_traj_is_dense=True", planning)
        self.assertIn("self.mpc.update_dense_ref_traj(active_traj)", planning)
        self.assertNotIn("self.mpc.update_ref_traj(active_traj)", planning)

    def test_control_requires_fresh_scan_and_uses_hard_speed_cap(self):
        initializer = self.method_source("__init__")
        control = self.method_source("_control_loop")
        publisher = self.method_source("_publish_velocity")

        self.assertIn(
            "self.control_max_v = live_test_max_v(args.max_v)",
            initializer,
        )
        self.assertIn("last_scan_time=", control)
        self.assertIn("scan_timeout=self.args.scan_timeout", control)
        self.assertIn("self.control_max_v", control)
        self.assertNotIn(
            "np.clip(controls[0, 0], 0.0, self.args.max_v)",
            control,
        )
        self.assertIn(
            "np.clip(linear, 0.0, self.control_max_v)",
            publisher,
        )

    def test_plan_diagnostics_record_adjustment_result(self):
        planning = self.method_source("_planning_loop")

        for field in (
            '"laser_adjustment_reason"',
            '"laser_adjustment_side"',
            '"laser_clearance_before_m"',
            '"laser_clearance_after_m"',
            '"laser_max_offset_m"',
            '"adjusted_local_xy"',
        ):
            self.assertIn(field, planning)


if __name__ == "__main__":
    unittest.main()
