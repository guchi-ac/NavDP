import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from utils_tasks.client_utils import navigator_close


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_navdp_server():
    fake_policy_agent = types.ModuleType("policy_agent")
    fake_policy_agent.NavDP_Agent = object
    server_path = REPO_ROOT / "baselines" / "navdp" / "navdp_server.py"
    spec = importlib.util.spec_from_file_location(
        "navdp_server_close_test",
        server_path,
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"policy_agent": fake_policy_agent}):
        spec.loader.exec_module(module)
    return module


class NavigatorCloseServerTests(unittest.TestCase):
    def test_close_endpoint_is_idempotent_and_clears_writer(self):
        server = load_navdp_server()
        writer = mock.Mock()
        server.navdp_fps_writer = writer
        client = server.app.test_client()

        first = client.post("/navigator_close")
        second = client.post("/navigator_close")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.get_json(), {"algo": "navdp", "status": "closed"})
        writer.close.assert_called_once_with()
        self.assertIsNone(server.navdp_fps_writer)


class NavigatorCloseClientTests(unittest.TestCase):
    @mock.patch("utils_tasks.client_utils.requests.post")
    def test_client_posts_close_with_a_bounded_timeout(self, post):
        post.return_value.json.return_value = {
            "algo": "navdp",
            "status": "closed",
        }

        result = navigator_close(port=9001, timeout=2.5)

        post.assert_called_once_with(
            "http://localhost:9001/navigator_close",
            timeout=2.5,
        )
        post.return_value.raise_for_status.assert_called_once_with()
        self.assertEqual(result["status"], "closed")

    def test_ros_client_closes_server_after_stopping_workers(self):
        source = (
            REPO_ROOT
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        ).read_text(encoding="utf-8")

        self.assertIn("navigator_close", source)
        self.assertIn("navigator_close(port=self.args.server_port)", source)
        self.assertIn("SignalHandlerOptions.NO", source)
        self.assertIn("except KeyboardInterrupt:", source)
        self.assertIn("if rclpy.ok():", source)
        self.assertLess(
            source.index("self.visualization_thread.join"),
            source.index("navigator_close(port=self.args.server_port)"),
        )


if __name__ == "__main__":
    unittest.main()
