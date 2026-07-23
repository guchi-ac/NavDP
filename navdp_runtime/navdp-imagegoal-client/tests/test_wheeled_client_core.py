import ast
import json
import math
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from utils_tasks import visualization_utils
from utils_tasks import wheeled_client_core as client_core
from utils_tasks.wheeled_client_core import (
    control_stop_reason,
    trajectory_to_world,
    yaw_from_quaternion,
)


class GeometryTests(unittest.TestCase):
    def test_pitch20_d435_mount_points_rgb_optical_axis_down_20_degrees(self):
        base_from_mount = client_core.transform_matrix_from_translation_quaternion(
            translation_xyz=np.array([0.092070325, 0.0, 1.254818867]),
            rotation_xyzw=np.array([0.0, 0.171753592, 0.0, 0.985139941]),
        )

        rgb_optical_forward_at_zero = np.array(
            [0.9999843357625597, 0.004062782406564658, -0.003849938782610085]
        )
        forward = base_from_mount[:3, :3] @ rgb_optical_forward_at_zero
        down_angle = math.degrees(
            math.atan2(-forward[2], np.linalg.norm(forward[:2]))
        )

        self.assertAlmostEqual(down_angle, 20.0, places=5)

    def test_extracts_yaw_from_quaternion(self):
        angle = math.radians(70.0)

        yaw = yaw_from_quaternion(0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))

        self.assertAlmostEqual(yaw, angle)

    def test_transforms_camera_aligned_trajectory_into_odom_frame(self):
        local = np.array([[1.0, 0.0], [0.0, 1.0]])

        world = trajectory_to_world(
            local,
            odom_xy_yaw=np.array([2.0, 3.0, math.pi / 2]),
            camera_x=1.0,
            camera_y=0.0,
            camera_yaw=0.0,
        )

        np.testing.assert_allclose(world, [[2.0, 5.0], [1.0, 4.0]], atol=1e-7)

    def test_extracts_planar_camera_pose_from_optical_forward_axis(self):
        self.assertTrue(hasattr(client_core, "camera_pose_from_transform"))
        if not hasattr(client_core, "camera_pose_from_transform"):
            return

        pose = client_core.camera_pose_from_transform(
            translation_xyz=np.array([0.045, -0.011, 1.313]),
            # ROS optical +Z points along base +X; quaternion yaw itself is -90 deg.
            rotation_xyzw=np.array([0.5, -0.5, 0.5, -0.5]),
        )

        np.testing.assert_allclose(pose, [0.045, -0.011, 0.0], atol=1e-7)

    def test_builds_rigid_transform_from_ros_quaternion(self):
        angle = math.pi / 2

        transform = client_core.transform_matrix_from_translation_quaternion(
            translation_xyz=np.array([1.0, 2.0, 3.0]),
            rotation_xyzw=np.array(
                [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
            ),
        )

        np.testing.assert_allclose(
            transform @ np.array([1.0, 0.0, 0.0, 1.0]),
            [1.0, 3.0, 3.0, 1.0],
            atol=1e-7,
        )


class TrajectoryManagerTests(unittest.TestCase):
    def make_manager(self, **changes):
        values = dict(
            point_spacing=0.05,
            join_distance=0.50,
            join_heading_degrees=60.0,
            min_remaining=0.20,
        )
        values.update(changes)
        return client_core.TrajectoryManager(**values)

    def test_initial_candidate_starts_at_chassis_and_fills_near_field(self):
        manager = self.make_manager()

        result = manager.update(
            np.array([2.0, 3.0]),
            np.array([[3.0, 3.0], [4.0, 3.0]]),
            candidate_eligible=True,
        )

        self.assertTrue(result.candidate_accepted)
        self.assertEqual(result.reason, "initialized")
        np.testing.assert_array_equal(result.active_traj[0], [2.0, 3.0])
        distances = np.linalg.norm(np.diff(result.active_traj, axis=0), axis=1)
        np.testing.assert_allclose(distances, 0.05, atol=1e-9)

    def test_advance_prunes_traversed_history_and_reanchors_at_chassis(self):
        manager = self.make_manager()
        manager.update(
            [0.0, 0.0],
            np.array([[1.0, 0.0], [2.0, 0.0]]),
            candidate_eligible=True,
        )

        result = manager.update([0.35, 0.02])

        np.testing.assert_array_equal(result.active_traj[0], [0.35, 0.02])
        self.assertGreater(result.active_traj[-1, 0], 1.9)

    def test_join_preserves_history_before_candidate(self):
        manager = self.make_manager()
        manager.update(
            [0.0, 0.0],
            np.array([[1.0, 0.0], [2.0, 0.0]]),
            candidate_eligible=True,
        )

        result = manager.update(
            [0.10, 0.0],
            np.array([[1.05, 0.02], [2.0, 0.5]]),
            candidate_eligible=True,
        )

        self.assertTrue(result.candidate_accepted)
        np.testing.assert_array_equal(result.active_traj[0], [0.10, 0.0])
        self.assertTrue(
            np.any(np.isclose(result.active_traj[:, 0], 0.50, atol=0.03))
        )

    def test_rejects_distant_candidate_without_mutating_history(self):
        manager = self.make_manager(join_distance=0.10)
        manager.update(
            [0.0, 0.0],
            np.array([[1.0, 0.0], [2.0, 0.0]]),
            candidate_eligible=True,
        )

        result = manager.update(
            [0.10, 0.0],
            np.array([[1.0, 1.0], [2.0, 1.0]]),
            candidate_eligible=True,
        )

        self.assertFalse(result.candidate_accepted)
        self.assertEqual(result.reason, "join_distance")
        self.assertLess(np.max(np.abs(result.active_traj[:, 1])), 0.11)

    def test_rejects_heading_discontinuity(self):
        manager = self.make_manager(join_heading_degrees=30.0)
        manager.update(
            [0.0, 0.0],
            np.array([[1.0, 0.0], [2.0, 0.0]]),
            candidate_eligible=True,
        )

        result = manager.update(
            [0.10, 0.0],
            np.array([[1.0, 0.0], [1.0, 1.0]]),
            candidate_eligible=True,
        )

        self.assertFalse(result.candidate_accepted)
        self.assertEqual(result.reason, "join_heading")

    def test_low_critic_candidate_retains_history_until_exhausted(self):
        manager = self.make_manager(min_remaining=0.20)
        manager.update(
            [0.0, 0.0],
            np.array([[0.5, 0.0], [1.0, 0.0]]),
            candidate_eligible=True,
        )

        retained = manager.update(
            [0.50, 0.0],
            np.array([[0.5, 1.0], [1.0, 1.0]]),
            candidate_eligible=False,
        )
        exhausted = manager.update([0.95, 0.0])

        self.assertEqual(retained.reason, "candidate_low_critic")
        self.assertIsNotNone(retained.active_traj)
        self.assertIsNone(exhausted.active_traj)
        self.assertEqual(exhausted.reason, "history_exhausted")

    def test_invalid_candidate_retains_history(self):
        manager = self.make_manager()
        manager.update(
            [0.0, 0.0],
            np.array([[0.5, 0.0], [1.0, 0.0]]),
            candidate_eligible=True,
        )

        result = manager.update(
            [0.10, 0.0],
            np.array([[math.nan, 0.0], [1.0, 0.0]]),
            candidate_eligible=True,
        )

        self.assertEqual(result.reason, "candidate_invalid")
        self.assertIsNotNone(result.active_traj)

    def test_result_does_not_allow_mutating_manager_state(self):
        manager = self.make_manager()
        first = manager.update(
            [0.0, 0.0],
            np.array([[0.5, 0.0], [1.0, 0.0]]),
            candidate_eligible=True,
        )

        with self.assertRaises(ValueError):
            first.active_traj[0] = [99.0, 99.0]
        second = manager.update([0.10, 0.0])

        self.assertLess(second.active_traj[0, 0], 1.0)


class PostureGoalTests(unittest.TestCase):
    def test_configures_mira3_navdp_posture_without_moving_torso_height(self):
        self.assertTrue(hasattr(client_core, "configure_navdp_posture_goal"))

        class Goal:
            pass

        goal = Goal()
        configured = client_core.configure_navdp_posture_goal(goal)

        self.assertIs(configured, goal)
        self.assertEqual(goal.work_mode, 0)
        self.assertEqual(goal.input_mode, 0)
        self.assertEqual(goal.torso_roll, 0.0)
        self.assertEqual(goal.torso_height, 0.0)
        self.assertEqual(goal.torso_yaw, 0.0)
        self.assertEqual(goal.head_yaw, 0.0)
        self.assertAlmostEqual(goal.head_pitch, -19.7795845)
        self.assertEqual(goal.torso_mask, [False, True])
        self.assertEqual(goal.head_mask, [True, True])
        self.assertEqual(goal.max_velocity, 0.1)


class PostureActionRunnerTests(unittest.TestCase):
    class Future:
        def __init__(self, value=None, done=True):
            self.value = value
            self.is_done = done

        def done(self):
            return self.is_done

        def result(self):
            return self.value

    class GoalHandle:
        def __init__(self, result_future, accepted=True, cancel_future=None):
            self.accepted = accepted
            self.result_future = result_future
            self.cancel_future = cancel_future
            self.cancel_calls = 0

        def get_result_async(self):
            return self.result_future

        def cancel_goal_async(self):
            self.cancel_calls += 1
            return self.cancel_future

    class ActionClient:
        def __init__(self, goal_future, server_ready=True):
            self.goal_future = goal_future
            self.server_ready = server_ready
            self.sent_goals = []

        def wait_for_server(self, timeout_sec):
            self.server_timeout = timeout_sec
            return self.server_ready

        def send_goal_async(self, goal):
            self.sent_goals.append(goal)
            return self.goal_future

    @staticmethod
    def result_response(status=4, success=True, message="ok"):
        result = type("Result", (), {"success": success, "msg": message})()
        return type("Response", (), {"status": status, "result": result})()

    @staticmethod
    def cancellation_response(accepted=True):
        goals = [object()] if accepted else []
        return type("CancelResponse", (), {"goals_canceling": goals})()

    def make_runner(self, client, spin=None):
        self.assertTrue(hasattr(client_core, "PostureActionRunner"))
        if spin is None:
            spin = lambda node, future, timeout_sec: None
        return client_core.PostureActionRunner(
            node=object(),
            action_client=client,
            timeout=10.0,
            spin_until_future_complete=spin,
            goal_factory=lambda: type("Goal", (), {})(),
            succeeded_status=4,
        )

    def test_waits_for_successful_posture_result(self):
        result_future = self.Future(self.result_response())
        handle = self.GoalHandle(result_future)
        client = self.ActionClient(self.Future(handle))
        runner = self.make_runner(client)

        result = runner.run()

        self.assertTrue(result.success)
        self.assertIsNone(runner.goal_handle)
        self.assertEqual(client.server_timeout, 10.0)
        self.assertEqual(len(client.sent_goals), 1)
        self.assertEqual(client.sent_goals[0].torso_mask, [False, True])

    def test_rejects_all_fail_closed_action_outcomes(self):
        cases = (
            (
                self.ActionClient(self.Future(), server_ready=False),
                "server unavailable",
            ),
            (self.ActionClient(self.Future(done=False)), "timed out sending"),
            (
                self.ActionClient(
                    self.Future(self.GoalHandle(self.Future(), accepted=False))
                ),
                "was rejected",
            ),
            (
                self.ActionClient(
                    self.Future(
                        self.GoalHandle(
                            self.Future(self.result_response(status=6))
                        )
                    )
                ),
                "did not succeed",
            ),
            (
                self.ActionClient(
                    self.Future(
                        self.GoalHandle(
                            self.Future(
                                self.result_response(success=False, message="fault")
                            )
                        )
                    )
                ),
                "reported failure: fault",
            ),
        )
        for client, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.make_runner(client).run()

    def test_result_timeout_waits_for_accepted_cancellation(self):
        cancel_future = self.Future(self.cancellation_response())
        handle = self.GoalHandle(
            self.Future(done=False),
            cancel_future=cancel_future,
        )
        runner = self.make_runner(self.ActionClient(self.Future(handle)))

        with self.assertRaisesRegex(RuntimeError, "timed out waiting"):
            runner.run()

        self.assertEqual(handle.cancel_calls, 1)
        self.assertIsNone(runner.goal_handle)

    def test_interrupt_leaves_goal_available_for_bounded_cleanup(self):
        result_future = self.Future()
        cancel_future = self.Future(self.cancellation_response())
        handle = self.GoalHandle(result_future, cancel_future=cancel_future)

        def interrupt_result(node, future, timeout_sec):
            if future is result_future:
                raise KeyboardInterrupt

        runner = self.make_runner(
            self.ActionClient(self.Future(handle)),
            spin=interrupt_result,
        )
        with self.assertRaises(KeyboardInterrupt):
            runner.run()

        self.assertIs(runner.goal_handle, handle)
        runner.spin_until_future_complete = lambda node, future, timeout_sec: None
        runner.cancel()
        self.assertEqual(handle.cancel_calls, 1)
        self.assertIsNone(runner.goal_handle)


class StartupSequenceTests(unittest.TestCase):
    def run_sequence(self, enable_control, align_posture):
        self.assertTrue(hasattr(client_core, "run_navdp_startup"))
        events = []
        client_core.run_navdp_startup(
            enable_control=enable_control,
            align_posture=lambda: (events.append("posture"), align_posture())[1],
            publish_camera_tf=lambda: events.append("tf"),
            start_local_nav=lambda: events.append("local_nav"),
            start_workers=lambda: events.append("workers"),
        )
        return events

    def test_enabled_startup_orders_posture_before_all_navigation_work(self):
        events = self.run_sequence(True, lambda: None)

        self.assertEqual(events, ["posture", "tf", "local_nav", "workers"])

    def test_dry_run_skips_posture(self):
        events = self.run_sequence(False, lambda: None)

        self.assertEqual(events, ["tf", "local_nav", "workers"])

    def test_posture_failure_prevents_tf_and_navigation_startup(self):
        self.assertTrue(hasattr(client_core, "run_navdp_startup"))
        events = []

        def fail_posture():
            events.append("posture")
            raise RuntimeError("posture failed")

        with self.assertRaisesRegex(RuntimeError, "posture failed"):
            client_core.run_navdp_startup(
                enable_control=True,
                align_posture=fail_posture,
                publish_camera_tf=lambda: events.append("tf"),
                start_local_nav=lambda: events.append("local_nav"),
                start_workers=lambda: events.append("workers"),
            )

        self.assertEqual(events, ["posture"])


class DiagnosticsWriterTests(unittest.TestCase):
    def test_jsonl_writer_flushes_numpy_values_and_closes_idempotently(self):
        self.assertTrue(hasattr(client_core, "JsonlWriter"))
        if not hasattr(client_core, "JsonlWriter"):
            return

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mpc.jsonl"
            writer = client_core.JsonlWriter(output)
            writer.write(
                {
                    "type": "control",
                    "odom": np.array([1.0, 2.0, 0.5]),
                    "solve_ms": np.float64(4.25),
                }
            )

            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["type"], "control")
            self.assertEqual(record["odom"], [1.0, 2.0, 0.5])
            self.assertEqual(record["solve_ms"], 4.25)
            writer.close()
            writer.close()


class VideoFinalizationTests(unittest.TestCase):
    def test_transcodes_mp4v_to_h264_before_atomic_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "video.partial.mp4"
            output = Path(directory) / "video.mp4"
            temporary.write_bytes(b"mp4v")

            def fake_run(command, **kwargs):
                self.assertEqual(command[0], "ffmpeg")
                self.assertIn("libx264", command)
                self.assertIn("yuv420p", command)
                self.assertIn("+faststart", command)
                self.assertTrue(kwargs["start_new_session"])
                Path(command[-1]).write_bytes(b"h264")

            with mock.patch.object(client_core.subprocess, "run", fake_run):
                encoded = client_core.finalize_mp4(temporary, output)

            self.assertTrue(encoded)
            self.assertEqual(output.read_bytes(), b"h264")
            self.assertFalse(temporary.exists())
            self.assertFalse((Path(directory) / "video.h264.partial.mp4").exists())

    def test_retries_h264_transcode_three_times_with_backoff(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "video.partial.mp4"
            output = Path(directory) / "video.mp4"
            temporary.write_bytes(b"mp4v")

            attempts = []

            def fake_run(command, **kwargs):
                attempts.append(command)
                if len(attempts) < 3:
                    raise client_core.subprocess.CalledProcessError(
                        1,
                        command,
                        stderr=b"Resource temporarily unavailable",
                    )
                Path(command[-1]).write_bytes(b"h264")

            with mock.patch.object(
                client_core.subprocess,
                "run",
                fake_run,
            ), mock.patch.object(client_core.time, "sleep") as sleep:
                encoded = client_core.finalize_mp4(temporary, output)

            self.assertTrue(encoded)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_args_list, [mock.call(0.5), mock.call(0.5)])
            for command in attempts:
                self.assertEqual(command[command.index("-threads") + 1], "2")
            self.assertEqual(output.read_bytes(), b"h264")
            self.assertFalse(temporary.exists())

    def test_keeps_partial_and_reports_error_when_h264_transcode_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "video.partial.mp4"
            output = Path(directory) / "video.mp4"
            temporary.write_bytes(b"mp4v")

            error = client_core.subprocess.CalledProcessError(
                -9,
                ["ffmpeg"],
                stderr=b"",
            )
            with mock.patch.object(
                client_core.subprocess,
                "run",
                side_effect=error,
            ) as run, mock.patch.object(client_core.time, "sleep") as sleep:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "returncode=-9.*signal=9",
                ):
                    client_core.finalize_mp4(temporary, output)

            self.assertEqual(run.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertTrue(temporary.exists())
            self.assertFalse(output.exists())


class ControlDeadmanTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(
            now=10.0,
            enable_control=True,
            arrival_blocked=False,
            trajectory_ready=True,
            last_frame_time=9.9,
            last_odom_time=9.9,
            last_plan_time=9.9,
            frame_timeout=1.0,
            odom_timeout=0.5,
            plan_timeout=1.0,
        )

    def reason(self, **changes):
        values = {**self.base, **changes}
        return control_stop_reason(**values)

    def test_allows_fresh_safe_control(self):
        self.assertIsNone(self.reason())

    def test_stops_when_control_is_disabled(self):
        self.assertEqual(self.reason(enable_control=False), "control_disabled")

    def test_stops_for_arrival_before_other_checks(self):
        self.assertEqual(
            self.reason(arrival_blocked=True, trajectory_ready=False),
            "arrival",
        )

    def test_stops_when_active_trajectory_is_missing(self):
        self.assertEqual(
            self.reason(trajectory_ready=False),
            "trajectory_missing",
        )

    def test_stops_for_missing_or_stale_inputs(self):
        cases = (
            ("frame_missing", {"last_frame_time": None}),
            ("frame_stale", {"last_frame_time": 8.9}),
            ("odom_missing", {"last_odom_time": None}),
            ("odom_stale", {"last_odom_time": 9.4}),
            ("plan_missing", {"last_plan_time": None}),
            ("plan_stale", {"last_plan_time": 8.9}),
        )
        for expected, changes in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.reason(**changes), expected)


class VisualizationPipelineTests(unittest.TestCase):
    def test_latest_frame_replaces_stale_queued_frame(self):
        self.assertTrue(hasattr(client_core, "put_latest"))
        work_queue = queue.Queue(maxsize=1)

        client_core.put_latest(work_queue, "old")
        client_core.put_latest(work_queue, "new")

        self.assertEqual(work_queue.get_nowait(), "new")
        self.assertTrue(work_queue.empty())

    def test_display_resize_scales_rgbd_and_camera_intrinsic(self):
        self.assertTrue(
            hasattr(client_core, "resize_rgbd_for_visualization")
        )
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32)
        intrinsic = np.array(
            [
                [600.0, 0.0, 320.0],
                [0.0, 600.0, 240.0],
                [0.0, 0.0, 1.0],
            ]
        )

        resized_rgb, resized_depth, resized_intrinsic = (
            client_core.resize_rgbd_for_visualization(
                rgb,
                depth,
                intrinsic,
                target_width=320,
            )
        )

        self.assertEqual(resized_rgb.shape, (240, 320, 3))
        self.assertEqual(resized_depth.shape, (240, 320))
        np.testing.assert_allclose(
            resized_intrinsic,
            [
                [300.0, 0.0, 160.0],
                [0.0, 300.0, 120.0],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_visualization_can_skip_depth_occupancy(self):
        manager = visualization_utils.VisualizationManager(history_size=1)
        with mock.patch.object(
            manager,
            "build_occupancy_grid",
            side_effect=AssertionError("depth occupancy must not be built"),
        ):
            image = manager.visualize_trajectory(
                np.zeros((40, 80, 3), dtype=np.uint8),
                np.ones((40, 80, 1), dtype=np.float32),
                np.eye(3),
                np.array([[0.0, 0.0], [1.0, 0.0]]),
                robot_pose=np.zeros(3),
                show_depth=False,
            )

        self.assertEqual(image.shape, (40, 120, 3))

    def test_selected_trajectory_is_red(self):
        image = visualization_utils.VisualizationManager(
            history_size=1
        ).visualize_trajectory(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 1), dtype=np.float32),
            np.eye(3),
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            robot_pose=np.zeros(3),
            show_depth=False,
        )

        blue, green, red = image[40, 150]
        self.assertGreater(red, green * 2)
        self.assertGreater(red, blue * 2)

    def test_selected_trajectory_overlays_candidates_in_red(self):
        selected = np.array([[0.0, 0.0], [2.0, 0.0]])
        image = visualization_utils.VisualizationManager(
            history_size=1
        ).visualize_trajectory(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 1), dtype=np.float32),
            np.eye(3),
            selected,
            robot_pose=np.zeros(3),
            all_trajectories_points=np.array([selected]),
            all_trajectories_values=np.array([-1.2]),
            show_depth=False,
        )

        blue, green, red = image[140, 50]
        self.assertGreater(red, green * 2)
        self.assertGreater(red, blue * 2)

    def test_skipped_prefix_is_gray_before_red_tracked_trajectory(self):
        image = visualization_utils.VisualizationManager(
            history_size=1
        ).visualize_trajectory(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 1), dtype=np.float32),
            np.eye(3),
            np.array([[1.0, 0.0], [2.0, 0.0]]),
            robot_pose=np.zeros(3),
            trajectory_prefix_points=np.array(
                [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]
            ),
            show_depth=False,
        )

        prefix_region = image[41:50, 147:154]
        gray_pixels = (
            (prefix_region[:, :, 0] > 40)
            & (prefix_region[:, :, 0] < 200)
            & (np.abs(prefix_region[:, :, 0] - prefix_region[:, :, 1]) < 8)
            & (np.abs(prefix_region[:, :, 1] - prefix_region[:, :, 2]) < 8)
        )
        self.assertTrue(gray_pixels.any())

    def test_gray_prefix_dashes_continue_across_dense_waypoints(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        visualization_utils._draw_dashed_polyline(
            image,
            np.array(
                [[10.0, 50.0], [14.0, 50.0], [18.0, 50.0], [22.0, 50.0]]
            ),
        )

        self.assertGreater(image[50, 12, 0], 0)
        self.assertLess(image[50, 18, 0], image[50, 12, 0] * 0.75)


class RosClientSourceTests(unittest.TestCase):
    @staticmethod
    def client_source():
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        return client_path.read_text(encoding="utf-8")

    def test_client_has_real_robot_inputs_and_explicit_control_gate(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )

        source = client_path.read_text(encoding="utf-8")

        for required in (
            "/cam_head/d435/color/image_raw",
            "/cam_head/d435/aligned_depth_to_color/image_raw",
            "/cam_head/d435/color/camera_info",
            "/odom",
            "/cmd_vel",
            "TwistStamped",
            "--goal-image",
            "--enable-control",
            "imagegoal_step",
            "RgbdGoalVerifier",
            "Mpc_controller",
            "VisualizationManager",
            "/navdp/visualization",
            "latest_visualization.jpg",
            "all_trajectories",
            "Queue(maxsize=1)",
            "target=self._visualization_loop",
            'name="navdp_visualization"',
            "--visualization-width",
            "--visualization-fps",
            "--opencv-threads",
            "cv2.setNumThreads(args.opencv_threads)",
        ):
            self.assertIn(required, source)

    def test_enabled_client_owns_local_nav_action_lifecycle(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            "from interfaces.action import Skill",
            "from rclpy.action import ActionClient",
            "self.local_nav_action_client = ActionClient(",
            '"/skill_behavior_tree"',
            'goal.head.action_name = "local_nav"',
            "wait_for_server(",
            "send_goal_async(goal)",
            "cancel_goal_async()",
            'parser.add_argument("--cmd-topic", default="/cmd_vel")',
        ):
            self.assertIn(required, source)

    def test_enabled_client_completes_posture_before_tf_and_chassis_startup(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        initializer = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        initializer_source = ast.get_source_segment(source, initializer)
        starts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "start"
        ]
        self.assertEqual(len(starts), 1)
        start = starts[0]
        start_source = ast.get_source_segment(source, start)

        for required in (
            "from action_msgs.msg import GoalStatus",
            "from interfaces.action import Skill, Torso",
            "PostureActionRunner",
            "self.torso_action_client = None",
            "self.posture_action_runner = None",
            "if self.args.enable_control:",
            '"/Torso/torso_action_service"',
            "run_navdp_startup(",
            'parser.add_argument("--posture-timeout", type=float, default=10.0)',
        ):
            self.assertIn(required, source)

        for forbidden in (
            "self._set_navdp_posture()",
            "publish_camera_tf=self._publish_d435_mount_transform",
            "self._start_local_nav()",
            "self.control_thread.start()",
        ):
            self.assertNotIn(forbidden, initializer_source)
        for callback in (
            "align_posture=self._set_navdp_posture",
            "publish_camera_tf=self._publish_d435_mount_transform",
            "start_local_nav=self._start_local_nav",
            "start_workers=self._start_workers",
        ):
            self.assertIn(callback, start_source)
        main = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_source = ast.get_source_segment(source, main)
        self.assertLess(
            main_source.index("node = NavdpImageGoalClient(args)"),
            main_source.index("node.start()"),
        )

    def test_stop_cancels_inflight_posture_before_other_shutdown_work(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        stop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "stop"
        )
        stop_source = ast.get_source_segment(source, stop)

        self.assertIn("self._cancel_posture()", stop_source)
        self.assertLess(
            stop_source.index("self._cancel_posture()"),
            stop_source.index("self._cancel_local_nav()"),
        )
        self.assertIn("self.planning_thread.ident is not None", stop_source)

    def test_stop_cancels_local_nav_before_waiting_for_workers(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        stop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "stop"
        )
        stop_source = ast.get_source_segment(source, stop)

        self.assertLess(
            stop_source.index("self._cancel_local_nav()"),
            stop_source.index("self.planning_thread.join"),
        )

    def test_shutdown_handles_term_and_hup_and_masks_repeat_signals(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            "signal.SIGINT",
            "signal.SIGTERM",
            "signal.SIGHUP",
            "_install_graceful_shutdown_handlers()",
            "signal.signal(signum, signal.SIG_IGN)",
            "_stop_ignoring_shutdown_signals(node)",
        ):
            self.assertIn(required, source)

    def test_controller_copy_records_internnav_mit_provenance(self):
        controller_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "controllers.py"
        )

        source = controller_path.read_text(encoding="utf-8")

        self.assertIn("InternRobotics/InternNav", source)
        self.assertIn("MIT License", source)
        self.assertIn("class Mpc_controller", source)

    def test_controller_cost_tracks_reference_yaw(self):
        controller_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "controllers.py"
        )
        source = controller_path.read_text(encoding="utf-8")

        self.assertIn("Q = np.diag([10.0, 10.0, 5.0])", source)
        self.assertIn("reference_poses_from_xy(ref_traj, x0[2])", source)
        self.assertNotIn("np.zeros((ref_traj.shape[0], 1))", source)

    def test_client_uses_pose_only_for_kinematic_mpc(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        self.assertIn("self.mpc.solve(latest_odom)", source)
        self.assertNotIn("self.mpc.solve(latest_odom, latest_odom_twist)", source)

    def test_client_does_not_expose_removed_velocity_lag_parameters(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for removed in (
            "--mpc-linear-time-constant",
            "--mpc-angular-time-constant",
            "linear_time_constant=self.args.mpc_linear_time_constant",
            "angular_time_constant=self.args.mpc_angular_time_constant",
        ):
            self.assertNotIn(removed, source)

    def test_client_limits_openblas_before_numpy_is_imported(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        self.assertIn("OPENBLAS_NUM_THREADS", source)
        self.assertIn("NAVDP_BLAS_THREADS", source)
        self.assertLess(
            source.index("OPENBLAS_NUM_THREADS"),
            source.index("import numpy"),
        )

    def test_client_publishes_d435_mount_tf_and_consumes_optical_tf(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            "Buffer()",
            "StaticTransformBroadcaster",
            "TransformStamped",
            "publish_camera_tf=self._publish_d435_mount_transform",
            "sendTransform",
            'transform.header.frame_id = self.args.base_frame',
            'transform.child_frame_id = "d435_link"',
            "transform.transform.translation.x = 0.092070325",
            "transform.transform.translation.y = 0.0",
            "transform.transform.translation.z = 1.254818867",
            "transform.transform.rotation.x = 0.0",
            "transform.transform.rotation.y = 0.171753592",
            "transform.transform.rotation.z = 0.0",
            "transform.transform.rotation.w = 0.985139941",
            "TransformListener",
            "lookup_transform",
            'default="base_link"',
            'parser.add_argument("--camera-frame", default="d435_color_optical_frame")',
            "RosTime.from_msg(stamp)",
            "_camera_transform_in_base(rgb_message.header.stamp)",
            "camera_pose_from_transform",
            "transform_matrix_from_translation_quaternion",
            "MultiThreadedExecutor",
            "executor.spin()",
            "executor.shutdown()",
        ):
            self.assertIn(required, source)
        for removed in (
            "--camera-mount-frame",
            "--d435-base-frame",
            "apply_camera_pitch_sign",
            "--camera-pitch-sign",
            "corrected_transform",
            "--camera-x",
            "--camera-y",
            "--camera-yaw",
        ):
            self.assertNotIn(removed, source)

    def test_client_publishes_mpc_linear_velocity_without_hidden_scaling(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        self.assertIn("command.twist.linear.x = linear", source)
        self.assertNotIn("1.5*linear", source)

    def test_readme_documents_navdp_only_d435_tf_runtime(self):
        readme_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "README_NAVDP_CLIENT_ZH.md"
        )
        source = readme_path.read_text(encoding="utf-8")

        for required in (
            "USE_URDF_UTILS=0",
            "USE_REALSENSE_D435=1",
            "USE_REALSENSE_D455=0",
            "head_pitch: -19.7795845",
            "仅在使用 `--enable-control` 时自动整姿",
            "`torso_mask=[false, true]`",
            "`head_mask=[true, true]`",
            "--posture-timeout 10.0",
            "无需另起",
            "`static_transform_publisher`",
            "/home/dev/navdp_deployment/navdp_runtime/navdp-imagegoal-client",
            "--goal-image goal_far.jpg",
        ):
            self.assertIn(required, source)
        for removed in (
            "navdp_d435_tf",
            "--camera-mount-frame",
            "--d435-base-frame",
            "ros2 action send_goal /Torso/torso_action_service",
            "torso_mask: [false, false]",
            "/home/dev/navdp_runtime/navdp-imagegoal-client",
        ):
            self.assertNotIn(removed, source)

    def test_client_uses_navdp_trajectory_without_footprint_filtering(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        self.assertIn(
            "trajectory, all_trajectories, all_values = imagegoal_step(",
            source,
        )
        for removed in (
            "depth_to_obstacle_points",
            "select_safe_trajectory",
            "recheck_remaining_trajectory",
            "trajectory_clearance",
            "footprint_safe",
            "--initial-overlap-tolerance",
            "reusing previous full-safe plan",
            'reason = "footprint"',
            "critic count does not match trajectory count",
            "all NavDP critic values are invalid",
            "not np.isfinite(candidate_local).all()",
        ):
            self.assertNotIn(removed, source)

    def test_client_records_trajectory_visualization(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            "show_depth=False",
            "cv2.VideoWriter",
            "self.visualization_video_writer.release()",
            "--visualization-video-dir",
            "NavDP-official-bebb436/navdp_visualizations",
        ):
            self.assertIn(required, source)
        self.assertNotIn("all_trajectories_rejected", source)

    def test_client_visualizes_skipped_prefix_separately_from_mpc_trajectory(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            "raw_local_xy",
            "trajectory_prefix",
            "trajectory_prefix_points=state.trajectory_prefix",
        ):
            self.assertIn(required, source)

    def test_client_routes_model_candidate_through_trajectory_manager(self):
        source = self.client_source()

        self.assertIn("TrajectoryManager(", source)
        self.assertIn("candidate_eligible=critic_safe", source)
        self.assertIn("active_traj = trajectory_update.active_traj", source)
        self.assertIn("self.mpc.update_ref_traj(active_traj)", source)
        self.assertNotIn("self.mpc.update_ref_traj(world_xy)", source)

    def test_client_gates_control_on_active_trajectory(self):
        source = self.client_source()

        self.assertIn("self.trajectory_ready = False", source)
        self.assertIn("trajectory_ready=self.trajectory_ready", source)
        self.assertNotIn("critic_safe=self.critic_safe", source)

    def test_client_blocks_control_when_active_trajectory_install_fails(self):
        source = self.client_source()

        self.assertIn("failed to install active trajectory", source)
        self.assertIn('reason = "active_trajectory_error"', source)

    def test_client_exposes_trajectory_manager_defaults(self):
        source = self.client_source()

        for required in (
            '--trajectory-point-spacing", type=float, default=0.05',
            '--trajectory-join-distance", type=float, default=0.50',
            '--trajectory-join-heading-deg", type=float, default=60.0',
            '--trajectory-min-remaining", type=float, default=0.20',
        ):
            self.assertIn(required, source)

    def test_client_records_candidate_decision_and_active_trajectory(self):
        source = self.client_source()

        for required in (
            '"candidate_world_xy": retained_world_xy',
            '"active_traj": active_traj',
            '"candidate_accepted": trajectory_update.candidate_accepted',
            '"trajectory_reason": trajectory_update.reason',
            '"trajectory_join_distance_m"',
            "trajectory_update.join_distance_m",
            '"trajectory_remaining_length_m"',
            "trajectory_update.remaining_length_m",
        ):
            self.assertIn(required, source)

    def test_client_records_synchronized_mpc_diagnostics(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            "--mpc-log-dir",
            "latest_odom_twist",
            '"type": "plan"',
            '"type": "control"',
            '"desired_velocity": [linear, angular]',
            '"actual_velocity": latest_odom_twist',
            "predicted_states",
            "reference_states",
            "solve_ms",
            "self.mpc_diagnostics.close()",
        ):
            self.assertIn(required, source)

    def test_mpc_diagnostics_passes_dense_plan_to_reference_sampler(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        tree = ast.parse(client_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "find_reference_traj"
        ]

        self.assertTrue(
            any(
                len(call.args) == 2
                and isinstance(call.args[1], ast.Attribute)
                and call.args[1].attr == "ref_traj"
                for call in calls
            )
        )

    def test_client_captures_full_camera_transform_and_bounded_odom_history(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            "base_from_camera: np.ndarray",
            "transform_matrix_from_translation_quaternion",
            "self.odom_history = deque(maxlen=600)",
            "self.odom_history.append(pose.copy())",
        ):
            self.assertIn(required, source)

    def test_client_snapshots_successful_mpc_solution_for_visualization(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            "class MpcVisualizationSnapshot",
            "predicted_states: np.ndarray",
            "command: np.ndarray",
            "solve_ms: float",
            "updated_at: float",
            "self.latest_mpc_visualization =",
            "MpcVisualizationSnapshot(",
        ):
            self.assertIn(required, source)

    def test_client_queues_bev_frames_before_odom_or_plan_exists(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        queue_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_queue_visualization"
        )
        queue_source = ast.get_source_segment(source, queue_method)

        self.assertIn("VisualizationRequest(", queue_source)
        self.assertIn("state=self.visualization_state", queue_source)
        self.assertNotIn("snapshot.odom_xy_yaw is None", queue_source)
        self.assertNotIn("state is None", queue_source)

    def test_client_records_independent_mpc_rgb_bev_video(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        for required in (
            'f"{run_stamp}_mpc_rgb_bev.mp4"',
            'f"{run_stamp}_mpc_rgb_bev.partial.mp4"',
            "render_mpc_rgb_bev(",
            "self.mpc_bev_video_writer.release()",
            "self.mpc_bev_video_temporary,",
            "MPC BEV recording disabled",
            "--bev-sample-stride",
            "--mpc-bev-timeout",
        ):
            self.assertIn(required, source)

    def test_secondary_video_finalizes_after_worker_join(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        stop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "stop"
        )
        stop_source = ast.get_source_segment(source, stop)

        self.assertLess(
            stop_source.index("self.visualization_thread.join"),
            stop_source.index("self.mpc_bev_video_writer.release()"),
        )
        self.assertIn("self.visualization_thread.join()", stop_source)
        self.assertNotIn("self.visualization_thread.join(timeout=", stop_source)

    def test_client_transcodes_both_recordings_to_h264_on_shutdown(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        stop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "stop"
        )
        stop_source = ast.get_source_segment(source, stop)

        self.assertIn("finalize_mp4", source)
        self.assertEqual(stop_source.count("finalize_mp4("), 2)
        self.assertIn("H.264", stop_source)
        self.assertNotIn("mp4v fallback", stop_source)

    def test_bev_append_runs_even_without_footprint_visualization_state(self):
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        source = client_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        render_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_render_and_publish_visualization"
        )
        render_source = ast.get_source_segment(source, render_method)

        self.assertIn("self._append_mpc_bev_video(snapshot)", render_source)
        self.assertNotIn(
            "if state is None or snapshot.odom_xy_yaw is None:\n            return",
            render_source,
        )

    def test_viewer_serves_ros_visualization_as_mjpeg(self):
        viewer_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_visualization_server.py"
        )

        source = viewer_path.read_text(encoding="utf-8")

        for required in (
            "/navdp/visualization",
            "/cam_head/d435/color/image_raw",
            "ThreadingHTTPServer",
            "/stream.mjpg",
            "/latest.jpg",
            "/capture-goal",
            "GoalCaptureStore",
            "sys.path.insert(0, str(REPO_ROOT))",
            "拍摄 Goal",
            "--goal-dir",
            "--port",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
