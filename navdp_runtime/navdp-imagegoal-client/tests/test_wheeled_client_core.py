import ast
import json
import math
import queue
import subprocess
import sys
import tempfile
import time
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
    @staticmethod
    def level_optical_transform(height):
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        transform[:3, 3] = [0.0, 0.0, height]
        return transform

    def test_official_camera_transform_is_level_at_fixed_height(self):
        self.assertEqual(client_core.NAVDP_OFFICIAL_CAMERA_HEIGHT_M, 0.2)

        transform = client_core.navdp_official_base_from_camera()

        np.testing.assert_array_equal(
            transform,
            np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.2],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_navdp_virtual_pixels_match_official_height_formula(self):
        intrinsic = np.array(
            [[100.0, 0.0, 2.0], [0.0, 120.0, 2.0], [0.0, 0.0, 1.0]]
        )
        local_xy = np.array([[1.0, 0.5], [2.0, -0.5]])

        pixels = client_core.navdp_virtual_pixels(
            local_xy,
            intrinsic,
            image_height=5,
            virtual_camera_height=0.2,
        )

        np.testing.assert_allclose(
            pixels,
            [[-48.0, 26.0], [27.0, 14.0]],
            atol=1e-12,
        )

    def test_navdp_virtual_pixels_reject_nonpositive_forward_point(self):
        with self.assertRaisesRegex(
            ValueError,
            "virtual_reprojection: forward distance must be positive",
        ):
            client_core.navdp_virtual_pixels(
                np.array([[0.0, 0.0], [1.0, 0.0]]),
                np.eye(3),
                image_height=5,
            )

    def test_virtual_height_equals_real_height_reprojects_identity(self):
        intrinsic = np.array(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
        )
        local_xy = np.array([[0.5, -0.1], [1.0, 0.2], [2.0, 0.4]])

        ground = client_core.reproject_navdp_to_ground_base(
            local_xy,
            intrinsic,
            image_height=5,
            base_from_camera=self.level_optical_transform(0.2),
            virtual_camera_height=0.2,
        )

        np.testing.assert_allclose(ground, local_xy, atol=1e-12)

    def test_pitched_camera_uses_full_optical_rotation(self):
        intrinsic = np.array(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
        )
        level = self.level_optical_transform(1.0)
        pitch = math.radians(20.0)
        pitch_rotation = np.array(
            [
                [math.cos(pitch), 0.0, math.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch), 0.0, math.cos(pitch)],
            ]
        )
        pitched = level.copy()
        pitched[:3, :3] = pitch_rotation @ level[:3, :3]
        local_xy = np.array([[1.0, 0.0], [2.0, 0.0]])

        ground = client_core.reproject_navdp_to_ground_base(
            local_xy,
            intrinsic,
            image_height=5,
            base_from_camera=pitched,
            virtual_camera_height=0.2,
        )

        first_ray = pitch_rotation @ np.array([1.0, 0.0, -0.2])
        second_ray = pitch_rotation @ np.array([1.0, 0.0, -0.1])
        expected = np.array(
            [
                [-first_ray[0] / first_ray[2], 0.0],
                [-second_ray[0] / second_ray[2], 0.0],
            ]
        )
        np.testing.assert_allclose(ground, expected, atol=1e-12)

    def test_ground_reprojection_rejects_parallel_ray(self):
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        transform[:3, 3] = [0.0, 0.0, 1.0]

        with self.assertRaisesRegex(
            ValueError,
            "virtual_reprojection: ray is parallel to ground",
        ):
            client_core.reproject_navdp_to_ground_base(
                np.array([[1.0, 0.0], [2.0, 0.0]]),
                np.array(
                    [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
                ),
                image_height=5,
                base_from_camera=transform,
            )

    def test_ground_reprojection_rejects_intersection_behind_camera(self):
        transform = self.level_optical_transform(-1.0)

        with self.assertRaisesRegex(
            ValueError,
            "virtual_reprojection: ground intersection is behind camera",
        ):
            client_core.reproject_navdp_to_ground_base(
                np.array([[1.0, 0.0], [2.0, 0.0]]),
                np.array(
                    [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
                ),
                image_height=5,
                base_from_camera=transform,
            )

    def test_ground_reprojection_rejects_nan_local_input(self):
        with self.assertRaisesRegex(
            ValueError,
            "virtual_reprojection: local_xy must be finite",
        ):
            client_core.reproject_navdp_to_ground_base(
                np.array([[1.0, 0.0], [np.nan, 0.0]]),
                np.eye(3),
                image_height=5,
                base_from_camera=self.level_optical_transform(0.2),
            )

    def test_ground_reprojection_rejects_reversed_forward_progress(self):
        with self.assertRaisesRegex(
            ValueError,
            "virtual_reprojection: trajectory reverses forward progress",
        ):
            client_core.reproject_navdp_to_ground_base(
                np.array([[2.0, 0.0], [1.0, 0.0]]),
                np.array(
                    [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
                ),
                image_height=5,
                base_from_camera=self.level_optical_transform(0.2),
            )

    def test_normalizes_direct_tracking_trajectory_without_resampling(self):
        trajectory = np.array(
            [[0.5, 0.1], [0.5, 0.1], [1.0, 0.2], [1.5, 0.4]]
        )

        normalized = client_core.normalize_tracking_trajectory(trajectory)

        np.testing.assert_array_equal(normalized, trajectory)

    def test_tracking_generation_rejects_a_result_after_invalidation(self):
        self.assertTrue(
            client_core.tracking_generation_is_current(
                captured_generation=4,
                current_generation=4,
                trajectory_ready=True,
            )
        )
        self.assertFalse(
            client_core.tracking_generation_is_current(
                captured_generation=4,
                current_generation=5,
                trajectory_ready=True,
            )
        )
        self.assertFalse(
            client_core.tracking_generation_is_current(
                captured_generation=4,
                current_generation=4,
                trajectory_ready=False,
            )
        )

    def test_rejects_invalid_direct_tracking_trajectory(self):
        for trajectory in (
            np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [np.nan, 0.0]]),
            np.array([1.0, 2.0]),
        ):
            with self.subTest(trajectory=trajectory):
                with self.assertRaises(ValueError):
                    client_core.normalize_tracking_trajectory(trajectory)

    def test_pending_selected_survives_failed_install_and_rejected_retry(self):
        old = np.array([[0.0, 0.0], [1.0, 0.0]])
        accepted = np.array([[0.5, 0.2], [1.5, 0.4]])
        rejected = np.array([[0.5, 1.0], [1.5, 1.0]])
        state = client_core.SelectedDiffusionInstallState()
        state = state.stage(old, candidate_accepted=True).commit()

        state = state.stage(accepted, candidate_accepted=True)
        state = state.stage(rejected, candidate_accepted=False)
        state = state.commit()

        np.testing.assert_array_equal(state.installed, accepted)
        self.assertIsNone(state.pending)
        self.assertFalse(state.installed.flags.writeable)
        accepted[0] = [9.0, 9.0]
        np.testing.assert_array_equal(
            state.installed,
            [[0.5, 0.2], [1.5, 0.4]],
        )

    def test_unavailable_trajectory_clears_selected_provenance(self):
        state = client_core.SelectedDiffusionInstallState()
        state = state.stage(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            candidate_accepted=True,
        ).commit()
        state = state.stage(
            np.array([[0.5, 0.2], [1.5, 0.4]]),
            candidate_accepted=True,
        )

        state = state.clear()

        self.assertIsNone(state.installed)
        self.assertIsNone(state.pending)

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

    def test_client_tracks_reprojected_path_directly_but_snapshots_raw_selected(self):
        source = self.client_source()

        for required in (
            "reproject_navdp_to_ground_base",
            "raw_selected_world_xy",
            "reprojected_base_xy",
            "reprojected_world_xy",
            "normalize_tracking_trajectory(\n"
            "                        reprojected_world_xy\n"
            "                    )",
            ".stage(raw_selected_world_xy, True)",
        ):
            self.assertIn(required, source)
        self.assertNotIn("TrajectoryManager", source)
        self.assertNotIn("trajectory_manager.update", source)
        self.assertNotIn(
            "candidate_world_xy=retained_raw_world_xy",
            source,
        )

    def test_invalid_reprojection_does_not_retain_an_old_trajectory(self):
        source = self.client_source()

        self.assertIn("reprojection_error = None", source)
        self.assertIn("except ValueError as error:", source)
        self.assertIn("reprojection_error = str(error)", source)
        self.assertIn("reprojected_world_xy = None", source)
        self.assertIn("self.installed_active_traj = None", source)
        self.assertIn("self._invalidate_tracking_state()", source)
        self.assertNotIn("trajectory_manager.update", source)

    def test_client_exposes_virtual_camera_height_default(self):
        source = self.client_source()
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        help_result = subprocess.run(
            [sys.executable, str(client_path), "--help"],
            cwd=client_path.parents[2],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertIn(
            'parser.add_argument("--virtual-camera-height", type=float, default=0.2)',
            source,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--virtual-camera-height", help_result.stdout)

    def test_client_logs_raw_and_reprojected_plan_geometry(self):
        source = self.client_source()

        for required in (
            '"raw_selected_world_xy": raw_selected_world_xy',
            '"reprojected_base_xy": reprojected_base_xy',
            '"reprojected_world_xy": reprojected_world_xy',
            '"virtual_camera_height_m": self.args.virtual_camera_height',
            '"reprojection_status":',
            '"reprojection_reason": reprojection_error',
        ):
            self.assertIn(required, source)

    def test_client_distinguishes_unattempted_reprojection_and_throttles_rejections(self):
        source = self.client_source()

        for required in (
            'reprojection_status = "not_attempted"',
            'reprojection_status = "ok"',
            'reprojection_status = "rejected"',
            '"reprojection_status": reprojection_status',
            "self.last_reprojection_error_log = 0.0",
            "reprojection_log_time - self.last_reprojection_error_log >= 2.0",
            "self.last_reprojection_error_log = reprojection_log_time",
        ):
            self.assertIn(required, source)

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

    def test_controller_copy_records_upstream_navdp_provenance(self):
        controller_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "controllers.py"
        )

        source = controller_path.read_text(encoding="utf-8")

        self.assertIn("InternRobotics/NavDP", source)
        self.assertIn("bebb436a9856acbd6ed2a63234a99db6bac2fd3a", source)
        self.assertIn("class Mpc_controller", source)

    def test_controller_cost_matches_upstream_zero_yaw_reference(self):
        controller_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "controllers.py"
        )
        source = controller_path.read_text(encoding="utf-8")

        self.assertIn("Q = np.diag([10.0, 10.0, 0.0])", source)
        self.assertIn("R = np.diag([0.02, 0.15])", source)
        self.assertIn("np.zeros((ref_traj.shape[0], 1))", source)
        self.assertNotIn("reference_poses_from_xy", source)

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
            "--virtual-camera-height 0.2",
            "虚拟相机高度",
            "黄色",
            "青色",
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

    def test_client_does_not_drop_a_configurable_diffusion_prefix(self):
        source = self.client_source()

        self.assertNotIn("skip_trajectory_points", source)
        self.assertNotIn("--skip-trajectory-points", source)

    def test_client_installs_complete_reprojected_diffusion_in_upstream_mpc(self):
        source = self.client_source()

        self.assertIn(
            "active_traj = normalize_tracking_trajectory(\n"
            "                        reprojected_world_xy",
            source,
        )
        self.assertIn("next_mpc = Mpc_controller(", source)
        self.assertIn("desired_v=self.args.max_v", source)
        self.assertNotIn("TrajectoryManager", source)
        self.assertNotIn("trajectory_update", source)
        self.assertNotIn("blind_steps=", source)
        self.assertNotIn("N=trajectory_update", source)
        self.assertNotIn("retained_reprojected_world_xy", source)

    def test_client_rebuilds_upstream_mpc_for_every_valid_plan(self):
        source = self.client_source()

        self.assertIn("next_mpc = Mpc_controller(", source)
        self.assertNotIn("self.mpc.update_ref_traj(", source)
        self.assertNotIn("prediction_steps", source)

    def test_client_gates_control_on_active_trajectory(self):
        source = self.client_source()

        self.assertIn("self.trajectory_ready = False", source)
        self.assertIn("trajectory_ready=self.trajectory_ready", source)
        self.assertNotIn("critic_safe=self.critic_safe", source)

    def test_client_blocks_control_when_active_trajectory_install_fails(self):
        source = self.client_source()

        self.assertIn("failed to install active trajectory", source)
        self.assertIn('reason = "active_trajectory_error"', source)

    def test_client_invalidates_inflight_control_and_bev_snapshot(self):
        source = self.client_source()

        self.assertIn("self.trajectory_generation += 1", source)
        self.assertIn("self.latest_mpc_visualization = None", source)
        self.assertIn("tracking_generation_is_current(", source)
        self.assertIn("captured_generation=control_generation", source)
        self.assertIn('reason = "plan_superseded"', source)

    def test_client_does_not_expose_trajectory_manager_options(self):
        source = self.client_source()
        client_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "navdp_imagegoal_client.py"
        )
        help_result = subprocess.run(
            [sys.executable, str(client_path), "--help"],
            cwd=client_path.parents[2],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertNotIn("--trajectory-", help_result.stdout)
        self.assertNotIn("trajectory_point_spacing", source)
        self.assertNotIn("trajectory_join_distance", source)
        self.assertNotIn("trajectory_join_heading_deg", source)
        self.assertNotIn("trajectory_min_remaining", source)

    def test_client_records_direct_reprojected_active_trajectory(self):
        source = self.client_source()

        for required in (
            '"reprojected_world_xy": reprojected_world_xy',
            '"active_traj": active_traj',
            '"mpc_horizon": self.mpc.N',
        ):
            self.assertIn(required, source)
        self.assertNotIn('"candidate_accepted"', source)
        self.assertNotIn('"trajectory_reason"', source)
        self.assertNotIn('"trajectory_blind_', source)
        self.assertNotIn('"trajectory_diffusion_', source)
        self.assertNotIn('"trajectory_manager_update_ms"', source)

    def test_bev_uses_rgbd_snapshot_odom_as_the_render_pose(self):
        source = self.client_source()
        tree = ast.parse(source)
        render_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_render_mpc_bev"
        )
        render_source = ast.get_source_segment(source, render_method)
        self.assertIn(
            "frame_odom = (\n"
            "            None\n"
            "            if snapshot.odom_xy_yaw is None\n"
            "            else snapshot.odom_xy_yaw.copy()\n"
            "        )",
            render_source,
        )

        calls = [
            node
            for node in ast.walk(render_method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        freshness_call = next(
            call for call in calls if call.func.id == "bev_freshness"
        )
        freshness_keywords = {
            keyword.arg: keyword.value
            for keyword in freshness_call.keywords
            if keyword.arg is not None
        }
        self.assertIsInstance(freshness_keywords["current_odom"], ast.Name)
        self.assertEqual(freshness_keywords["current_odom"].id, "frame_odom")

        render_call = next(
            call for call in calls if call.func.id == "render_mpc_rgb_bev"
        )
        render_keywords = {
            keyword.arg: keyword.value
            for keyword in render_call.keywords
            if keyword.arg is not None
        }
        self.assertIsInstance(
            render_keywords["current_odom_xy_yaw"], ast.Name
        )
        self.assertEqual(
            render_keywords["current_odom_xy_yaw"].id, "frame_odom"
        )

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

    def test_client_snapshots_active_trajectory_for_fresh_bev_rendering(self):
        source = self.client_source()
        tree = ast.parse(source)
        render_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_render_mpc_bev"
        )
        render_source = ast.get_source_segment(source, render_method)

        for required in (
            "active_traj: np.ndarray",
            "selected_diffusion: Optional[np.ndarray]",
            "self.installed_active_traj: Optional[np.ndarray] = None",
            "self.selected_diffusion_state = SelectedDiffusionInstallState()",
            "next_active_traj = np.asarray(active_traj).copy()",
            "next_active_traj.setflags(write=False)",
            "self.installed_active_traj = next_active_traj",
            "active_traj_snapshot = self.installed_active_traj.copy()",
            "active_traj_snapshot.setflags(write=False)",
            "self.selected_diffusion_state.installed",
            "active_traj=active_traj_snapshot",
            "selected_diffusion=selected_diffusion_snapshot",
            "active_traj = None",
            "selected_diffusion = None",
            "active_traj = mpc_snapshot.active_traj",
            "selected_diffusion = mpc_snapshot.selected_diffusion",
            "active_traj=active_traj",
            "selected_diffusion=selected_diffusion",
        ):
            self.assertIn(required, source)

        fresh_block_start = render_source.index(
            'if mpc_fresh and odom_status == "ODOM OK":'
        )
        active_traj_assignment = render_source.index(
            "active_traj = mpc_snapshot.active_traj"
        )
        selected_assignment = render_source.index(
            "selected_diffusion = mpc_snapshot.selected_diffusion"
        )
        render_call = render_source.index("active_traj=active_traj")
        selected_render_call = render_source.index(
            "selected_diffusion=selected_diffusion"
        )
        self.assertLess(fresh_block_start, active_traj_assignment)
        self.assertLess(fresh_block_start, selected_assignment)
        self.assertLess(active_traj_assignment, render_call)
        self.assertLess(selected_assignment, selected_render_call)

    def test_client_publishes_selected_only_after_mpc_construction(self):
        source = self.client_source()
        tree = ast.parse(source)
        planning_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_planning_loop"
        )
        install_try = next(
            node
            for node in ast.walk(planning_method)
            if isinstance(node, ast.Try)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "Mpc_controller"
                for call in ast.walk(node)
            )
        )
        install_calls = [
            call
            for call in ast.walk(install_try)
            if isinstance(call, ast.Call)
        ]
        mpc_install_lines = [
            call.lineno
            for call in install_calls
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == "Mpc_controller"
            )
            or (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "update_ref_traj"
            )
        ]
        selected_commit = next(
            call
            for call in install_calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "commit"
        )
        selected_publish = next(
            node
            for node in ast.walk(install_try)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "selected_diffusion_state"
                for target in node.targets
            )
        )

        self.assertGreater(selected_commit.lineno, max(mpc_install_lines))
        self.assertGreater(selected_publish.lineno, max(mpc_install_lines))

    def test_client_hides_odom_frame_paths_when_odom_is_stale_but_mpc_is_fresh(self):
        source = self.client_source()
        tree = ast.parse(source)
        render_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_render_mpc_bev"
        )
        render_source = ast.get_source_segment(source, render_method)

        self.assertIn(
            'if mpc_fresh and odom_status == "ODOM OK":',
            render_source,
        )
        guarded_paths = render_source[
            render_source.index('if mpc_fresh and odom_status == "ODOM OK":') :
            render_source.index('if odom_status != "ODOM OK":')
        ]
        self.assertIn("predicted_states = mpc_snapshot.predicted_states", guarded_paths)
        self.assertIn("active_traj = mpc_snapshot.active_traj", guarded_paths)
        self.assertIn(
            "selected_diffusion = mpc_snapshot.selected_diffusion",
            guarded_paths,
        )

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
