#!/usr/bin/env python3
# Source: https://github.com/InternRobotics/NavDP
# File: utils_tasks/tracking_utils.py
# Commit: bebb436a9856acbd6ed2a63234a99db6bac2fd3a

import casadi as ca
import numpy as np
from scipy.interpolate import interp1d


def reference_poses_from_xy(reference_xy, current_yaw):
    reference_xy = np.asarray(reference_xy, dtype=np.float64)
    if (
        reference_xy.ndim != 2
        or reference_xy.shape[1] != 2
        or len(reference_xy) < 2
    ):
        raise ValueError("reference_xy must have shape (N, 2), N >= 2")

    deltas = np.diff(reference_xy, axis=0)
    valid = np.linalg.norm(deltas, axis=1) > np.finfo(np.float64).eps
    if not np.any(valid):
        yaws = np.full(len(reference_xy), current_yaw)
        return np.column_stack((reference_xy, yaws))

    valid_indices = np.flatnonzero(valid)
    segment_yaws = np.unwrap(np.arctan2(deltas[valid, 1], deltas[valid, 0]))
    yaws = np.interp(np.arange(len(reference_xy)), valid_indices, segment_yaws)
    yaws += 2.0 * np.pi * np.round(
        (current_yaw - yaws[0]) / (2.0 * np.pi)
    )
    return np.column_stack((reference_xy, yaws))


class Mpc_controller:
    def __init__(
        self,
        global_planed_traj,
        N=10,
        desired_v=0.5,
        v_max=0.5,
        w_max=0.5,
        ref_gap=3,
    ):
        self.N = N
        self.desired_v = desired_v
        self.ref_gap = ref_gap
        self.T = 0.1
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = N // ref_gap + 1

        opti = ca.Opti()
        opt_controls = opti.variable(N, 2)
        v, w = opt_controls[:, 0], opt_controls[:, 1]
        opt_states = opti.variable(N + 1, 3)
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3 * self.ref_traj_len)

        f = lambda x_, u_: ca.vertcat(  # noqa: E731
            *[
                u_[0] * ca.cos(x_[2]),
                u_[0] * ca.sin(x_[2]),
                u_[1],
            ]
        )

        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(N):
            x_next = (
                opt_states[i, :]
                + f(opt_states[i, :], opt_controls[i, :]).T * self.T
            )
            opti.subject_to(opt_states[i + 1, :] == x_next)

        Q = np.diag([10.0, 10.0, 0.0])
        R = np.diag([0.02, 0.15])
        obj = 0
        for i in range(N):
            obj = obj + ca.mtimes(
                [opt_controls[i, :], R, opt_controls[i, :].T]
            )
            if i % ref_gap == 0:
                nn = i // ref_gap
                pose_error = (
                    opt_states[i, :]
                    - opt_xs[nn * 3 : nn * 3 + 3].T
                )
                obj = obj + ca.mtimes(
                    [pose_error, Q, pose_error.T]
                )
        opti.minimize(obj)

        opti.subject_to(opti.bounded(0.0, v, v_max))
        opti.subject_to(opti.bounded(-w_max, w, w_max))
        opti.solver(
            "ipopt",
            {
                "ipopt.max_iter": 100,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.acceptable_tol": 1e-8,
                "ipopt.acceptable_obj_change_tol": 1e-6,
            },
        )

        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

    def make_ref_denser(self, ref_traj, ratio=50):
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(
            0,
            len(ref_traj) - 1,
            num=len(ref_traj) * ratio,
        )
        interp_func_x = interp1d(
            x_orig,
            ref_traj[:, 0],
            kind="linear",
        )
        interp_func_y = interp1d(
            x_orig,
            ref_traj[:, 1],
            kind="linear",
        )
        return np.stack(
            (
                interp_func_x(new_x),
                interp_func_y(new_x),
            ),
            axis=1,
        )

    def update_ref_traj(self, global_planed_traj):
        dense_ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj = dense_ref_traj
        self.ref_traj_len = self.N // self.ref_gap + 1

    def solve(self, x0):
        ref_traj = self.find_reference_traj(x0, self.ref_traj)
        ref_traj = np.concatenate(
            (
                ref_traj,
                np.zeros((ref_traj.shape[0], 1)),
            ),
            axis=1,
        ).reshape(-1, 1)
        self.opti.set_value(self.opt_xs, ref_traj.reshape(-1, 1))
        u0 = (
            np.zeros((self.N, 2))
            if self.last_opt_u_controls is None
            else self.last_opt_u_controls
        )
        x00 = (
            np.zeros((self.N + 1, 3))
            if self.last_opt_x_states is None
            else self.last_opt_x_states
        )
        self.opti.set_value(self.opt_x0, x0)
        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, x00)
        sol = self.opti.solve()
        self.last_opt_u_controls = sol.value(self.opt_controls)
        self.last_opt_x_states = sol.value(self.opt_states)
        return self.last_opt_u_controls, self.last_opt_x_states

    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

    def find_reference_traj(self, x0, global_planed_traj):
        ref_traj_pts = []
        nearest_idx = np.argmin(
            np.linalg.norm(
                global_planed_traj - x0[:2].reshape((1, 2)),
                axis=1,
            )
        )
        desire_arc_length = self.desired_v * self.ref_gap * self.T
        cum_dist = np.cumsum(
            np.linalg.norm(
                np.diff(global_planed_traj, axis=0),
                axis=1,
            )
        )
        for i in range(nearest_idx, len(global_planed_traj) - 1):
            if (
                cum_dist[i] - cum_dist[nearest_idx]
                >= desire_arc_length * len(ref_traj_pts)
            ):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)
