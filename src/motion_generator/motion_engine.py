import os
import time
import json
import placo
import numpy as np


def _yaw_transform(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return T


class MotionEngine:
    _dt = 0.01
    _refine = 10

    _time_since_last_right_contact = 0.0
    _time_since_last_left_contact = 0.0

    _initial_delay = -1.0
    _t = -1.0
    _last_replan = 0

    _start_time = None

    _is_ignore_feet_contacts = False

    def __init__(self, robot_folder_path: str = "", gait_parameters: dict = {}) -> None:
        # Load limits
        self._limits_file_path = os.path.join(robot_folder_path, "limits.json")

        with open(self._limits_file_path, "r") as f:
            self._limits = json.load(f)

        # Load robot and parameters
        self._urdf_file_path = os.path.join(robot_folder_path, f"{gait_parameters['robot']}.urdf")

        self.robot = placo.HumanoidRobot(self._urdf_file_path)

        self.robot.set_velocity_limits(12.0)

        for key, lower in self._limits.items():
            if not key.endswith("_lower"):
                continue
            joint = key[:-len("_lower")]
            upper_key = f"{joint}_upper"
            if upper_key in self._limits:
                self.robot.set_joint_limits(joint, lower, self._limits[upper_key])

        self._robot_parameters = placo.HumanoidParameters()
        self._load_parameters(gait_parameters)

        # Load collision pairs
        #self._collisions_file_path = os.path.join(robot_folder_path, "collisions.json")
        #self.robot.load_collision_pairs(self._collisions_file_path)

        # Create kinematics solver
        self._solver = placo.KinematicsSolver(self.robot)

        self._solver.enable_velocity_limits(True)

        self._solver.enable_joint_limits(self._enable_joint_limits)

        self._solver.dt = self._dt / self._refine

        self._task = None
        self._generic_tasks = {}
        self._setup_ik_tasks()

        # Create joints task
        self._joints = self._robot_parameters.joints

        joint_angles = self._robot_parameters.joint_angles
        masked_joint_angles = {joint: np.deg2rad(degree) for joint, degree in joint_angles.items()}

        self._joints_task = self._solver.add_joints_task()

        self._joints_task.set_joints(masked_joint_angles)

        self._joints_task.configure("joints", "soft", 1.0)

        # Place robot in initial pose
        if self._ik_mode == "walk_tasks":
            self._task.reach_initial_pose(
                np.eye(4),
                self._robot_parameters.feet_spacing,
                self._robot_parameters.walk_com_height,
                self._robot_parameters.walk_trunk_pitch,
            )

        # Create Footsteps planner
        self._d_x = 0.0
        self._d_y = 0.0
        self._d_theta = 0.0
        self._number_of_steps = 5

        self._footsteps_planner = placo.FootstepsPlannerRepetitive(self._robot_parameters)
        
        self._footsteps_planner.configure(self._d_x, self._d_y, self._d_theta, self._number_of_steps)

        # Plan footsteps
        self._T_world_left = placo.flatten_on_floor(self._get_T_world_left_for_planning())
        self._T_world_right = placo.flatten_on_floor(self._get_T_world_right_for_planning())

        self._footsteps = self._footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self._T_world_left, self._T_world_right
        )

        self._supports = placo.FootstepsPlanner.make_supports(
            self._footsteps, 0.0, True, self._robot_parameters.has_double_support(), True
        )

        # Create pattern generator
        self._pattern_generator = placo.WalkPatternGenerator(self.robot, self._robot_parameters)

        # Nominal CoM for LIPM: same horizontal placement as reach_initial_pose (mid-feet at walk height).
        # With trunk_mode, IK tracks trunk position, not true CoM — robot.com_world() can sit outside the
        # support polygon and the LIPM QP then fails with "Failed to plan CoM trajectory".
        Tl = np.asarray(self._T_world_left, dtype=float)
        Tr = np.asarray(self._T_world_right, dtype=float)
        p_com_init = np.array(
            [
                0.5 * (Tl[0, 3] + Tr[0, 3]),
                0.5 * (Tl[1, 3] + Tr[1, 3]),
                self._robot_parameters.walk_com_height,
            ]
        )
        self._generic_trunk_position_offset = np.asarray(self.robot.get_T_world_frame("trunk"))[:3, 3] - p_com_init
        self._generic_trunk_position_offset[2] = 0.0
        self._trajectory = self._pattern_generator.plan(self._supports, p_com_init, 0.0)
        self._generic_trunk_orientation_offset = np.asarray(self._trajectory.get_R_world_trunk(0.0)).T @ np.asarray(
            self.robot.get_T_world_frame("trunk")
        )[:3, :3]
        if self._ik_mode == "generic_frame":
            self._update_generic_tasks_from_trajectory(0.0)

        # Period
        self._period = 2 * self._robot_parameters.single_support_duration + 2 * self._robot_parameters.double_support_duration()
        print(f"Period: {self._period}")

    
    @property
    def robot_parameters(self):
        return self._robot_parameters

    @property
    def t(self) -> float:
        return self._t

    @property
    def period(self):
        return self._period


    def _load_parameters(self, gait_parameters: dict) -> None:
        self._robot_parameters.trunk_mode = gait_parameters.get("trunk_mode", False)
        self._robot_parameters.double_support_ratio = gait_parameters.get("double_support_ratio", self._robot_parameters.double_support_ratio)
        self._robot_parameters.startend_double_support_ratio = gait_parameters.get("startend_double_support_ratio", self._robot_parameters.startend_double_support_ratio)
        self._robot_parameters.planned_timesteps = gait_parameters.get("planned_timesteps", self._robot_parameters.planned_timesteps)
        
        #self._robot_parameters.replan_timesteps = gait_parameters.get("replan_timesteps", self._robot_parameters.replan_timesteps)
        self._replan_timesteps = gait_parameters.get("replan_timesteps", 10)
        
        self._robot_parameters.walk_com_height = gait_parameters.get("walk_com_height", self._robot_parameters.walk_com_height)
        self._robot_parameters.walk_foot_height = gait_parameters.get("walk_foot_height", self._robot_parameters.walk_foot_height)
        self._robot_parameters.walk_trunk_pitch = gait_parameters.get("walk_trunk_pitch", self._robot_parameters.walk_trunk_pitch)
        self._robot_parameters.walk_foot_rise_ratio = gait_parameters.get("walk_foot_rise_ratio", self._robot_parameters.walk_foot_rise_ratio)
        self._robot_parameters.single_support_duration = gait_parameters.get("single_support_duration", self._robot_parameters.single_support_duration)
        self._robot_parameters.single_support_timesteps = gait_parameters.get("single_support_timesteps", self._robot_parameters.single_support_timesteps)
        self._robot_parameters.foot_length = gait_parameters.get("foot_length", self._robot_parameters.foot_length)
        self._robot_parameters.feet_spacing = gait_parameters.get("feet_spacing", self._robot_parameters.feet_spacing)
        self._robot_parameters.zmp_margin = gait_parameters.get("zmp_margin", self._robot_parameters.zmp_margin)
        self._robot_parameters.foot_zmp_target_x = gait_parameters.get("foot_zmp_target_x", self._robot_parameters.foot_zmp_target_x)
        self._robot_parameters.foot_zmp_target_y = gait_parameters.get("foot_zmp_target_y", self._robot_parameters.foot_zmp_target_y)
        self._robot_parameters.walk_max_dtheta = gait_parameters.get("walk_max_dtheta", self._robot_parameters.walk_max_dtheta)
        self._robot_parameters.walk_max_dy = gait_parameters.get("walk_max_dy", self._robot_parameters.walk_max_dy)
        self._robot_parameters.walk_max_dx_forward = gait_parameters.get("walk_max_dx_forward", self._robot_parameters.walk_max_dx_forward)
        self._robot_parameters.walk_max_dx_backward = gait_parameters.get("walk_max_dx_backward", self._robot_parameters.walk_max_dx_backward)
        self._robot_parameters.joints = gait_parameters.get("joints", [])
        self._robot_parameters.joint_angles = gait_parameters.get("joint_angles", {})
        self._ik_mode = gait_parameters.get("ik_mode", "walk_tasks")
        self._enable_joint_limits = gait_parameters.get("enable_joint_limits", self._ik_mode != "walk_tasks")
        self._ik_foot_orientation_axes = gait_parameters.get("ik_foot_orientation_axes", "yz" if self._ik_mode == "walk_tasks" else "xyz")
        self._ik_trunk_orientation_axes = gait_parameters.get("ik_trunk_orientation_axes", "xyz")
        self._ik_foot_task_weight = gait_parameters.get("ik_foot_task_weight", 10.0)
        self._ik_trunk_task_weight = gait_parameters.get("ik_trunk_task_weight", 5.0)
        self._right_foot_ik_yaw_offset = gait_parameters.get("right_foot_ik_yaw_offset", 0.0)

    def _setup_ik_tasks(self) -> None:
        if self._ik_mode == "walk_tasks":
            self._setup_walk_tasks()
        elif self._ik_mode == "generic_frame":
            self._setup_generic_frame_tasks()
        else:
            raise ValueError(f"Unknown ik_mode: {self._ik_mode}")

    def _setup_walk_tasks(self) -> None:
        self._task = placo.WalkTasks()

        self._task.trunk_mode = self._robot_parameters.trunk_mode

        self._task.com_x = 0.0

        self._task.initialize_tasks(self._solver, self.robot)

        self._task.left_foot_task.orientation().mask.set_axises(self._ik_foot_orientation_axes, "local")
        self._task.right_foot_task.orientation().mask.set_axises(self._ik_foot_orientation_axes, "local")

    def _setup_generic_frame_tasks(self) -> None:
        # The walk planner sees corrected humanoid foot frames; frame tasks still target the URDF model frames.
        self._T_walk_from_model_left = np.eye(4)
        self._T_walk_from_model_right = _yaw_transform(self._right_foot_ik_yaw_offset)
        self._T_model_from_walk_left = np.linalg.inv(self._T_walk_from_model_left)
        self._T_model_from_walk_right = np.linalg.inv(self._T_walk_from_model_right)

        left_foot_task = self._solver.add_frame_task("left_foot", self.robot.get_T_world_left())
        right_foot_task = self._solver.add_frame_task("right_foot", self.robot.get_T_world_right())
        trunk_task = self._solver.add_frame_task("trunk", self.robot.get_T_world_frame("trunk"))

        left_foot_task.configure("left_foot", "soft", self._ik_foot_task_weight)
        right_foot_task.configure("right_foot", "soft", self._ik_foot_task_weight)
        trunk_task.configure("trunk", "soft", self._ik_trunk_task_weight)

        for foot_task in (left_foot_task, right_foot_task):
            foot_task.position().mask.set_axises("xyz", "world")
            foot_task.orientation().mask.set_axises(self._ik_foot_orientation_axes, "local")

        trunk_task.position().mask.set_axises("xyz", "world")
        trunk_task.orientation().mask.set_axises(self._ik_trunk_orientation_axes, "local")

        self._generic_tasks = {
            "left_foot": left_foot_task,
            "right_foot": right_foot_task,
            "trunk": trunk_task,
        }

    def _get_T_world_left_for_planning(self) -> np.ndarray:
        if self._ik_mode == "generic_frame":
            return np.asarray(self.robot.get_T_world_left()) @ self._T_walk_from_model_left
        return self.robot.get_T_world_left()

    def _get_T_world_right_for_planning(self) -> np.ndarray:
        if self._ik_mode == "generic_frame":
            return np.asarray(self.robot.get_T_world_right()) @ self._T_walk_from_model_right
        return self.robot.get_T_world_right()

    def _update_generic_tasks_from_trajectory(self, t: float) -> None:
        T_world_left_walk = np.asarray(self._trajectory.get_T_world_left(t))
        T_world_right_walk = np.asarray(self._trajectory.get_T_world_right(t))

        T_world_left_model = T_world_left_walk @ self._T_model_from_walk_left
        T_world_right_model = T_world_right_walk @ self._T_model_from_walk_right

        p_world_trunk = np.asarray(self._trajectory.get_p_world_CoM(t)) + self._generic_trunk_position_offset
        R_world_trunk = np.asarray(self._trajectory.get_R_world_trunk(t)) @ self._generic_trunk_orientation_offset
        T_world_trunk = np.eye(4)
        T_world_trunk[:3, :3] = R_world_trunk
        T_world_trunk[:3, 3] = p_world_trunk

        self._generic_tasks["left_foot"].T_world_frame = T_world_left_model
        self._generic_tasks["right_foot"].T_world_frame = T_world_right_model
        self._generic_tasks["trunk"].T_world_frame = T_world_trunk


    def get_angles(self) -> dict:
        return {joint: self.robot.get_joint(joint) for joint in self._joints}

    def get_supports(self):
        return self._trajectory.get_supports()

    def get_current_support_phase(self):
        if self._trajectory.support_is_both(self._t):
            return [1, 1]
        elif str(self._trajectory.support_side(self._t)) == "left":
            return [1, 0]
        elif str(self._trajectory.support_side(self._t)) == "right":
            return [0, 1]
        else:
            raise ValueError(f"Invalid support phase at time {self._t}")

    def set_trajectory(self, dx: float, dy: float, dtheta: float) -> None:
        self._d_x = dx
        self._d_y = dy
        self._d_theta = dtheta

        self._footsteps_planner.configure(self._d_x, self._d_y, self._d_theta, self._number_of_steps)

    def tick(self, dt: float) -> None:
        if self._start_time is None:
            self._start_time = time.time()

        if not self._is_ignore_feet_contacts:
            self._time_since_last_left_contact = 0.0
            self._time_since_last_right_contact = 0.0

        falling = not self._is_ignore_feet_contacts and (
            self._time_since_last_left_contact > self._robot_parameters.single_support_duration or
            self._time_since_last_right_contact > self._robot_parameters.single_support_duration
        )

        for k in range(self._refine):
            if not falling:
                trajectory_t = self._t - dt + k * self._dt / self._refine
                if self._ik_mode == "walk_tasks":
                    self._task.update_tasks_from_trajectory(self._trajectory, trajectory_t)
                else:
                    self._update_generic_tasks_from_trajectory(trajectory_t)

            self.robot.update_kinematics()
            _ = self._solver.solve(True)

        if (self._t - self._last_replan > self._replan_timesteps * self._robot_parameters.dt() and 
            self._pattern_generator.can_replan_supports(self._trajectory, self._t)):
            
            self._supports = self._pattern_generator.replan_supports(self._footsteps_planner, self._trajectory, self._t, self._last_replan)

            self._trajectory = self._pattern_generator.replan(self._supports, self._trajectory, self._t)

            self._last_replan = self._t

        self._time_since_last_left_contact += dt
        self._time_since_last_right_contact += dt
        self._t += dt
