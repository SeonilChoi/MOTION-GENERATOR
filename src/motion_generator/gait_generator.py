import os
import time
import json
import argparse
import threading
import webbrowser
import numpy as np

from placo_utils.visualization import robot_viz, robot_frame_viz, footsteps_viz
from scipy.spatial.transform import Rotation as R

from motion_generator.motion_engine import MotionEngine


class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, ".5f"))


def open_browser():
    webbrowser.open("http://127.0.0.1:7000/static/")


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_yaw_velocity(curr_quat, prev_quat, dt):
    if prev_quat is None:
        return 0.0

    prev_yaw = R.from_quat(prev_quat).as_euler("zyx")[0]
    curr_yaw = R.from_quat(curr_quat).as_euler("zyx")[0]

    return wrap_to_pi(curr_yaw - prev_yaw) / dt


def compute_angular_velocity(curr_quat, prev_quat, dt):
    if prev_quat is None:
        prev_quat = curr_quat

    r0 = R.from_quat(prev_quat)
    r1 = R.from_quat(curr_quat)

    r_relative = r1 * r0.inv()

    angular_velocity = r_relative.as_rotvec() / dt
    angular_velocity[2] = compute_yaw_velocity(curr_quat, prev_quat, dt)

    return list(angular_velocity)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_neck_step_motion(gait_parameters, joint_names):
    config = gait_parameters.get("neck_step_motion", {})
    if not config.get("enabled", False) or "neck_pitch" not in joint_names:
        return None

    base_angle = np.deg2rad(float(config.get("base_angle_deg", 0.0)))
    counter_head_config = config.get("counter_head_pitch", {})
    counter_head_enabled = counter_head_config.get("enabled", False) and "head_pitch" in joint_names

    return {
        "neck_index": joint_names.index("neck_pitch"),
        "head_pitch_index": joint_names.index("head_pitch") if counter_head_enabled else None,
        "base_angle": base_angle,
        "fold_angle": np.deg2rad(float(config.get("fold_angle_deg", config.get("left_support_angle_deg", 5.0)))),
        "cycle_duration": max(float(config.get("cycle_duration", config.get("transition_duration", 0.6))), 1e-3),
        "min_interval": max(float(config.get("min_interval", config.get("cycle_duration", config.get("transition_duration", 0.6)))), 0.0),
        "counter_head_enabled": counter_head_enabled,
        "counter_head_ratio": float(counter_head_config.get("ratio", 1.0)),
        "counter_head_max_angle": np.deg2rad(float(counter_head_config.get("max_angle_deg", 7.0))),
        "cycle_start": None,
        "cycle_start_angle": base_angle,
        "last_single_support": None,
        "last_cycle_start": -np.inf,
    }


def update_neck_step_motion(neck_motion, foot_contacts, t):
    if neck_motion is None:
        return

    contacts = np.asarray(foot_contacts, dtype=int)
    if contacts[0] == 1 and contacts[1] == 0:
        single_support = "left"
    elif contacts[0] == 0 and contacts[1] == 1:
        single_support = "right"
    else:
        return

    if single_support != neck_motion["last_single_support"] and t - neck_motion["last_cycle_start"] >= neck_motion["min_interval"]:
        neck_motion["cycle_start_angle"] = sample_neck_step_motion(neck_motion, t)
        neck_motion["cycle_start"] = t
        neck_motion["last_cycle_start"] = t

    neck_motion["last_single_support"] = single_support


def sample_neck_step_motion(neck_motion, t):
    if neck_motion is None:
        return 0.0

    if neck_motion["cycle_start"] is None:
        return neck_motion["base_angle"]

    elapsed = t - neck_motion["cycle_start"]
    if elapsed < 0.0 or elapsed > neck_motion["cycle_duration"]:
        return neck_motion["base_angle"]

    cycle_ratio = elapsed / neck_motion["cycle_duration"]
    if cycle_ratio <= 0.5:
        alpha = smoothstep(cycle_ratio * 2.0)
        return neck_motion["cycle_start_angle"] + alpha * (
            neck_motion["base_angle"] + neck_motion["fold_angle"] - neck_motion["cycle_start_angle"]
        )

    alpha = smoothstep((cycle_ratio - 0.5) * 2.0)
    return neck_motion["base_angle"] + neck_motion["fold_angle"] * (1.0 - alpha)


def sample_neck_counter_head_pitch(neck_motion, t):
    if neck_motion is None or not neck_motion["counter_head_enabled"]:
        return 0.0

    neck_delta = sample_neck_step_motion(neck_motion, t) - neck_motion["base_angle"]
    counter_angle = -neck_delta * neck_motion["counter_head_ratio"]
    max_angle = neck_motion["counter_head_max_angle"]
    return np.clip(counter_angle, -max_angle, max_angle)


def apply_neck_step_motion(joints_positions, neck_motion, t):
    if neck_motion is None:
        return joints_positions

    joints_positions = joints_positions.copy()
    joints_positions[neck_motion["neck_index"]] = sample_neck_step_motion(neck_motion, t)
    return joints_positions


def apply_neck_counter_head_motion(joints_positions, neck_motion, t):
    if neck_motion is None or not neck_motion["counter_head_enabled"]:
        return joints_positions

    joints_positions = joints_positions.copy()
    joints_positions[neck_motion["head_pitch_index"]] += sample_neck_counter_head_pitch(neck_motion, t)
    return joints_positions


def apply_neck_step_motion_to_robot(robot, neck_motion, t):
    if neck_motion is None:
        return

    robot.set_joint("neck_pitch", sample_neck_step_motion(neck_motion, t))
    robot.update_kinematics()


def apply_neck_counter_head_motion_to_robot(robot, neck_motion, t):
    if neck_motion is None or not neck_motion["counter_head_enabled"]:
        return

    robot.set_joint("head_pitch", robot.get_joint("head_pitch") + sample_neck_counter_head_pitch(neck_motion, t))
    robot.update_kinematics()


def make_head_motion(gait_parameters, joint_names, duration):
    config = gait_parameters.get("head_motion", {})
    if not config.get("enabled", False):
        return None

    required_joints = ("head_pitch", "head_yaw")
    if any(joint not in joint_names for joint in required_joints):
        return None

    yaw_limit = np.deg2rad(float(config.get("progress_yaw_limit_deg", 25.0)))
    min_forward = float(config.get("progress_min_forward", 0.03))
    lateral_yaw = np.arctan2(gait_parameters["dy"], max(abs(gait_parameters["dx"]), min_forward))
    turn_yaw = float(config.get("turn_yaw_gain", 2.0)) * gait_parameters["dth"]
    progress_yaw = np.clip(lateral_yaw + turn_yaw, -yaw_limit, yaw_limit)
    progress_pose = {"head_pitch": 0.0, "head_yaw": progress_yaw}

    glance_offsets_deg = config.get("glance_offsets_deg", config.get("poses_deg", {}))
    glance_names = config.get("glance_sequence", list(glance_offsets_deg.keys()))
    glance_offsets = {
        name: {
            "head_pitch": np.deg2rad(values.get("head_pitch", 0.0)),
            "head_yaw": np.deg2rad(values.get("head_yaw", 0.0)),
        }
        for name, values in glance_offsets_deg.items()
    }
    glance_names = [name for name in glance_names if name in glance_offsets]

    progress_hold_duration = float(config.get("progress_hold_duration", config.get("hold_duration", 3.0)))
    glance_hold_duration = float(config.get("glance_hold_duration", 1.0))
    transition_duration = float(config.get("transition_duration", 0.75))
    progress_hold_duration = max(progress_hold_duration, 1e-3)
    glance_hold_duration = max(glance_hold_duration, 1e-3)
    transition_duration = max(transition_duration, 1e-3)

    rng = np.random.default_rng(config.get("seed"))
    min_progress_segments = int(config.get("min_progress_segments_between_glances", 1))
    max_progress_segments = int(config.get("max_progress_segments_between_glances", 3))
    min_progress_segments = max(0, min_progress_segments)
    max_progress_segments = max(min_progress_segments, max_progress_segments)
    segments_until_glance = rng.integers(min_progress_segments, max_progress_segments + 1)
    previous_glance = None
    segments = []
    t = 0.0

    antenna_config = config.get("antenna_on_up", {})
    antenna_enabled = (
        antenna_config.get("enabled", False)
        and "left_antenna" in joint_names
        and "right_antenna" in joint_names
    )

    while t < duration + transition_duration:
        if glance_names and segments_until_glance <= 0:
            candidates = [name for name in glance_names if name != previous_glance]
            name = rng.choice(candidates if candidates else glance_names)
            offset = glance_offsets[name]
            target = {
                "head_pitch": offset["head_pitch"],
                "head_yaw": np.clip(progress_pose["head_yaw"] + offset["head_yaw"], -yaw_limit, yaw_limit),
            }
            hold_duration = glance_hold_duration
            previous_glance = name
            segments_until_glance = rng.integers(min_progress_segments, max_progress_segments + 1)
        else:
            target = progress_pose
            hold_duration = progress_hold_duration
            segments_until_glance -= 1

        segment_duration = transition_duration + hold_duration
        segments.append({"start": t, "duration": segment_duration, "target": target})
        t += segment_duration

    return {
        "pitch_index": joint_names.index("head_pitch"),
        "yaw_index": joint_names.index("head_yaw"),
        "initial": {"head_pitch": 0.0, "head_yaw": 0.0},
        "segments": segments,
        "progress_hold_duration": progress_hold_duration,
        "glance_hold_duration": glance_hold_duration,
        "transition_duration": transition_duration,
        "antenna_enabled": antenna_enabled,
        "antenna_pitch_reference": np.deg2rad(float(antenna_config.get("head_pitch_reference_deg", 12.0))),
        "antenna_back_angle": np.deg2rad(float(antenna_config.get("back_angle_deg", 22.0))),
        "left_antenna_index": joint_names.index("left_antenna") if antenna_enabled else None,
        "right_antenna_index": joint_names.index("right_antenna") if antenna_enabled else None,
    }


def compute_antenna_angle(head_motion, head_pitch):
    pitch_reference = max(head_motion["antenna_pitch_reference"], 1e-6)
    pitch_ratio = np.clip(head_pitch / pitch_reference, 0.0, 1.0)
    return pitch_ratio * head_motion["antenna_back_angle"]


def add_antenna_motion(head_motion, pose):
    if not head_motion["antenna_enabled"]:
        return pose

    pose = pose.copy()
    antenna_angle = compute_antenna_angle(head_motion, pose["head_pitch"])
    pose["left_antenna"] = antenna_angle
    pose["right_antenna"] = antenna_angle
    return pose


def apply_antenna_motion(joints_positions, head_motion):
    if head_motion is None or not head_motion["antenna_enabled"]:
        return joints_positions

    joints_positions = joints_positions.copy()
    antenna_angle = compute_antenna_angle(head_motion, joints_positions[head_motion["pitch_index"]])
    joints_positions[head_motion["left_antenna_index"]] = antenna_angle
    joints_positions[head_motion["right_antenna_index"]] = antenna_angle
    return joints_positions


def apply_antenna_motion_to_robot(robot, head_motion):
    if head_motion is None or not head_motion["antenna_enabled"]:
        return

    antenna_angle = compute_antenna_angle(head_motion, robot.get_joint("head_pitch"))
    robot.set_joint("left_antenna", antenna_angle)
    robot.set_joint("right_antenna", antenna_angle)
    robot.update_kinematics()


def sample_head_motion(head_motion, t):
    segments = head_motion["segments"]
    segment_index = len(segments) - 1
    for i, segment in enumerate(segments):
        if t < segment["start"] + segment["duration"]:
            segment_index = i
            break

    segment = segments[segment_index]
    segment_t = t - segment["start"]

    previous_pose = head_motion["initial"] if segment_index == 0 else segments[segment_index - 1]["target"]
    target_pose = segment["target"]

    if segment_t < head_motion["transition_duration"]:
        alpha = smoothstep(segment_t / head_motion["transition_duration"])
        pose = {
            joint: previous_pose[joint] + alpha * (target_pose[joint] - previous_pose[joint])
            for joint in ("head_pitch", "head_yaw")
        }
        return add_antenna_motion(head_motion, pose)

    return add_antenna_motion(head_motion, target_pose)


def apply_head_motion(joints_positions, head_motion, t):
    if head_motion is None:
        return joints_positions

    pose = sample_head_motion(head_motion, t)
    joints_positions = joints_positions.copy()
    joints_positions[head_motion["pitch_index"]] = pose["head_pitch"]
    joints_positions[head_motion["yaw_index"]] = pose["head_yaw"]
    if head_motion["antenna_enabled"]:
        joints_positions[head_motion["left_antenna_index"]] = pose["left_antenna"]
        joints_positions[head_motion["right_antenna_index"]] = pose["right_antenna"]
    return joints_positions


def apply_head_motion_to_robot(robot, head_motion, t):
    if head_motion is None:
        return

    pose = sample_head_motion(head_motion, t)
    robot.set_joint("head_pitch", pose["head_pitch"])
    robot.set_joint("head_yaw", pose["head_yaw"])
    if head_motion["antenna_enabled"]:
        robot.set_joint("left_antenna", pose["left_antenna"])
        robot.set_joint("right_antenna", pose["right_antenna"])
    robot.update_kinematics()


def make_frame(
    root_position,
    root_orientation_quat,
    joints_positions,
    left_toe_position,
    right_toe_position,
    world_linear_velocity,
    world_angular_velocity,
    joints_velocities,
    left_toe_linear_velocity,
    right_toe_linear_velocity,
    foot_contacts,
):
    return (
        root_position
        + root_orientation_quat
        + joints_positions
        + left_toe_position
        + right_toe_position
        + world_linear_velocity
        + world_angular_velocity
        + joints_velocities
        + left_toe_linear_velocity
        + right_toe_linear_velocity
        + foot_contacts
    )


def set_frame_info(
    episode,
    root_position,
    root_orientation_quat,
    joints_positions,
    left_toe_position,
    right_toe_position,
    world_linear_velocity,
    world_angular_velocity,
    joints_velocities,
    left_toe_linear_velocity,
    right_toe_linear_velocity,
    foot_contacts,
):
    offset_root_position = 0
    offset_root_orientation_quat = offset_root_position + len(root_position)
    offset_joints_positions = offset_root_orientation_quat + len(root_orientation_quat)
    offset_left_toe_position = offset_joints_positions + len(joints_positions)
    offset_right_toe_position = offset_left_toe_position + len(left_toe_position)
    offset_world_linear_velocity = offset_right_toe_position + len(right_toe_position)
    offset_world_angular_velocity = offset_world_linear_velocity + len(world_linear_velocity)
    offset_joints_velocities = offset_world_angular_velocity + len(world_angular_velocity)
    offset_left_toe_linear_velocity = offset_joints_velocities + len(joints_velocities)
    offset_right_toe_linear_velocity = offset_left_toe_linear_velocity + len(left_toe_linear_velocity)
    offset_foot_contacts = offset_right_toe_linear_velocity + len(right_toe_linear_velocity)

    episode["frame_offset"].append(
        {
            "root_position": offset_root_position,
            "root_orientation_quat": offset_root_orientation_quat,
            "joints_positions": offset_joints_positions,
            "left_toe_position": offset_left_toe_position,
            "right_toe_position": offset_right_toe_position,
            "world_linear_velocity": offset_world_linear_velocity,
            "world_angular_velocity": offset_world_angular_velocity,
            "joints_velocities": offset_joints_velocities,
            "left_toe_linear_velocity": offset_left_toe_linear_velocity,
            "right_toe_linear_velocity": offset_right_toe_linear_velocity,
            "foot_contacts": offset_foot_contacts,
        }
    )

    episode["frame_size"].append(
        {
            "root_position": len(root_position),
            "root_orientation_quat": len(root_orientation_quat),
            "joints_positions": len(joints_positions),
            "left_toe_position": len(left_toe_position),
            "right_toe_position": len(right_toe_position),
            "world_linear_velocity": len(world_linear_velocity),
            "world_angular_velocity": len(world_angular_velocity),
            "joints_velocities": len(joints_velocities),
            "left_toe_linear_velocity": len(left_toe_linear_velocity),
            "right_toe_linear_velocity": len(right_toe_linear_velocity),
            "foot_contacts": len(foot_contacts),
        }
    )


def set_gait_metadata(episode, gait_parameters, motion_engine, average_x, average_y, average_z):
    episode["x_linear_velocity"] = average_x
    episode["y_linear_velocity"] = average_y
    episode["z_angular_velocity"] = average_z

    episode["gait_parameters"] = {
        "dx": gait_parameters["dx"],
        "dy": gait_parameters["dy"],
        "dth": gait_parameters["dth"],
        "duration": gait_parameters["duration"],
        "hardware": gait_parameters["hardware"],
        "trunk_mode": motion_engine.robot_parameters.trunk_mode,
        "double_support_ratio": motion_engine.robot_parameters.double_support_ratio,
        "startend_double_support_ratio": motion_engine.robot_parameters.startend_double_support_ratio,
        "planned_timesteps": motion_engine.robot_parameters.planned_timesteps,
        "replan_timesteps": gait_parameters["replan_timesteps"],
        "walk_com_height": motion_engine.robot_parameters.walk_com_height,
        "walk_foot_height": motion_engine.robot_parameters.walk_foot_height,
        "walk_trunk_pitch": np.rad2deg(motion_engine.robot_parameters.walk_trunk_pitch),
        "walk_foot_rise_ratio": motion_engine.robot_parameters.walk_foot_rise_ratio,
        "single_support_duration": motion_engine.robot_parameters.single_support_duration,
        "single_support_timesteps": motion_engine.robot_parameters.single_support_timesteps,
        "foot_length": motion_engine.robot_parameters.foot_length,
        "feet_spacing": motion_engine.robot_parameters.feet_spacing,
        "zmp_margin": motion_engine.robot_parameters.zmp_margin,
        "foot_zmp_target_x": motion_engine.robot_parameters.foot_zmp_target_x,
        "foot_zmp_target_y": motion_engine.robot_parameters.foot_zmp_target_y,
        "walk_max_dtheta": motion_engine.robot_parameters.walk_max_dtheta,
        "walk_max_dy": motion_engine.robot_parameters.walk_max_dy,
        "walk_max_dx_forward": motion_engine.robot_parameters.walk_max_dx_forward,
        "walk_max_dx_backward": motion_engine.robot_parameters.walk_max_dx_backward,
        "average_x_linear_velocity": average_x,
        "average_y_linear_velocity": average_y,
        "average_z_angular_velocity": average_z,
        "period": motion_engine.period,
    }
    if "head_motion" in gait_parameters:
        episode["gait_parameters"]["head_motion"] = gait_parameters["head_motion"]
    if "neck_step_motion" in gait_parameters:
        episode["gait_parameters"]["neck_step_motion"] = gait_parameters["neck_step_motion"]


def write_episode(args, episode):
    file_name = f"{args.index}.json"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", f"{args.robot}", file_name)

    with open(file_path, "w") as f:
        json.encoder.c_make_encoder = None
        json.encoder.float = RoundingFloat
        json.dump(episode, f, indent=4)


def main(args):
    ISAACSIM_FPS = 60
    MESHCAT_FPS = 20

    episode = {
        "fps": ISAACSIM_FPS,
        "frame_duration": np.around(1 / ISAACSIM_FPS, 4),
        "enable_cycle_offset_position": True,
        "enable_cycle_offset_rotation": False,
        "motion_weight": 1,
        "x_linear_velocity": [],
        "y_linear_velocity": [],
        "z_angular_velocity": [],
        "gait_parameters": [],
        "joints": [],
        "frame_offset": [],
        "frame_size": [],
        "frames": [],
    }

    is_stand = args.stand.lower().strip() in ("true", "1", "yes", "y", "on")
    episode["enable_cycle_offset_position"] = not is_stand

    # Load gait config
    gait_config_file = f"../../config/{args.robot}/gait.json"
    gait_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), gait_config_file)

    if not os.path.isfile(gait_config_file_path):
        raise FileNotFoundError(f"Gait config file not found: {gait_config_file_path}")

    with open(gait_config_file_path, "r") as f:
        gait_parameters = json.load(f)

    gait_parameters["dx"] = 0.0 if is_stand else args.dx
    gait_parameters["dy"] = 0.0 if is_stand else args.dy
    gait_parameters["dth"] = 0.0 if is_stand else args.dth

    gait_parameters["robot"] = args.robot

    # Set robot path
    robot_folder = f"../../robots/{args.robot}"
    robot_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), robot_folder)

    # Run motion engine
    motion_engine = MotionEngine(robot_folder_path, gait_parameters)

    joint_names = list(motion_engine.get_angles().keys())
    first_joints_positions = list(motion_engine.get_angles().values())
    first_T_world_fbase = np.array(motion_engine.robot.get_T_world_fbase(), copy=True)
    first_T_world_left_foot = np.array(motion_engine.robot.get_T_world_left(), copy=True)
    first_T_world_right_foot = np.array(motion_engine.robot.get_T_world_right(), copy=True)

    motion_engine.set_trajectory(gait_parameters["dx"], gait_parameters["dy"], gait_parameters["dth"])

    duration = gait_parameters.get("duration", 10.0)
    num_frames = int(round(duration * ISAACSIM_FPS))
    head_motion = make_head_motion(gait_parameters, joint_names, duration) if not is_stand else None
    neck_step_motion = make_neck_step_motion(gait_parameters, joint_names) if not is_stand else None

    if is_stand:
        root_position = list(first_T_world_fbase[:3, 3])
        root_orientation_quat = list(R.from_matrix(first_T_world_fbase[:3, :3]).as_quat())

        T_body_left_foot = np.linalg.inv(first_T_world_fbase) @ first_T_world_left_foot
        T_body_right_foot = np.linalg.inv(first_T_world_fbase) @ first_T_world_right_foot

        left_toe_position = list(T_body_left_foot[:3, 3])
        right_toe_position = list(T_body_right_foot[:3, 3])
        world_linear_velocity = [0.0, 0.0, 0.0]
        world_angular_velocity = [0.0, 0.0, 0.0]
        joints_velocities = [0.0] * len(first_joints_positions)
        left_toe_linear_velocity = [0.0, 0.0, 0.0]
        right_toe_linear_velocity = [0.0, 0.0, 0.0]
        foot_contacts = [1, 1]

        frame = make_frame(
            root_position,
            root_orientation_quat,
            first_joints_positions,
            left_toe_position,
            right_toe_position,
            world_linear_velocity,
            world_angular_velocity,
            joints_velocities,
            left_toe_linear_velocity,
            right_toe_linear_velocity,
            foot_contacts,
        )
        episode["frames"] = [frame.copy() for _ in range(num_frames)]
        episode["joints"] = joint_names
        set_frame_info(
            episode,
            root_position,
            root_orientation_quat,
            first_joints_positions,
            left_toe_position,
            right_toe_position,
            world_linear_velocity,
            world_angular_velocity,
            joints_velocities,
            left_toe_linear_velocity,
            right_toe_linear_velocity,
            foot_contacts,
        )
        set_gait_metadata(episode, gait_parameters, motion_engine, 0.0, 0.0, 0.0)

        print("computed x velocity: 0.0, mean average x velocity: 0.0")
        print("computed y velocity: 0.0, mean average y velocity: 0.0")
        print("computed th velocity: 0.0, mean average th velocity: 0.0")

        write_episode(args, episode)
        return

    viz = robot_viz(motion_engine.robot)
    #threading.Timer(1.0, open_browser).start()

    dt = 0.001
    
    start = time.time()

    skip_warmup = 0.0 # TODO

    last_record = 0.0
    last_meshcat_display = 0.0
    
    prev_root_position = [0.0, 0.0, 0.0]
    prev_root_orientation_quat = None
    prev_root_orientation_euler = [0.0, 0.0, 0.0]
    prev_left_toe_position = [0.0, 0.0, 0.0]
    prev_right_toe_position = [0.0, 0.0, 0.0]
    prev_joints_positions = None

    i = 0

    is_prev_initialized = False
    is_added_frame_info = False

    average_x_linear_velocity = []
    average_y_linear_velocity = []
    average_z_angular_velocity = []

    while True:
        motion_engine.tick(dt)
        if motion_engine.t <= 0 + skip_warmup:
            start = motion_engine.t
            last_record = motion_engine.t + 1 / ISAACSIM_FPS
            last_meshcat_display = motion_engine.t + 1 / MESHCAT_FPS
            continue

        if motion_engine.t - last_record >= 1 / ISAACSIM_FPS:
            T_world_fbase = motion_engine.robot.get_T_world_fbase()
            
            root_position = list(T_world_fbase[:3, 3])
            root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())

            joints_positions = list(motion_engine.get_angles().values())
            T_world_left_foot = motion_engine.robot.get_T_world_left()
            T_world_right_foot = motion_engine.robot.get_T_world_right()

            T_body_left_foot = np.linalg.inv(T_world_fbase) @ T_world_left_foot
            T_body_right_foot = np.linalg.inv(T_world_fbase) @ T_world_right_foot

            left_toe_position = list(T_body_left_foot[:3, 3])
            right_toe_position = list(T_body_right_foot[:3, 3])

            if not is_prev_initialized:
                prev_root_position = root_position.copy()
                prev_root_orientation_euler = (R.from_quat(root_orientation_quat).as_euler("xyz").copy())
                prev_left_toe_position = left_toe_position.copy()
                prev_right_toe_position = right_toe_position.copy()
                prev_joints_positions = joints_positions.copy()
                is_prev_initialized = True

            world_linear_velocity = list((np.array(root_position) - np.array(prev_root_position)) / (1 / ISAACSIM_FPS))
            world_angular_velocity = compute_angular_velocity(root_orientation_quat, prev_root_orientation_quat, (1 / ISAACSIM_FPS))

            average_x_linear_velocity.append(world_linear_velocity[0])
            average_y_linear_velocity.append(world_linear_velocity[1])
            average_z_angular_velocity.append(world_angular_velocity[2])

            body_rotation_matrix = T_world_fbase[:3, :3]
            body_linear_velocity = list(body_rotation_matrix @ world_linear_velocity)
            body_angular_velocity = list(body_rotation_matrix.T @ world_angular_velocity)

            joints_velocities = list((np.array(joints_positions) - np.array(prev_joints_positions)) / (1 / ISAACSIM_FPS))

            left_toe_linear_velocity = list((np.array(left_toe_position) - np.array(prev_left_toe_position)) / (1 / ISAACSIM_FPS))
            right_toe_linear_velocity = list((np.array(right_toe_position) - np.array(prev_right_toe_position)) / (1 / ISAACSIM_FPS))

            foot_contacts = motion_engine.get_current_support_phase()

            if is_prev_initialized:
                frame_t = len(episode["frames"]) / ISAACSIM_FPS
                update_neck_step_motion(neck_step_motion, foot_contacts, frame_t)
                joints_positions = apply_neck_step_motion(joints_positions, neck_step_motion, frame_t)
                joints_positions = apply_head_motion(joints_positions, head_motion, frame_t)
                joints_positions = apply_neck_counter_head_motion(joints_positions, neck_step_motion, frame_t)
                joints_positions = apply_antenna_motion(joints_positions, head_motion)
                joints_velocities = list((np.array(joints_positions) - np.array(prev_joints_positions)) / (1 / ISAACSIM_FPS))

                episode["frames"].append(
                    make_frame(
                        root_position,
                        root_orientation_quat,
                        joints_positions,
                        left_toe_position,
                        right_toe_position,
                        world_linear_velocity,
                        world_angular_velocity,
                        joints_velocities,
                        left_toe_linear_velocity,
                        right_toe_linear_velocity,
                        foot_contacts,
                    )
                )

                if not is_added_frame_info:
                    set_frame_info(
                        episode,
                        root_position,
                        root_orientation_quat,
                        joints_positions,
                        left_toe_position,
                        right_toe_position,
                        world_linear_velocity,
                        world_angular_velocity,
                        joints_velocities,
                        left_toe_linear_velocity,
                        right_toe_linear_velocity,
                        foot_contacts,
                    )

                    episode["joints"] = joint_names

                    is_added_frame_info = True

            last_record = motion_engine.t

            prev_root_position = root_position.copy()
            prev_root_orientation_quat = root_orientation_quat.copy()
            prev_root_orientation_euler = (R.from_quat(root_orientation_quat).as_euler("xyz").copy())
            prev_left_toe_position = left_toe_position.copy()
            prev_right_toe_position = right_toe_position.copy()
            prev_joints_positions = joints_positions.copy()

            is_prev_initialized = True

        if motion_engine.t - last_meshcat_display >= 1 / MESHCAT_FPS:
            last_meshcat_display = motion_engine.t
            apply_head_motion_to_robot(motion_engine.robot, head_motion, motion_engine.t)
            apply_neck_step_motion_to_robot(motion_engine.robot, neck_step_motion, motion_engine.t)
            apply_neck_counter_head_motion_to_robot(motion_engine.robot, neck_step_motion, motion_engine.t)
            apply_antenna_motion_to_robot(motion_engine.robot, head_motion)
            viz.display(motion_engine.robot.state.q)

            robot_frame_viz(motion_engine.robot, "trunk")
            robot_frame_viz(motion_engine.robot, "left_foot")
            robot_frame_viz(motion_engine.robot, "right_foot")

            footsteps_viz(motion_engine.get_supports())

        if len(episode["frames"]) >= num_frames:
            break

        i += 1

    mean_average_x_linear_velocity = np.around(np.mean(average_x_linear_velocity[120:]), 4)
    mean_average_y_linear_velocity = np.around(np.mean(average_y_linear_velocity[120:]), 4)
    mean_average_z_angular_velocity = np.around(np.mean(average_z_angular_velocity[120:]), 4)

    set_gait_metadata(
        episode,
        gait_parameters,
        motion_engine,
        mean_average_x_linear_velocity,
        mean_average_y_linear_velocity,
        mean_average_z_angular_velocity,
    )

    x_velocity = np.around(gait_parameters["dx"] * 2 / motion_engine.period, 3)
    y_velocity = np.around(gait_parameters["dy"] * 2 / motion_engine.period, 3)
    th_velocity = np.around(gait_parameters["dth"] * 2 / motion_engine.period, 3)

    print(f"computed x velocity: {x_velocity}, mean average x velocity: {mean_average_x_linear_velocity}")
    print(f"computed y velocity: {y_velocity}, mean average y velocity: {mean_average_y_linear_velocity}")
    print(f"computed th velocity: {th_velocity}, mean average th velocity: {mean_average_z_angular_velocity}")

    write_episode(args, episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", type=int, required=True)

    parser.add_argument("--robot", type=str, required=True, choices=["bdx", "olaf"])

    parser.add_argument("--dx", type=float, required=True)

    parser.add_argument("--dy", type=float, required=True)

    parser.add_argument("--dth", type=float, required=True)

    parser.add_argument("--stand", type=str, required=True)

    args = parser.parse_args()

    main(args)
