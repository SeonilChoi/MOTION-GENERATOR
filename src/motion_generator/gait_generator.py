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

    # Load gait config
    gait_config_file = f"../../config/{args.robot}/gait.json"
    gait_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), gait_config_file)

    if not os.path.isfile(gait_config_file_path):
        raise FileNotFoundError(f"Gait config file not found: {gait_config_file_path}")

    with open(gait_config_file_path, "r") as f:
        gait_parameters = json.load(f)

    gait_parameters["dx"] = args.dx
    gait_parameters["dy"] = args.dy
    gait_parameters["dth"] = args.dth

    gait_parameters["robot"] = args.robot

    # Set robot path
    robot_folder = f"../../robots/{args.robot}"
    robot_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), robot_folder)

    # Run motion engine
    motion_engine = MotionEngine(robot_folder_path, gait_parameters)

    first_joints_positions = list(motion_engine.get_angles().values())
    first_T_world_fbase = motion_engine.robot.get_T_world_fbase()
    first_T_world_left_foot = motion_engine.robot.get_T_world_left()
    first_T_world_right_foot = motion_engine.robot.get_T_world_right()

    motion_engine.set_trajectory(gait_parameters["dx"], gait_parameters["dy"], gait_parameters["dth"])

    viz = robot_viz(motion_engine.robot)
    #threading.Timer(1.0, open_browser).start()

    dt = 0.001
    duration = gait_parameters.get("duration", 10.0)
    
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
            if is_stand:
                T_world_fbase = first_T_world_fbase
            else:
                T_world_fbase = motion_engine.robot.get_T_world_fbase()
            
            root_position = list(T_world_fbase[:3, 3])
            root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())

            if is_stand:
                joints_positions = first_joints_positions
                T_world_left_foot = first_T_world_left_foot
                T_world_right_foot = first_T_world_right_foot
            else:
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
                episode["frames"].append(
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

                if not is_added_frame_info:
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

                    episode["joints"] = list(motion_engine.get_angles().keys())

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
            viz.display(motion_engine.robot.state.q)

            robot_frame_viz(motion_engine.robot, "trunk")
            robot_frame_viz(motion_engine.robot, "left_foot")
            robot_frame_viz(motion_engine.robot, "right_foot")

            footsteps_viz(motion_engine.get_supports())

        if len(episode["frames"]) == duration * ISAACSIM_FPS:
            break

        i += 1

    mean_average_x_linear_velocity = np.around(np.mean(average_x_linear_velocity[120:]), 4)
    mean_average_y_linear_velocity = np.around(np.mean(average_y_linear_velocity[120:]), 4)
    mean_average_z_angular_velocity = np.around(np.mean(average_z_angular_velocity[120:]), 4)

    episode["x_linear_velocity"] = mean_average_x_linear_velocity
    episode["y_linear_velocity"] = mean_average_y_linear_velocity
    episode["z_angular_velocity"] = mean_average_z_angular_velocity

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
        "average_x_linear_velocity": mean_average_x_linear_velocity,
        "average_y_linear_velocity": mean_average_y_linear_velocity,
        "average_z_angular_velocity": mean_average_z_angular_velocity,
        "period": motion_engine.period,
    }

    x_velocity = np.around(gait_parameters["dx"] * 2 / motion_engine.period, 3)
    y_velocity = np.around(gait_parameters["dy"] * 2 / motion_engine.period, 3)
    th_velocity = np.around(gait_parameters["dth"] * 2 / motion_engine.period, 3)

    print(f"computed x velocity: {x_velocity}, mean average x velocity: {mean_average_x_linear_velocity}")
    print(f"computed y velocity: {y_velocity}, mean average y velocity: {mean_average_y_linear_velocity}")
    print(f"computed th velocity: {th_velocity}, mean average th velocity: {mean_average_z_angular_velocity}")

    file_name = f"{args.index}.json"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", f"{args.robot}", file_name)

    with open(file_path, "w") as f:
        json.encoder.c_make_encoder = None
        json.encoder.float = RoundingFloat
        json.dump(episode, f, indent=4)


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
