import os
from typing import List
import numpy as np
import dataclasses
import json
import jsonlines
from termcolor import colored, cprint
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import torch.multiprocessing as mp

import tyro
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class

from pointact.robot_envs.rlbench_utils.environments import (
    CAMERA_ATTR, Mover, RLBenchEnv,
    get_robot_joints_pose_size
)
from pointact.robot_envs.rlbench_utils.eval_utils import (
    project_action_chunks_on_image, save_images_to_video,
    set_random_seed, write_to_file,
    get_rlbench_robot_workspace
)

from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient


CAMERA_NAMES = ("left_shoulder", "right_shoulder", "wrist", "front")


def _as_float_list(value):
    return np.asarray(value, dtype=np.float64).tolist()


def _quaternion_error_degrees(target_xyzw, actual_xyzw):
    target = np.asarray(target_xyzw, dtype=np.float64)
    actual = np.asarray(actual_xyzw, dtype=np.float64)
    denom = np.linalg.norm(target) * np.linalg.norm(actual)
    if denom == 0:
        return None
    cosine = np.clip(abs(float(np.dot(target, actual))) / denom, 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(cosine)))


class WaterPlantsGeometryTracker:
    """Collect exact task geometry at every simulator tick of a policy action."""

    def __init__(self, task):
        self.task = task
        self.core = task._task
        self.head = self.core.head
        self.waterer = self.core.waterer
        self.pour_sensor = self.core.pour_point_reached._detector
        self.success_sensor = self.core.success_sensor
        self.tip = task._scene.robot.arm.get_tip()
        self.gripper = task._scene.robot.gripper
        self.active = None
        self.episode_id = None
        self.episode_records = []
        self.sim_tick = 0

    def start_episode(self, episode_id):
        self.episode_id = int(episode_id)
        self.episode_records = []
        self.sim_tick = 0
        self.active = None

    def _snapshot(self, check_drop_detection=True):
        head_pose = np.asarray(self.head.get_pose(), dtype=np.float64)
        waterer_pose = np.asarray(self.waterer.get_pose(), dtype=np.float64)
        pour_pose = np.asarray(self.pour_sensor.get_pose(), dtype=np.float64)
        success_pose = np.asarray(self.success_sensor.get_pose(), dtype=np.float64)
        tip_pose = np.asarray(self.tip.get_pose(), dtype=np.float64)
        drops = list(self.core.drops)
        detected = (
            [bool(self.success_sensor.is_detected(drop)) for drop in drops]
            if check_drop_detection
            else []
        )
        drop_poses = [np.asarray(drop.get_pose(), dtype=np.float64) for drop in drops]
        grasped_handles = {obj.get_handle() for obj in self.gripper.get_grasped_objects()}
        return {
            "sim_tick": self.sim_tick,
            "head_pose_world": _as_float_list(head_pose),
            "waterer_pose_world": _as_float_list(waterer_pose),
            "pour_point_pose_world": _as_float_list(pour_pose),
            "success_sensor_pose_world": _as_float_list(success_pose),
            "eef_pose_world": _as_float_list(tip_pose),
            "head_position_in_pour_frame": _as_float_list(
                self.head.get_position(relative_to=self.pour_sensor)
            ),
            "head_pose_in_waterer_frame": _as_float_list(
                self.head.get_pose(relative_to=self.waterer)
            ),
            "waterer_pose_in_eef_frame": _as_float_list(
                self.waterer.get_pose(relative_to=self.tip)
            ),
            "head_to_pour_center_m": float(np.linalg.norm(head_pose[:3] - pour_pose[:3])),
            # WaterPlants.step() has already evaluated the exact DetectedCondition
            # before the Scene callback runs. Reading the persistent task flag
            # preserves that authoritative event even if the head has exited the
            # sensor volume by the time this snapshot is taken.
            "head_detected_by_pour_sensor": bool(self.core.reachedOnce),
            "task_reached": bool(self.core.reached),
            "task_reached_once": bool(self.core.reachedOnce),
            "waterer_grasped": self.waterer.get_handle() in grasped_handles,
            "drop_count": len(drops),
            "drop_poses_world": [_as_float_list(pose) for pose in drop_poses],
            "drops_detected_by_success_sensor": detected,
            "num_drops_detected": int(sum(detected)) if check_drop_detection else None,
            "drop_to_success_center_m": [
                float(np.linalg.norm(pose[:3] - success_pose[:3]))
                for pose in drop_poses
            ],
        }

    def start_action(self, step_id, commanded_action, observed_eef_before):
        before = self._snapshot(check_drop_detection=False)
        self.active = {
            "episode": self.episode_id,
            "step": int(step_id),
            "commanded_action": _as_float_list(commanded_action),
            "observed_eef_before": _as_float_list(observed_eef_before),
            "before": before,
            "sim_steps": 0,
            "head_detected_any": before["head_detected_by_pour_sensor"],
            "first_head_detected_sim_tick": (
                before["sim_tick"] if before["head_detected_by_pour_sensor"] else None
            ),
            "first_head_detected_snapshot": (
                before if before["head_detected_by_pour_sensor"] else None
            ),
            "drops_spawned_any": before["drop_count"] > 0,
            "first_drops_spawned_sim_tick": (
                before["sim_tick"] if before["drop_count"] > 0 else None
            ),
            "max_drops_detected": 0,
            "all_five_drops_detected_any": False,
            "first_all_five_drops_detected_sim_tick": None,
            "min_head_to_pour_center_m": before["head_to_pour_center_m"],
            "min_head_to_pour_snapshot": before,
            "waterer_grasped_all_ticks": before["waterer_grasped"],
            "waterer_grasped_any_tick": before["waterer_grasped"],
            "waterer_lost_after_grasp": False,
        }

    def simulator_step(self):
        self.sim_tick += 1
        if self.active is None:
            return
        snapshot = self._snapshot(check_drop_detection=False)
        record = self.active
        record["sim_steps"] += 1
        if snapshot["head_to_pour_center_m"] < record["min_head_to_pour_center_m"]:
            record["min_head_to_pour_center_m"] = snapshot["head_to_pour_center_m"]
            record["min_head_to_pour_snapshot"] = snapshot
        if snapshot["head_detected_by_pour_sensor"]:
            record["head_detected_any"] = True
            if record["first_head_detected_sim_tick"] is None:
                record["first_head_detected_sim_tick"] = self.sim_tick
                record["first_head_detected_snapshot"] = snapshot
        if snapshot["drop_count"] > 0:
            record["drops_spawned_any"] = True
            if record["first_drops_spawned_sim_tick"] is None:
                record["first_drops_spawned_sim_tick"] = self.sim_tick
        if snapshot["waterer_grasped"]:
            record["waterer_grasped_any_tick"] = True
        elif record["waterer_grasped_any_tick"]:
            record["waterer_lost_after_grasp"] = True
        record["waterer_grasped_all_ticks"] &= snapshot["waterer_grasped"]

    def finish_action(self, observed_eef_after, reward, terminate, error=None):
        if self.active is None:
            return None
        after = self._snapshot()
        record = self.active
        record["after"] = after
        record["observed_eef_after"] = (
            None if observed_eef_after is None else _as_float_list(observed_eef_after)
        )
        record["reward"] = float(reward)
        record["terminate"] = bool(terminate)
        record["error"] = error
        if after["task_reached_once"]:
            record["head_detected_any"] = True
            if record["first_head_detected_sim_tick"] is None:
                record["first_head_detected_sim_tick"] = self.sim_tick
                record["first_head_detected_snapshot"] = after
        if after["drop_count"] > 0:
            record["drops_spawned_any"] = True
            if record["first_drops_spawned_sim_tick"] is None:
                record["first_drops_spawned_sim_tick"] = self.sim_tick
        record["max_drops_detected"] = after["num_drops_detected"]
        record["all_five_drops_detected_any"] = after["num_drops_detected"] == 5
        record["first_all_five_drops_detected_sim_tick"] = (
            self.sim_tick if after["num_drops_detected"] == 5 else None
        )
        command = np.asarray(record["commanded_action"], dtype=np.float64)
        if observed_eef_after is not None:
            actual = np.asarray(observed_eef_after, dtype=np.float64)
            record["eef_target_position_error_m"] = float(
                np.linalg.norm(command[:3] - actual[:3])
            )
            record["eef_target_rotation_error_deg"] = _quaternion_error_degrees(
                command[3:7], actual[3:7]
            )
        else:
            record["eef_target_position_error_m"] = None
            record["eef_target_rotation_error_deg"] = None
        self.episode_records.append(record)
        self.active = None
        return record

    def episode_summary(self, success):
        records = self.episode_records
        reached_records = [record for record in records if record["head_detected_any"]]
        spawned_records = [record for record in records if record["drops_spawned_any"]]
        all_drops_records = [
            record for record in records if record["all_five_drops_detected_any"]
        ]
        min_record = min(
            records,
            key=lambda record: record["min_head_to_pour_center_m"],
            default=None,
        )
        max_drop_record = max(
            records,
            key=lambda record: record["max_drops_detected"],
            default=None,
        )
        return {
            "episode": self.episode_id,
            "success": bool(success),
            "success_without_pour": bool(success and not reached_records),
            "policy_steps": len(records),
            "pour_sensor_triggered": bool(reached_records),
            "first_pour_sensor_step": reached_records[0]["step"] if reached_records else None,
            "drops_spawned": bool(spawned_records),
            "first_drops_spawned_step": spawned_records[0]["step"] if spawned_records else None,
            "all_five_drops_detected": bool(all_drops_records),
            "first_all_five_drops_detected_step": (
                all_drops_records[0]["step"] if all_drops_records else None
            ),
            "max_drops_detected": (
                max_drop_record["max_drops_detected"] if max_drop_record else 0
            ),
            "min_head_to_pour_center_m": (
                min_record["min_head_to_pour_center_m"] if min_record else None
            ),
            "min_head_to_pour_step": min_record["step"] if min_record else None,
            "waterer_lost_after_grasp": any(
                record["waterer_lost_after_grasp"] for record in records
            ),
            "max_eef_target_position_error_m": max(
                (
                    record["eef_target_position_error_m"]
                    for record in records
                    if record["eef_target_position_error_m"] is not None
                ),
                default=None,
            ),
            "max_eef_target_rotation_error_deg": max(
                (
                    record["eef_target_rotation_error_deg"]
                    for record in records
                    if record["eef_target_rotation_error_deg"] is not None
                ),
                default=None,
            ),
        }


@dataclasses.dataclass
class ClientArgs:
    seed: int = 7  # Random Seed (for reproducibility)

    # rlbench
    # test_level: str = "l1"  # l1, l2, l3, l4
    taskvar: str = ""
    taskvar_file: str = ""
    num_episodes: int = 25  # Number of episodes per taskvar
    max_steps: int = 25  # Maximum number of steps per episode
    image_size: int = 256
    microstep_data_dir: str = ""
    select_cameras: List[str] = ("front", "left_shoulder", "right_shoulder", "wrist")

    no_env_reward: bool = False
    stop_with_policy: bool = False

    # model
    repo_id: str = ""  # post-process: norm and unnorm state/action
    pretrained_path: str = ""
    pred_rot_type: str = "euler" # euler, quat, rot6d
    delta_action: bool = False

    replan_steps: int = 8
    clip_within_workspace: bool = False
    action_ensemble: bool = False
    reset_policy_each_episode: bool = False
    reset_environment_rng_each_episode: bool = False

    # point cloud
    remove_arm: bool = False

    # server-client
    num_workers: int = 1
    host: str = "localhost"
    port: int = 15000

    # output
    save_video: bool = False
    continuous_video_fps: float = 0.0
    save_obs_outs: bool = False
    project_action_on_image: bool = False
    save_dir: str = ""

    verbose: bool = False


def save_obs_state(output_dir, obs_id, obs_state_dict):
    np.savez(os.path.join(output_dir, f"obs_{obs_id:06d}.npz"), **obs_state_dict)

def write_transition(output_dir, transition):
    with jsonlines.open(os.path.join(output_dir, "transitions.jsonl"), "a", flush=True) as outf:
        outf.write(transition)


def rgb_to_uint8(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = image * 255.0 if image.size and image.max() <= 1.0 else image
    return np.clip(image, 0, 255).astype(np.uint8)


def observation_video_frame(obs_state_dict, select_cameras):
    images = []
    for cam_name in select_cameras:
        cam_idx = CAMERA_NAMES.index(cam_name)
        images.append(rgb_to_uint8(obs_state_dict["rgb"][cam_idx]))
    return np.concatenate(images, 1)


def is_action_similar(prev_action, action):
    pos_delta = prev_action[:3] - action[:3]
    prev_quat = prev_action[3:7]
    curr_quat = action[3:7]
    quat_norm = float(np.dot(prev_quat, prev_quat) * np.dot(curr_quat, curr_quat))
    quat_dot = float(np.dot(prev_quat, curr_quat))
    if (
        np.dot(pos_delta, pos_delta) < 1e-3
        and quat_norm > 0.0
        and (quat_dot * quat_dot / quat_norm) > 0.999
        and (prev_action[7] > 0.5) == (action[7] > 0.5)
    ):
        return True
    return False


def producer_fn(proc_id, args, taskvar, pred_file, producer_queue):

    policy_client = PolicyClient(args.host, args.port)

    is_server_running = False
    while not is_server_running:
        is_server_running = policy_client.ping()
    print(f"Server is running on host {args.host} port {args.port}")

    set_random_seed(args.seed)

    # Initialize RLBench env
    task_str, variation_id = taskvar.split("+")
    variation_id = int(variation_id)

    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_depth=True,
        apply_pc=True,
        apply_mask=False,
        apply_cameras=CAMERA_NAMES,
        headless=True,
        image_size=[args.image_size, args.image_size],
        cam_rand_factor=0,
        cam_params_to_opencv=True,
        use_metric_depth=True,
    )

    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task.set_variation(variation_id)
    move = Mover(task, max_tries=10)

    continuous_images = []
    continuous_callback_step = 0
    continuous_recording = False
    continuous_frame_stride = 1
    continuous_actual_fps = 0.0

    if args.continuous_video_fps > 0:
        simulation_dt = task._scene.pyrep.get_simulation_timestep()
        continuous_frame_stride = max(
            1, round(1.0 / (args.continuous_video_fps * simulation_dt))
        )
        continuous_actual_fps = 1.0 / (
            continuous_frame_stride * simulation_dt
        )

        def take_continuous_snap():
            nonlocal continuous_callback_step
            if not continuous_recording:
                return
            continuous_callback_step += 1
            if continuous_callback_step % continuous_frame_stride != 0:
                return

            images = []
            for cam_name in args.select_cameras:
                sensor = getattr(task._scene, CAMERA_ATTR[cam_name])
                sensor.handle_explicitly()
                images.append(rgb_to_uint8(sensor.capture_rgb()))
            continuous_images.append(np.concatenate(images, 1))

        task._scene.register_step_callback(take_continuous_snap)
        print(
            "continuous video",
            f"requested={args.continuous_video_fps:g}Hz",
            f"actual={continuous_actual_fps:g}Hz",
            f"simulation_dt={simulation_dt:g}s",
            f"stride={continuous_frame_stride}",
        )

    if task_str != "water_plants":
        raise ValueError(
            "run_water_plants_geometry_client.py only supports water_plants"
        )
    geometry_tracker = None

    if args.microstep_data_dir != "":
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation_id}", "episodes")
        demos = []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                try:
                    demo = env.get_demo(task_str, variation_id, idx, load_images=False)
                    demos.append(demo)
                except Exception as e:
                    print("\tProblem to load demo_id:", idx, ep)
                    print(e)
        num_episodes = len(demos)
    else:
        demos = None
        num_episodes = args.num_episodes

    success_rate = 0
    for episode_id in tqdm(range(num_episodes)):
        replay_images = []
        continuous_recording = False
        continuous_images.clear()
        continuous_callback_step = 0

        if args.reset_environment_rng_each_episode:
            set_random_seed(args.seed + episode_id)

        if demos is None:
            instructions, obs = task.reset()
        else:
            print("Resetting to demo", episode_id)
            instructions, obs = task.reset_to_demo(demos[episode_id])  # type: ignore

        if geometry_tracker is None:
            geometry_tracker = WaterPlantsGeometryTracker(task)
            existing_step_callback = task._scene._step_callback

            def geometry_and_existing_step_callback():
                if existing_step_callback is not None:
                    existing_step_callback()
                geometry_tracker.simulator_step()

            task._scene.register_step_callback(geometry_and_existing_step_callback)
        geometry_tracker.start_episode(episode_id)

        if args.reset_policy_each_episode:
            policy_client.reset(
                options={"seed": args.seed + episode_id, "episode_id": episode_id}
            )

        # instruction = random.choice(instructions)
        instruction = instructions[0]
        action_plan = []

        obs_state_dict = env.get_observation(obs)
        move.reset(obs_state_dict["gripper"])
        if args.continuous_video_fps > 0:
            continuous_images.append(
                observation_video_frame(obs_state_dict, args.select_cameras)
            )
            continuous_recording = True

        if args.save_obs_outs:
            output_dir = os.path.join(args.save_dir, "obs_outs", taskvar, f"episode_{episode_id:06d}")
            replans_dir = os.path.join(output_dir, "replans")
            os.makedirs(replans_dir, exist_ok=True)
            open(os.path.join(output_dir, "transitions.jsonl"), "w").close()
            with open(os.path.join(output_dir, "meta.json"), "w") as outf:
                json.dump(
                    {
                        "taskvar": taskvar,
                        "task": task_str,
                        "variation": variation_id,
                        "episode_id": episode_id,
                        "instruction": instruction,
                        "seed": args.seed,
                        "repo_id": args.repo_id,
                        "replan_steps": args.replan_steps,
                        "pred_rot_type": args.pred_rot_type,
                        "select_cameras": list(args.select_cameras),
                    },
                    outf,
                    indent=2,
                )
            save_obs_state(output_dir, 0, obs_state_dict)

        prev_action = None
        current_replan_file = None
        current_replan_step = None
        has_reward_accumulated = False
        for step_id in range(args.max_steps):
            state = obs_state_dict["gripper"]
            # convert rotation: quat to euler/rot6d/axisangle
            if args.pred_rot_type in ["euler", "rot6d", "axisangle"]:
                rot = convert_rotation(
                    state[3:7], "quat", args.pred_rot_type, quat_order_src="xyzw", euler_order_dst="xyz"
                )
                state = np.concatenate([state[:3], rot, state[7:]], 0)

            # if not action_plan:
            if step_id % args.replan_steps == 0:
                batch = {
                    "observation.state": [state],
                    "task": [instruction],
                    "repo_id": [args.repo_id],
                }
                if args.remove_arm:
                    batch["observation.robot_joints_bbox"] = [
                        get_robot_joints_pose_size(obs_state_dict["arm_links_info"])
                    ]

                if args.save_video:
                    replay_images.append([])
                for cam_name in args.select_cameras:
                    cam_idx = CAMERA_NAMES.index(cam_name)
                    rgb_image = obs_state_dict["rgb"][cam_idx]
                    batch[f"observation.images.{cam_name}_image"] = [rgb_image]
                    if args.save_video:
                        replay_images[-1].append(np.array(rgb_image))

                    # batch[f"observation.depths.{cam_name}"] = [obs_state_dict["depth"][cam_idx]]
                    # batch[f"observation.camera_extrinsics.{cam_name}"] = [obs_state_dict["camera_extrinsics"][cam_name]]
                    # batch[f"observation.camera_intrinsics.{cam_name}"] = [obs_state_dict["camera_intrinsics"][cam_name]]

                    batch[f"observation.points.{cam_name}"] = [obs_state_dict["pc"][cam_idx]]

                ov_out = policy_client.get_action(
                    batch, options={"pred_rot_type": args.pred_rot_type, "remove_arm": args.remove_arm}
                )
                action_chunk = ov_out.action[0].copy()

                # TODO: rotation delta is not simple addition, here only perform delta on position
                if args.delta_action:
                    abs_action_chunk = [state[:3]]
                    for action in action_chunk:
                        abs_action_chunk.append(abs_action_chunk[-1] + action[:3])
                    action_chunk = np.concatenate(
                        [np.array(abs_action_chunk[1:]), action_chunk[:, 3:]], 1
                    )

                assert len(action_chunk) >= args.replan_steps, (
                    f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                )
                if args.action_ensemble:
                    for i, action in enumerate(action_plan):
                        # action_plan[i] = (action + action_chunk[i]) / 2
                        # TODO: only ensemble position not rotation
                        action_plan[i][:3] = (action[:3] + action_chunk[i, :3]) / 2
                        action_plan[i][3:] = action_chunk[i, 3:]
                    action_plan.extend(action_chunk[len(action_plan):])
                else:
                    action_plan.extend(action_chunk[: args.replan_steps])

                if args.save_video:
                    if args.project_action_on_image:
                        for c, cam_name in enumerate(args.select_cameras):
                            extrinsics = obs_state_dict["camera_extrinsics"][cam_name]
                            intrinsics = obs_state_dict["camera_intrinsics"][cam_name]
                            replay_images[-1][c] = project_action_chunks_on_image(
                                replay_images[-1][c], action_chunk, extrinsics, intrinsics
                            )
                    replay_images[-1] = np.concatenate(replay_images[-1], 1)

                if args.save_obs_outs:
                    current_replan_file = os.path.join("replans", f"step_{step_id:06d}.npz")
                    np.savez(os.path.join(output_dir, current_replan_file), action_chunk=action_chunk)
                current_replan_step = step_id

            # action = action_plan.popleft()
            action = action_plan[0]
            action_plan = action_plan[1:]

            if prev_action is None:
                prev_action = action.copy()
            elif args.stop_with_policy:
                if is_action_similar(prev_action, action):
                    if args.save_obs_outs:
                        write_transition(
                            output_dir,
                            {
                                "step_id": step_id,
                                "obs_before": f"obs_{step_id:06d}.npz",
                                "obs_after": None,
                                "executed_action": None,
                                "candidate_action": np.asarray(action).tolist(),
                                "reward": reward,
                                "terminate": True,
                                "is_replan": step_id == current_replan_step,
                                "replan_file": current_replan_file,
                                "action_chunk_index": None if current_replan_step is None else step_id - current_replan_step,
                                "stop_reason": "policy",
                            },
                        )
                    break
                prev_action = action.copy()

            if args.clip_within_workspace:
                workspace = get_rlbench_robot_workspace()
                action[0] = np.clip(action[0], a_min=workspace["X_BBOX"][0], a_max=workspace["X_BBOX"][1])
                action[1] = np.clip(action[1], a_min=workspace["Y_BBOX"][0], a_max=workspace["Y_BBOX"][1])
                action[2] = np.clip(action[2], a_min=workspace["Z_BBOX"][0], a_max=workspace["Z_BBOX"][1])

            # update the observation based on the predicted action
            geometry_tracker.start_action(step_id, action, obs_state_dict["gripper"])
            try:
                obs, reward, terminate, _ = move(action, verbose=False)
                # error_type = None
                obs_state_dict = env.get_observation(obs)  # type: ignore
                geometry_tracker.finish_action(
                    obs_state_dict["gripper"], reward, terminate
                )
                if args.save_obs_outs:
                    save_obs_state(output_dir, step_id + 1, obs_state_dict)
                    write_transition(
                        output_dir,
                        {
                            "step_id": step_id,
                            "obs_before": f"obs_{step_id:06d}.npz",
                            "obs_after": f"obs_{step_id + 1:06d}.npz",
                            "executed_action": np.asarray(action).tolist(),
                            "reward": float(reward),
                            "terminate": bool(terminate),
                            "is_replan": step_id == current_replan_step,
                            "replan_file": current_replan_file,
                            "action_chunk_index": None if current_replan_step is None else step_id - current_replan_step,
                        },
                    )
                if reward == 1:
                    if not has_reward_accumulated:
                        success_rate += 1 / num_episodes
                        has_reward_accumulated = True
                    if not args.no_env_reward:
                        break
                # if terminate:
                #     print("The episode has terminated!")

            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(args.taskvar, episode_id, step_id, e)
                # error_type = str(e)
                if args.save_obs_outs:
                    write_transition(
                        output_dir,
                        {
                            "step_id": step_id,
                            "obs_before": f"obs_{step_id:06d}.npz",
                            "obs_after": None,
                            "executed_action": np.asarray(action).tolist(),
                            "reward": 0.0,
                            "terminate": True,
                            "is_replan": step_id == current_replan_step,
                            "replan_file": current_replan_file,
                            "action_chunk_index": None if current_replan_step is None else step_id - current_replan_step,
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                reward = 0
                geometry_tracker.finish_action(
                    None,
                    reward,
                    True,
                    error=f"{type(e).__name__}: {e}",
                )
                break

        continuous_recording = False
        if args.save_video:
            replay_images.append(
                observation_video_frame(obs_state_dict, args.select_cameras)
            )
        if args.continuous_video_fps > 0:
            continuous_images.append(
                observation_video_frame(obs_state_dict, args.select_cameras)
            )

        print(
            taskvar, "Episode", episode_id, "Step", step_id+1,
            "Reward", reward, "Accumulated SR: %.2f" % (success_rate * 100),
            "Estimated SR: %.2f" % (success_rate * num_episodes / (episode_id+1) * 100)
        )

        if args.save_video:
            reward_str = "success" if reward == 1 else "failure"
            video_path = os.path.join(
                args.save_dir, "videos", f"{taskvar}_episode_{episode_id:06d}_{reward_str}.mp4"
            )
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            save_images_to_video(replay_images, video_path, fps=5)

        continuous_video_path = None
        if args.continuous_video_fps > 0:
            reward_str = "success" if reward == 1 else "failure"
            continuous_video_path = os.path.join(
                args.save_dir,
                "continuous_videos_2hz",
                f"{taskvar}_episode_{episode_id:06d}_{reward_str}.mp4",
            )
            os.makedirs(os.path.dirname(continuous_video_path), exist_ok=True)
            save_images_to_video(
                continuous_images,
                continuous_video_path,
                fps=continuous_actual_fps,
            )

        if args.save_dir:
            for geometry_record in geometry_tracker.episode_records:
                write_to_file(
                    os.path.join(args.save_dir, "water_plants_geometry_steps.jsonl"),
                    geometry_record,
                )
            write_to_file(
                os.path.join(args.save_dir, "water_plants_geometry_episodes.jsonl"),
                geometry_tracker.episode_summary(reward == 1),
            )
            write_to_file(
                os.path.join(args.save_dir, "episode_results.jsonl"),
                {
                    "task": task_str,
                    "variation": variation_id,
                    "episode": episode_id,
                    "success": bool(reward == 1),
                    "reward": float(reward),
                    "policy_steps": step_id + 1,
                    "keyframes": len(replay_images),
                    "continuous_frames": len(continuous_images),
                    "continuous_fps": continuous_actual_fps,
                },
            )


    print(f"Success Rate: {success_rate * 100:.2f}%")

    if pred_file is not None:
        write_to_file(
            pred_file,
            {
                "task": task_str, "variation": variation_id,
                "num_episodes": num_episodes, "sr": success_rate,
                "checkpoint": args.pretrained_path,
            }
        )

    env.env.shutdown()
    print(colored(f"Taskvar: {taskvar} SR: {success_rate:.2f}", "black", "on_yellow"))



def producer_entry(proc_id, args, taskvar, pred_file, producer_queue):
    try:
        producer_fn(proc_id, args, taskvar, pred_file, producer_queue)
    finally:
        producer_queue.put(proc_id)


def main(args: ClientArgs) -> None:

    if os.path.exists(os.path.join(args.pretrained_path, "processor_config.json")):
        processor_config = json.load(open(os.path.join(args.pretrained_path, "processor_config.json")))
        if not args.repo_id:
            args.repo_id = list(processor_config["robot_config"]["features"].keys())[0]
            print(f"Reset repo_id as {args.repo_id}")
        select_cameras = processor_config["robot_config"]["select_video_keys"][args.repo_id]
        args.select_cameras = tuple([cam_name.split(".")[-1][:-len("_image")] for cam_name in select_cameras])
    print("select cameras", args.select_cameras)

    existed_taskvars = set()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        pred_file = os.path.join(args.save_dir, "results.jsonl")
        if os.path.exists(pred_file):
            with jsonlines.open(pred_file, "r") as f:
                for item in f:
                    if item["checkpoint"] == args.pretrained_path:
                        existed_taskvars.add("{}+{}".format(item["task"], item["variation"]))
            print(f"Exist #{len(existed_taskvars)} taskvars in {pred_file}.")
    else:
        pred_file = None

    if args.taskvar:
        taskvars = [args.taskvar]
    elif args.taskvar_file:
        taskvars = json.load(open(args.taskvar_file))
    else:
        raise NotImplementedError("Must define taskvar or taskvar_file")

    taskvars = [taskvar for taskvar in taskvars if taskvar not in existed_taskvars]
    print("Evaluate #taskvars", len(taskvars))
    if len(taskvars) == 0:
        return

    producer_queue = mp.Queue(args.num_workers * 2)

    producers = {}
    i = 0
    while i < len(taskvars):
        taskvar = taskvars[i]
        if len(producers) < args.num_workers:
            print("start", i, taskvar)
            producer = mp.Process(
                target=producer_entry,
                args=(i, args, taskvar, pred_file, producer_queue),
                name=taskvar
            )
            producer.start()
            producers[i] = producer
            i += 1
        else:
            proc_id = producer_queue.get()
            producers[proc_id].join()
            if producers[proc_id].exitcode != 0:
                raise RuntimeError(
                    f"RLBench worker {producers[proc_id].name} exited with "
                    f"code {producers[proc_id].exitcode}"
                )
            del producers[proc_id]

    for p in producers.values():
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"RLBench worker {p.name} exited with code {p.exitcode}"
            )


if __name__ == "__main__":
    tyro.cli(main)
