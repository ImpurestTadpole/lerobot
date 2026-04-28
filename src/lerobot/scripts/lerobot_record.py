# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Records a dataset via teleoperation.  This is a pure data-collection
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
    xlerobot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
    xlerobot_vr,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.teleoperators.xlerobot_vr.xlerobot_vr import XLerobotVRTeleop, init_vr_listener
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, start_viz_thread, stop_viz_thread


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = True
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    control_interval = 1 / fps

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    last_status_print = 0  # Track when we last printed status
    
    # Hz rate tracking
    loop_times = []
    last_hz_print = time.perf_counter()
    hz_print_interval = 1.0  # Print Hz rate every 1 second
    prev_loop_start = time.perf_counter()  # Track previous loop start time
    frame_idx = 0  # For throttling visualization
    
    # Depth read throttling: set to 1 to read depth every frame.
    # For training with depth, it's usually better to keep depth time-aligned to RGB/actions.
    depth_read_interval = 1  # Read depth every frame
    last_depth_obs = {}  # Store last depth observation for frames where we skip depth
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        
        # Measure time since last loop (for Hz calculation)
        if prev_loop_start > 0:
            loop_period = start_loop_t - prev_loop_start
            loop_times.append(loop_period)
        prev_loop_start = start_loop_t

        # Update VR events if using VR teleop
        if isinstance(teleop, XLerobotVRTeleop):
            vr_events = teleop.get_vr_events()
            events.update(vr_events)

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation - skip depth reads on some frames for performance
        # We still read color cameras every frame for the dataset
        read_depth_this_frame = (frame_idx % depth_read_interval == 0)
        obs = robot.get_observation(skip_cameras=False, skip_depth=not read_depth_this_frame)
        
        # If we skipped depth this frame, merge last depth observation
        if not read_depth_this_frame and last_depth_obs:
            obs.update(last_depth_obs)
        elif read_depth_this_frame:
            # Store depth data for next frames where we skip depth
            last_depth_obs = {k: v for k, v in obs.items() if k.endswith("_depth")}

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from teleop
        if isinstance(teleop, Teleoperator):
            if hasattr(teleop, "update_observation_cache"):
                teleop.update_observation_cache(obs)
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)
        
        # Use the actually sent action for logging (includes normalization, clipping, etc.)
        action_values = _sent_action

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        # Visualization runs in a background thread - this call is non-blocking.
        # Frames are dropped if the worker is still busy (queue size = 1).
        if display_data:
            obs_for_rerun = {
                k: v for k, v in obs_processed.items()
                if not ("depth" in str(k).lower() or k.endswith("_depth"))
            }
            log_rerun_data(
                observation=obs_for_rerun, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )

        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t
        
        # Print Hz rate every second
        current_time = time.perf_counter()
        if current_time - last_hz_print >= hz_print_interval:
            if loop_times:
                # Calculate actual Hz from measured loop periods (time between loop starts)
                avg_loop_period = sum(loop_times) / len(loop_times)
                current_hz = 1.0 / avg_loop_period if avg_loop_period > 0 else 0.0
                target_hz = fps
                # Also show the number of loops in this interval
                num_loops = len(loop_times)
                ep_str = ""
                ep_idx = events.get("_episode_idx")
                ep_total = events.get("_episode_total")
                if ep_idx is not None and ep_total is not None:
                    ep_str = f"📹 Ep {ep_idx}/{ep_total} | t={timestamp:05.1f}s | "
                logging.info(
                    f"{ep_str}⚡ Control rate: {current_hz:.1f} Hz (target: {target_hz} Hz) | "
                    f"{num_loops} loops in {hz_print_interval:.1f}s"
                )
                loop_times.clear()  # Reset for next interval
            last_hz_print = current_time

        # Increment frame index for visualization and depth throttling
        frame_idx += 1
        
        # Print episode progress every 5 seconds
        if dataset is not None and timestamp - last_status_print >= 5.0:
            elapsed_min = int(timestamp // 60)
            elapsed_sec = int(timestamp % 60)
            remaining_sec = max(0, int(control_time_s - timestamp))
            remaining_min = remaining_sec // 60
            remaining_sec = remaining_sec % 60
            
            logging.info(
                f"📹 Episode {dataset.num_episodes} | "
                f"Time: {elapsed_min:02d}:{elapsed_sec:02d} / "
                f"{int(control_time_s // 60):02d}:{int(control_time_s % 60):02d} | "
                f"Remaining: {remaining_min:02d}:{remaining_sec:02d}"
            )
            last_status_print = timestamp


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
        start_viz_thread()
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            try:
                dataset = LeRobotDataset.resume(
                    cfg.dataset.repo_id,
                    root=cfg.dataset.root,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    vcodec=cfg.dataset.vcodec,
                    streaming_encoding=cfg.dataset.streaming_encoding,
                    encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                    encoder_threads=cfg.dataset.encoder_threads,
                    image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                    image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                    if num_cameras > 0
                    else 0,
                )
                sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
            except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
                logging.error(
                    f"Failed to resume dataset '{cfg.dataset.repo_id}': {e}\n"
                    "The dataset does not exist or is incomplete.\n"
                    "Remove --resume=true to create a new dataset, or ensure the dataset exists on the Hub."
                )
                return None
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        robot.connect()
        if teleop is not None:
            # For XLerobot VR, pass the robot reference so get_action() can compute real commands
            if isinstance(teleop, XLerobotVRTeleop):
                teleop.connect(robot=robot)
            else:
                teleop.connect()

        # Use VR listener if VR teleop, otherwise use keyboard listener
        if isinstance(teleop, XLerobotVRTeleop):
            listener, events = init_vr_listener(teleop)
        else:
            listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                # Defensive: ensure stale flags from a previous loop/reset phase don't affect
                # the semantics of finishing an episode early.
                # In particular, RIGHT B sets `exit_early` to finish an episode early and save it,
                # but if `rerecord_episode` was accidentally triggered earlier (and persists),
                # the episode will be discarded and restarted instead.
                events["rerecord_episode"] = False
                events["exit_early"] = False
                # Also clear the VR teleop's internal (sticky) flag, otherwise it can be
                # re-introduced on the next `teleop.get_vr_events()` call.
                if isinstance(teleop, XLerobotVRTeleop) and getattr(teleop, "vr_event_handler", None):
                    teleop.vr_event_handler.events["rerecord_episode"] = False
                    teleop.vr_event_handler.events["exit_early"] = False

                log_say(
                    f"Recording episode {dataset.num_episodes + 1} / {dataset.num_episodes + cfg.dataset.num_episodes - recorded_episodes}", 
                    cfg.play_sounds
                )
                logging.info(
                    f"🎬 Starting Episode {dataset.num_episodes + 1} "
                    f"(Progress: {recorded_episodes + 1}/{cfg.dataset.num_episodes} episodes)"
                )
                # Used only for logging inside record_loop.
                events["_episode_idx"] = recorded_episodes + 1
                events["_episode_total"] = cfg.dataset.num_episodes
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )
                events.pop("_episode_idx", None)
                events.pop("_episode_total", None)

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    if isinstance(teleop, XLerobotVRTeleop) and getattr(teleop, "vr_event_handler", None):
                        teleop.vr_event_handler.events["rerecord_episode"] = False
                        teleop.vr_event_handler.events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue
                
                # Save episode immediately before reset phase so an interrupt during reset
                # doesn't lose the episode.
                dataset.save_episode()
                recorded_episodes += 1

                logging.info(
                    f"✅ Episode {dataset.num_episodes} saved! "
                    f"({recorded_episodes}/{cfg.dataset.num_episodes} episodes done)"
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and recorded_episodes < cfg.dataset.num_episodes:
                    log_say("Reset the environment", cfg.play_sounds)

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if cfg.display_data:
            stop_viz_thread()

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                try:
                    dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
                    logging.info(f"✅ Dataset pushed to hub: {cfg.dataset.repo_id}")
                except Exception as e:
                    logging.warning(f"⚠️  Failed to push dataset to hub: {e}")
            else:
                logging.info(
                    "Skipping push to hub: no episodes saved "
                    "(recording was interrupted before saving any episodes)"
                )

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()