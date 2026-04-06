"""Launch dual-system HVLA inference: spawns S2 process + runs S1 in main process.

S2 (VLM, ~4-15Hz) extracts scene latent → shared memory.
S1 (action policy, ~22-30Hz) reads latent, controls robot.

Usage:
    python -m lerobot.policies.hvla.launch \
        --s1-checkpoint outputs/act_vlm/checkpoint-80000 \
        --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-pytorch/model.safetensors \
        --task "assemble cylinder into ring" \
        --resize-images 224x224 \
        --temporal-ensemble-coeff 0.01
"""

import argparse
import logging
import multiprocessing
import os
import signal

from lerobot.policies.hvla.ipc import (
    DEFAULT_S2_CAM_KEY_MAP,
    SharedLatentCache,
    SharedImageBuffer,
    XLEROBOT_S2_CAM_KEY_MAP,
    ZMQLatentSubscriber,
    ZMQImagePublisher,
    parse_shm_image_size,
    parse_s2_camera_map,
)
from lerobot.policies.hvla.logging_utils import setup_process_logging
from lerobot.policies.hvla.s1_process import _resolve_hvla_robot_config_path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HVLA dual-system inference (local, no server)")
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", default=None,
                        help="Path to S2 checkpoint (not needed with --attach-s2 or --zero-s2)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-images", default="224x224")
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--decode-subtask", action="store_true")
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--s2-image-keys", nargs="+",
                        default=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb"])
    parser.add_argument(
        "--shm-image-size",
        default=None,
        help="Height×Width for S1→S2 shared RGB buffers (match robot frames). "
        "Default: 720×1280, or 360×640 when --hvla-preset xlerobot.",
    )
    parser.add_argument(
        "--s2-camera-map",
        default=None,
        help="Comma-separated robot_cam:S2_slot pairs (obs keys from get_observation → SHM slot). "
        "Example xlerobot: head:base_0_rgb,left_wrist:left_wrist_0_rgb,right_wrist:right_wrist_0_rgb",
    )
    parser.add_argument(
        "--hvla-preset",
        choices=["none", "xlerobot"],
        default="none",
        help="xlerobot: sets S2 camera map to head/left_wrist/right_wrist → base_0/wrist slots "
        "and shm-image-size to 360x640 unless overridden.",
    )
    parser.add_argument("--zero-s2", action="store_true")
    parser.add_argument("--compile-s1", action="store_true", default=True,
                        help="torch.compile S1 denoise_step (default: on, ~2ms savings)")
    parser.add_argument("--no-compile-s1", dest="compile_s1", action="store_false",
                        help="Disable torch.compile for S1")
    parser.add_argument("--s1-type", choices=["act", "flow"], default="act",
                        help="S1 policy type: 'act' (ACTWithVLM, default) or 'flow' (flow matching with RTC)")
    parser.add_argument("--s2-throttle-ms", type=int, default=100,
                        help="Sleep ms after each S2 query to yield GPU to S1 (0=no throttle)")
    parser.add_argument("--osc-skip", action="store_true",
                        help="Enable both-arms oscillation skip: when both arms are flat, "
                             "jump ahead in chunk to where movement starts")
    parser.add_argument("--denoise-steps", type=int, default=None,
                        help="Number of flow matching denoising steps (default: 10 from config)")
    parser.add_argument("--s1-query-interval", type=int, default=2,
                        help="Number of action steps to wait before re-querying S1. "
                             "E.g. 20 = execute 20 actions (~660ms at 30fps) from current "
                             "chunk before next inference. 0 = query as fast as possible.")
    parser.add_argument("--max-step-delta", type=float, default=None,
                        help="Max degrees any joint can change per frame (e.g. 10). Prevents sudden jumps.")
    parser.add_argument("--save-grip-drops", type=str, default=None,
                        help="Directory to save observations when grip drops are detected (for offline analysis)")
    parser.add_argument("--record-dataset", type=str, default=None,
                        help="Record inference episode to LeRobotDataset (e.g. 'user/hvla_ep1'). "
                             "Saves obs+actions each frame, commits on shutdown.")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of rollout episodes to record (default: 1 = single run)")
    parser.add_argument("--episode-time-s", type=float, default=0,
                        help="Max seconds per episode (0 = run until Ctrl+C)")
    parser.add_argument("--reset-time-s", type=float, default=20,
                        help="Seconds to wait during reset phase between episodes (default: 20)")
    parser.add_argument("--teleop-config", type=str, default=None,
                        help="Path to teleop profile JSON for intervention / inverse follow. "
                             "When set, SPACE toggles between policy and human control.")
    parser.add_argument("--intervention-dataset", type=str, default=None,
                        help="Record intervention fragments to a separate LeRobotDataset "
                             "(e.g. 'user/hvla_interventions')")
    # --- RLT (online RL) ---
    parser.add_argument("--rlt-mode", action="store_true",
                        help="Enable RLT online RL: actor MLP refines S1 chunks. "
                             "Press R for success (+1 reward, ends episode).")
    parser.add_argument("--rl-token-checkpoint", type=str, default=None,
                        help="Path to trained RL token encoder checkpoint (Phase 1)")
    parser.add_argument("--rlt-checkpoint", type=str, default=None,
                        help="Path to existing RLT actor checkpoint dir (resumes training or deploys)")
    parser.add_argument("--rlt-deploy", action="store_true",
                        help="Deploy mode: actor forward only, no training loop")
    parser.add_argument("--rl-chunk-length", type=int, default=10,
                        help="RL action chunk length C (default: 10)")
    parser.add_argument("--rlt-output-dir", type=str, default="outputs/rlt_online",
                        help="Directory for RLT checkpoints and logs")
    # --- Dry run (config/policy validation without physical robot) ---
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate robot config, load policy, run synthetic inference steps — "
                             "no physical robot connection, exits after N steps.")
    parser.add_argument("--dry-run-steps", type=int, default=5,
                        help="Number of synthetic inference steps for --dry-run (default: 5)")
    # --- Cross-machine ZMQ mode (Jetson S1 ↔ GPU machine S2) ---
    parser.add_argument(
        "--zmq-latent-host",
        default=None,
        help=(
            "IP of the GPU machine running S2 (e.g. 192.168.1.10). "
            "When set, S1 receives latents over ZMQ instead of shared memory "
            "and publishes images to S2 over ZMQ. S2 must be started with "
            "--zmq-image-host <jetson-ip> on the GPU machine."
        ),
    )
    parser.add_argument(
        "--zmq-latent-port",
        type=int,
        default=ZMQLatentSubscriber.DEFAULT_PORT,
        help="ZMQ port for incoming latents from the GPU machine S2 (default: %(default)s).",
    )
    parser.add_argument(
        "--zmq-image-port",
        type=int,
        default=ZMQImagePublisher.DEFAULT_PORT,
        help="ZMQ port to publish robot images on for remote S2 (default: %(default)s).",
    )
    parser.add_argument(
        "--zmq-image-jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for images sent to remote S2 (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.hvla_preset == "xlerobot":
        if args.s2_camera_map is None:
            args.s2_camera_map = ",".join(f"{k}:{v}" for k, v in XLEROBOT_S2_CAM_KEY_MAP.items())
        if args.shm_image_size is None:
            args.shm_image_size = "360x640"
    if args.shm_image_size is None:
        args.shm_image_size = "720x1280"

    shm_h, shm_w = parse_shm_image_size(args.shm_image_size)
    s2_cam_key_map = (
        parse_s2_camera_map(args.s2_camera_map) if args.s2_camera_map else dict(DEFAULT_S2_CAM_KEY_MAP)
    )

    # s2-checkpoint only needed if no existing S2 process is found
    # (validated later, after attempting to attach)

    setup_process_logging()

    # Fail fast before SHM / S2 spawn — avoids leaked shared_memory when robot JSON is missing
    try:
        resolved_robot_config = _resolve_hvla_robot_config_path(args.robot_config)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return

    resize = tuple(int(x) for x in args.resize_images.split("x")) if args.resize_images else None
    s2_image_keys = tuple(args.s2_image_keys)

    # Use 'spawn' context — child process gets fresh CUDA context
    ctx = multiprocessing.get_context("spawn")

    zmq_mode = getattr(args, "zmq_latent_host", None) is not None

    # IPC setup — ZMQ (cross-machine) or shared memory (same-machine)
    s2_attached = False
    s2_proc = None

    if zmq_mode:
        # Cross-machine: S2 runs on GPU machine, S1 runs here (Jetson).
        # Publish images to S2 over ZMQ; receive latents over ZMQ.
        logger.info(
            "ZMQ mode: publishing images on port %d, subscribing to latents from %s:%d",
            args.zmq_image_port, args.zmq_latent_host, args.zmq_latent_port,
        )
        shared_images = ZMQImagePublisher(
            camera_keys=s2_image_keys,
            height=shm_h,
            width=shm_w,
            port=args.zmq_image_port,
            jpeg_quality=args.zmq_image_jpeg_quality,
        )
        shared_cache = ZMQLatentSubscriber(
            host=args.zmq_latent_host,
            port=args.zmq_latent_port,
        )
        logger.info(
            "ZMQ IPC ready. Make sure S2 is running on %s with "
            "--zmq-image-host <this-jetson-ip> --zmq-image-port %d --zmq-latent-port %d",
            args.zmq_latent_host, args.zmq_image_port, args.zmq_latent_port,
        )
        s2_attached = True  # treat as "attached" so no local S2 spawn

    elif not args.zero_s2:
        # Same-machine: shared memory.
        # Try to attach to an existing S2 process (started by s2_standalone.py)
        try:
            shared_cache = SharedLatentCache(create=False, name=SharedLatentCache.SHM_NAME)
            shared_images = SharedImageBuffer(
                camera_keys=s2_image_keys, height=shm_h, width=shm_w, create=False,
            )
            s2_attached = True
            logger.info("Attached to existing S2 (latent count=%d, age=%.0fms)",
                        shared_cache.count, shared_cache.age_ms)
        except FileNotFoundError:
            # No existing S2 — need to spawn one
            if args.s2_checkpoint is None:
                logger.error("No S2 process found and no --s2-checkpoint provided.\n"
                             "Either start S2 first:\n"
                             "  python -m lerobot.policies.hvla.s2_standalone --checkpoint ... --task ...\n"
                             "Or provide --s2-checkpoint to spawn one.")
                return
            logger.info("No existing S2 found, will spawn from checkpoint")
            shared_cache = SharedLatentCache(create=True)
            shared_images = SharedImageBuffer(
                camera_keys=s2_image_keys, height=shm_h, width=shm_w, create=True,
            )
            logger.info("S2 SHM image layout: %d×%d (keys=%s)", shm_h, shm_w, s2_image_keys)
    else:
        # --zero-s2: run S1 without S2 conditioning entirely.
        # Pass None to s1_process so S2 latent is never injected into the batch.
        shared_cache = None
        shared_images = None

    stop_event = ctx.Event()

    # Ctrl+C handling: first = graceful, second = force quit
    _ctrl_c_count = 0

    def signal_handler(sig, frame):
        nonlocal _ctrl_c_count
        _ctrl_c_count += 1
        if _ctrl_c_count >= 2:
            logger.warning("Force quit (second Ctrl+C)")
            os._exit(1)
        logger.info("Ctrl+C — shutting down (press again to force quit)...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start S2 process (unless already attached or using zero stub)
    s2_proc = None
    if s2_attached:
        logger.info("Using existing S2 process (no spawn)")
    elif args.zero_s2:
        logger.info("--zero-s2: S2 disabled, S1 runs without S2 conditioning")
    else:
        from lerobot.policies.hvla.s2_process import run_s2

        s2_proc = ctx.Process(
            target=run_s2,
            kwargs={
                "checkpoint_path": args.s2_checkpoint,
                "shared_cache": shared_cache,
                "shared_images": shared_images,
                "task": args.task,
                "device": args.device,
                "image_keys": s2_image_keys,
                "decode_subtask": args.decode_subtask,
                "norm_stats_path": args.norm_stats,
                "stop_event": stop_event,
                "throttle_ms": args.s2_throttle_ms,
            },
            daemon=True,
        )
        s2_proc.start()
        logger.info("S2 process started (PID %d)", s2_proc.pid)

    # Run S1 in main process
    from lerobot.policies.hvla.s1_process import run_s1

    try:
        run_s1(
            s1_checkpoint=args.s1_checkpoint,
            shared_cache=shared_cache,
            shared_images=shared_images,
            task=args.task,
            s2_cam_key_map=s2_cam_key_map,
            robot_config_path=resolved_robot_config,
            fps=args.fps,
            device=args.device,
            resize_images=resize,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            n_action_steps=args.n_action_steps,
            compile_s1=args.compile_s1,
            s1_type=args.s1_type,
            stop_event=stop_event,
            osc_skip=args.osc_skip,
            query_interval_steps=args.s1_query_interval,
            num_denoise_steps=args.denoise_steps,
            max_step_delta=args.max_step_delta,
            grip_drop_save_dir=args.save_grip_drops,
            record_dataset=args.record_dataset,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            reset_time_s=args.reset_time_s,
            teleop_config_path=args.teleop_config,
            intervention_dataset=args.intervention_dataset,
            rlt_mode=args.rlt_mode,
            rl_token_checkpoint=args.rl_token_checkpoint,
            rlt_checkpoint=args.rlt_checkpoint,
            rlt_deploy=args.rlt_deploy,
            rl_chunk_length=args.rl_chunk_length,
            rlt_output_dir=args.rlt_output_dir,
            dry_run=args.dry_run,
            dry_run_steps=args.dry_run_steps,
        )
    finally:
        stop_event.set()
        if s2_proc is not None:
            s2_proc.join(timeout=5.0)
            if s2_proc.is_alive():
                logger.warning("S2 process didn't exit cleanly, terminating")
                s2_proc.terminate()
        if shared_images is not None:
            shared_images.cleanup()
        if shared_cache is not None:
            shared_cache.cleanup()
        logger.info("HVLA shutdown complete")


if __name__ == "__main__":
    main()
