#!/usr/bin/env python3
"""
Voice-controlled robot interface using ElevenLabs Conversational AI.

This script allows you to control a robot arm through voice commands.
The ElevenLabs agent will trigger local functions based on your speech.

Setup:
1. Install: pip install elevenlabs pyaudio
2. Set your API key: export ELEVENLABS_API_KEY="your-key-here"
3. Create tools in ElevenLabs dashboard with these exact names:
   - plug_charger: "Call when user asks to plug in the charger"
   - plug_charger_and_switch: "Call when user asks to plug charger and turn on the switch"
   - turn_on_switch: "Call when user asks to turn on the switch"
   - stop_run: "Call when user asks to stop the current robot action"

Usage:
    python voice_control_robot.py --agent-id YOUR_AGENT_ID
"""

import argparse
import atexit
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from elevenlabs.conversational_ai.conversation import ClientTools

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging

DEFAULT_POLICY_PATH = "igkp/groot_100k"
DEFAULT_TASK = "plug charger into the socket"
DEFAULT_EPISODE_TIME_S = 300
DEFAULT_FPS = 30

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "kanishk_lerobot_follower_2"

CAMERA_SPECS = {
    "wrist": {"index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    "top": {"index_or_path": 12, "width": 640, "height": 480, "fps": 30},
    "base": {"index_or_path": 10, "width": 640, "height": 480, "fps": 30},
}

SWITCH_REPLAY_REPO_ID = "roborage/record_switch_v2"
SWITCH_REPLAY_EPISODE = 4

DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "voice_runs"


@dataclass
class _ActiveRun:
    thread: threading.Thread
    cancel_event: threading.Event
    label: str


class WarmRobotRuntime:
    def __init__(
        self,
        *,
        policy_path: str,
        run_root: Path,
        episode_time_s: int,
        fps: int,
    ):
        init_logging()
        self.policy_path = policy_path
        self.run_root = run_root
        self.episode_time_s = episode_time_s
        self.fps = fps

        self.robot = self._build_robot()
        (
            self.teleop_action_processor,
            self.robot_action_processor,
            self.robot_observation_processor,
        ) = make_default_processors()
        self.dataset_features = self._build_dataset_features()

        self.policy: PreTrainedPolicy | None = None
        self.preprocessor = None
        self.postprocessor = None
        self._policy_meta: LeRobotDatasetMetadata | None = None

        self._switch_dataset: LeRobotDataset | None = None
        self._switch_episode_frames = None
        self._switch_actions = None

        self.robot.connect()
        atexit.register(self.close)

    def _build_robot(self):
        cameras = {
            name: OpenCVCameraConfig(
                index_or_path=spec["index_or_path"],
                width=spec["width"],
                height=spec["height"],
                fps=spec["fps"],
            )
            for name, spec in CAMERA_SPECS.items()
        }
        robot_cfg = SOFollowerRobotConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras=cameras)
        return make_robot_from_config(robot_cfg)

    def _build_dataset_features(self):
        return combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=self.teleop_action_processor,
                initial_features=create_initial_features(action=self.robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_observation_processor,
                initial_features=create_initial_features(observation=self.robot.observation_features),
                use_videos=True,
            ),
        )

    def _create_run_metadata(self, prefix: str) -> tuple[LeRobotDatasetMetadata, Path]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_id = f"{prefix}_{ts}"
            run_dir = self.run_root / run_id
            try:
                meta = LeRobotDatasetMetadata.create(
                    repo_id=run_id,
                    fps=self.fps,
                    features=self.dataset_features,
                    robot_type=self.robot.name,
                    root=run_dir,
                    use_videos=True,
                )
                return meta, run_dir
            except FileExistsError:
                time.sleep(0.001)
        raise RuntimeError("Failed to create a unique run directory")

    def _write_run_manifest(self, run_dir: Path, payload: dict) -> None:
        manifest_path = run_dir / "run.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _ensure_policy_loaded(self) -> None:
        if self.policy is not None:
            return

        from lerobot.configs.policies import PreTrainedConfig

        policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        policy_cfg.pretrained_path = self.policy_path

        self._policy_meta, _ = self._create_run_metadata("policy_init")
        self.policy = make_policy(policy_cfg, ds_meta=self._policy_meta)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            dataset_stats=rename_stats(self._policy_meta.stats, {}),
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
                "rename_observations_processor": {"rename_map": {}},
            },
        )

    def run_policy(
        self,
        *,
        task: str,
        duration_s: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        self._ensure_policy_loaded()
        _, run_dir = self._create_run_metadata("policy_run")
        self._write_run_manifest(
            run_dir,
            {
                "task": task,
                "policy_path": self.policy_path,
                "episode_time_s": duration_s or self.episode_time_s,
                "fps": self.fps,
                "started_at": datetime.now().isoformat(),
            },
        )

        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("Policy is not initialized")

        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

        start_episode_t = time.perf_counter()
        while time.perf_counter() - start_episode_t < (duration_s or self.episode_time_s):
            if cancel_event is not None and cancel_event.is_set():
                break
            start_loop_t = time.perf_counter()

            obs = self.robot.get_observation()
            obs_processed = self.robot_observation_processor(obs)

            observation_frame = build_dataset_frame(self.dataset_features, obs_processed, prefix=OBS_STR)
            if cancel_event is not None and cancel_event.is_set():
                break
            action_values = predict_action(
                observation=observation_frame,
                policy=self.policy,
                device=get_safe_torch_device(self.policy.config.device),
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=self.policy.config.use_amp,
                task=task,
                robot_type=self.robot.robot_type,
            )
            if cancel_event is not None and cancel_event.is_set():
                break

            act_processed_policy = make_robot_action(action_values, self.dataset_features)
            robot_action_to_send = self.robot_action_processor((act_processed_policy, obs))
            _ = self.robot.send_action(robot_action_to_send)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / self.fps - dt_s, 0.0))

        return run_dir

    def _ensure_switch_dataset_loaded(self) -> None:
        if self._switch_dataset is not None:
            return

        dataset = LeRobotDataset(SWITCH_REPLAY_REPO_ID, episodes=[SWITCH_REPLAY_EPISODE])
        episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == SWITCH_REPLAY_EPISODE)
        actions = episode_frames.select_columns(ACTION)

        self._switch_dataset = dataset
        self._switch_episode_frames = episode_frames
        self._switch_actions = actions

    def run_switch_replay(self, *, cancel_event: threading.Event | None = None) -> Path:
        self._ensure_switch_dataset_loaded()
        _, run_dir = self._create_run_metadata("switch_replay")
        self._write_run_manifest(
            run_dir,
            {
                "dataset_repo_id": SWITCH_REPLAY_REPO_ID,
                "episode": SWITCH_REPLAY_EPISODE,
                "started_at": datetime.now().isoformat(),
            },
        )

        if self._switch_dataset is None or self._switch_episode_frames is None or self._switch_actions is None:
            raise RuntimeError("Switch replay dataset is not initialized")

        for idx in range(len(self._switch_episode_frames)):
            if cancel_event is not None and cancel_event.is_set():
                break
            start_episode_t = time.perf_counter()

            action_array = self._switch_actions[idx][ACTION]
            action = {
                name: action_array[i]
                for i, name in enumerate(self._switch_dataset.features[ACTION]["names"])
            }

            robot_obs = self.robot.get_observation()
            if cancel_event is not None and cancel_event.is_set():
                break
            processed_action = self.robot_action_processor((action, robot_obs))
            _ = self.robot.send_action(processed_action)

            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(max(1 / self._switch_dataset.fps - dt_s, 0.0))

        return run_dir

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()


class RobotVoiceController:
    """Voice-controlled robot interface."""

    def __init__(
        self,
        *,
        dry_run: bool = False,
        policy_path: str = DEFAULT_POLICY_PATH,
        episode_time_s: int = DEFAULT_EPISODE_TIME_S,
        fps: int = DEFAULT_FPS,
        run_root: Path = DEFAULT_RUN_ROOT,
    ):
        self.client_tools = ClientTools()
        self.dry_run = dry_run
        self._state_lock = threading.Lock()
        self._active_run: _ActiveRun | None = None
        self._runtime: WarmRobotRuntime | None = None
        self._runtime_config = {
            "policy_path": policy_path,
            "episode_time_s": episode_time_s,
            "fps": fps,
            "run_root": run_root,
        }
        self.setup_tools()

    def setup_tools(self):
        """Register all available robot control functions."""
        # Register each function with the exact name from ElevenLabs dashboard
        self.client_tools.register("plug_charger", self.plug_charger)
        self.client_tools.register("plug_charger_and_switch", self.plug_charger_and_switch)
        self.client_tools.register("turn_on_switch", self.turn_on_switch)
        self.client_tools.register("stop_run", self.stop_run)

    # ========== Tool Functions ==========
    # These get called when the ElevenLabs agent triggers them

    def _get_runtime(self) -> WarmRobotRuntime:
        if self._runtime is None:
            self._runtime = WarmRobotRuntime(**self._runtime_config)
        return self._runtime

    def warmup(self) -> None:
        """Pre-load the robot runtime and policy model so they're ready for voice commands."""
        print("🔥 Warming up robot runtime and loading Groot model...")
        runtime = self._get_runtime()
        runtime._ensure_policy_loaded()
        print("✅ Model loaded and ready!")

    def _preempt_active_run(self, *, timeout_s: float | None = 30.0) -> bool:
        with self._state_lock:
            if not self._active_run or not self._active_run.thread.is_alive():
                return True
            if self._active_run.thread is threading.current_thread():
                return False
            self._active_run.cancel_event.set()
            active_thread = self._active_run.thread

        print("\n⏹️  Preempting current run...")
        if timeout_s is None or timeout_s <= 0:
            active_thread.join()
        else:
            active_thread.join(timeout=timeout_s)
        if active_thread.is_alive():
            print("\n❗ Could not preempt the active run in time. Try again.")
            return False
        with self._state_lock:
            if self._active_run and not self._active_run.thread.is_alive():
                self._active_run = None
        return True

    def _run_action_thread(self, label, run_fn, cancel_event: threading.Event) -> None:
        try:
            run_dir = run_fn(cancel_event)
            if run_dir:
                print(f"\n✅ {label} completed! Run dir: {run_dir}")
            else:
                print(f"\n❌ {label} failed.")
        except Exception as exc:
            print(f"\n❌ {label} error: {exc}")
        finally:
            with self._state_lock:
                if self._active_run and self._active_run.thread is threading.current_thread():
                    self._active_run = None

    def _start_async_action(self, label, run_fn, dry_run_message) -> bool:
        if self.dry_run:
            print("\n" + "=" * 60)
            print(f"⚡ {label} triggered")
            print("=" * 60)
            print(f"\n🧪 Dry run enabled. {dry_run_message}")
            print("=" * 60 + "\n")
            return True

        with self._state_lock:
            if self._active_run and self._active_run.thread.is_alive():
                print("\n❗ Robot is busy. Try again after the current run finishes.")
                return False
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._run_action_thread,
                args=(label, run_fn, cancel_event),
                daemon=True,
            )
            self._active_run = _ActiveRun(thread=thread, cancel_event=cancel_event, label=label)

        print("\n" + "=" * 60)
        print(f"⚡ {label} started")
        print("=" * 60 + "\n")
        thread.start()
        return True

    def _run_sync_action(self, label, run_fn, dry_run_message) -> bool:
        if self.dry_run:
            print("\n" + "=" * 60)
            print(f"⚡ {label} triggered")
            print("=" * 60)
            print(f"\n🧪 Dry run enabled. {dry_run_message}")
            print("=" * 60 + "\n")
            return True

        print("\n" + "=" * 60)
        print(f"⚡ {label} triggered")
        print("=" * 60)
        try:
            run_dir = run_fn()
            if not run_dir:
                print("\n❌ Command failed.")
                print("=" * 60 + "\n")
                return False
            print(f"\n✅ Command completed successfully! Run dir: {run_dir}")
            print("=" * 60 + "\n")
            return True
        except Exception as exc:
            print(f"\n❌ Command error: {exc}")
            print("=" * 60 + "\n")
            return False

    def plug_charger(self, params):
        """Plug in the charger."""
        ok = self._start_async_action(
            "PLUG CHARGER",
            lambda cancel_event: self._get_runtime().run_policy(task=DEFAULT_TASK, cancel_event=cancel_event),
            "Would run warm Groot policy (plug charger).",
        )
        return "Started plugging in the charger" if ok else "Failed to start plugging in the charger"

    def plug_charger_and_switch(self, params):
        """Plug in charger and turn on the switch."""
        ok = self._start_async_action(
            "PLUG CHARGER AND TURN ON SWITCH",
            lambda cancel_event: self._get_runtime().run_policy(task=DEFAULT_TASK, cancel_event=cancel_event),
            "Would run warm Groot policy (plug charger, no auto-fallback).",
        )
        if ok:
            return "Started plug-in (switch handled by voice follow-up if needed)"
        return "Failed to start plug-in"

    def turn_on_switch(self, params):
        """Turn on the switch."""
        if not self._preempt_active_run(timeout_s=30.0):
            return "Failed to stop current run; try again"
        ok = self._start_async_action(
            "TURN ON SWITCH",
            lambda cancel_event: self._get_runtime().run_switch_replay(cancel_event=cancel_event),
            "Would replay switch trajectory (hard-coded).",
        )
        return "Started turning on the switch" if ok else "Failed to start turning on the switch"

    def stop_run(self, params):
        """Stop the currently running action."""
        with self._state_lock:
            active = self._active_run
            if not active or not active.thread.is_alive():
                return "No active run to stop"
            if active.thread is threading.current_thread():
                return "Stop requested from the active run; ignoring"
            active.cancel_event.set()
            active_thread = active.thread

        active_thread.join(timeout=10.0)
        if active_thread.is_alive():
            return "Stop requested, but the run is still stopping"
        with self._state_lock:
            if self._active_run and not self._active_run.thread.is_alive():
                self._active_run = None
        return "Stopped the current run"


def main():
    parser = argparse.ArgumentParser(description="Voice-controlled robot interface")
    parser.add_argument(
        "--agent-id",
        type=str,
        required=True,
        help="ElevenLabs agent ID from your dashboard"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=DEFAULT_POLICY_PATH,
        help="Groot policy path or Hugging Face repo ID"
    )
    parser.add_argument(
        "--episode-time-s",
        type=int,
        default=DEFAULT_EPISODE_TIME_S,
        help="Episode duration in seconds for policy runs"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Control loop FPS for policy runs"
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=str(DEFAULT_RUN_ROOT),
        help="Directory for per-run metadata/logs"
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ Error: ELEVENLABS_API_KEY not set!")
        print("\nSet it with:")
        print("  export ELEVENLABS_API_KEY='your-key-here'")
        print("Or pass it with --api-key")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🤖 Voice-Controlled Robot Interface")
    print("=" * 60)
    print("\nInitializing ElevenLabs Conversational AI...")
    print(f"Agent ID: {args.agent_id}")

    # Create ElevenLabs client
    client = ElevenLabs(api_key=api_key)

    # Create robot controller with registered tools
    controller = RobotVoiceController(
        dry_run=args.dry_run,
        policy_path=args.policy_path,
        episode_time_s=args.episode_time_s,
        fps=args.fps,
        run_root=Path(args.run_root),
    )

    print("\n✅ Robot controller initialized!")

    # Pre-load the model before starting voice agent
    if not args.dry_run:
        controller.warmup()

    print("\n📋 Available voice commands:")
    print("   • 'Plug in the charger' - Robot plugs in charger")
    print("   • 'Plug charger and turn on switch' - Do both tasks")
    print("   • 'Turn on the switch' - Robot turns on switch")

    print("\n🎤 Starting conversation...")
    print("   Speak into your microphone to control the robot")
    print("   Press Ctrl+C to exit")
    print("=" * 60 + "\n")

    # Create and start conversation
    conversation = Conversation(
        client=client,
        agent_id=args.agent_id,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        client_tools=controller.client_tools,
    )

    # Start the conversation (blocking)
    conversation.start_session()

    # Wait for conversation to end
    conversation.wait_for_session_end()

    print("\n" + "=" * 60)
    print("👋 Conversation ended")
    print("=" * 60)


if __name__ == "__main__":
    main()
