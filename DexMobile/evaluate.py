import os
from collections import Counter

import numpy as np
import pandas as pd
import pybullet as p
from Dualenv import Dualenv
from Monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from warnings import filterwarnings
filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for scenario screenshots. Install it with `pip install matplotlib`."
    ) from exc


GRASP_TYPE = "poPmAb25"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, GRASP_TYPE)
ENV_PATH = os.path.join(MODEL_DIR, f"{GRASP_TYPE}.pkl")
MONITOR_FILE = f"eval_{GRASP_TYPE}.csv"
SUMMARY_CSV = os.path.join(LOG_DIR, "eval_summary.csv")
SCENARIO_DIR = os.path.join(LOG_DIR, "scenarios")
SCENARIO_MD = os.path.join(BASE_DIR, "scenarios.md")
TRIALS = int(os.environ.get("DEXMOBILE_EVAL_TRIALS", 100))


def normalize_status(success_flag, fail_reason):
    if int(success_flag) == 1:
        return "success"
    if fail_reason in (None, "", "None"):
        return "unknown"
    return str(fail_reason)


def explain_status(status):
    explanations = {
        "success": (
            "Hand reached the handle, formed stable multi-finger contact, and lifted the bottle "
            "above the success-height threshold."
        ),
        "time out": "Policy did not finish reach-grasp-lift before the episode step limit.",
        "out of range": "End-effector drifted from the valid handle region before stable grasp contact.",
        "slipped": "Contact formed initially, but force closure was unstable and the bottle slipped during lift.",
        "press fail": "No stable grasp/contact pattern emerged in the operation phase.",
        "unknown": "Episode ended without an explicit failure code from the environment.",
    }
    return explanations.get(status, f"Episode ended with fail_reason='{status}'.")


def capture_screenshot(path):
    width, height = 960, 720
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.79, -0.455, 0.27],
        distance=0.9,
        yaw=50,
        pitch=-40,
        roll=0,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(width) / float(height),
        nearVal=0.1,
        farVal=3.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    image = np.reshape(np.array(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]
    plt.imsave(path, image)


def write_scenarios_md(scenarios):
    lines = [
        "# Handle-Grasp Scenarios (Object 3381)",
        "",
        "| trial | task_id | grasp | status | reward | explanation | screenshot |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in scenarios:
        lines.append(
            f"| {row['trial']} | {row['tid']} | {row['grasp']} | {row['status']} | "
            f"{row['reward']:.2f} | {row['explanation']} | {row['image']} |"
        )
    lines.append("")
    with open(SCENARIO_MD, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def evaluate():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    os.environ["DEXMOBILE_MONITOR_FILE"] = MONITOR_FILE

    env = Dualenv(renders=True, is_discrete=False, max_steps=1024)
    env = Monitor(env, LOG_DIR, log_extension=MONITOR_FILE)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(ENV_PATH, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=env)

    reason_counts = Counter()
    rows = []
    scenarios = []
    success_scenarios = 0
    failure_reasons_captured = set()

    for trial in range(1, TRIALS + 1):
        obs = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, infos = env.step(action)
            done = bool(done_arr[0]) if isinstance(done_arr, (np.ndarray, list, tuple)) else bool(done_arr)
            info = infos[0] if isinstance(infos, (list, tuple)) and infos else infos

        episode = info.get("episode", {})
        status = normalize_status(episode.get("s", 0), episode.get("f", "unknown"))
        reason_counts[status] += 1

        row = {
            "trial": trial,
            "tid": episode.get("tid", 7),
            "grasp": episode.get("g", GRASP_TYPE),
            "status": status,
            "reward": float(episode.get("r", 0.0)),
            "length": int(episode.get("l", 0)),
            "object_id": int(episode.get("i", 3381)),
            "affordance": episode.get("a", "Handle-grasp"),
        }
        rows.append(row)

        should_capture = False
        if status == "success" and success_scenarios < 2:
            should_capture = True
            success_scenarios += 1
        elif status != "success" and len(failure_reasons_captured) < 3 and status not in failure_reasons_captured:
            should_capture = True
            failure_reasons_captured.add(status)

        if should_capture:
            image_name = f"{status.replace(' ', '_')}_trial{trial}.png"
            image_path = os.path.join(SCENARIO_DIR, image_name)
            capture_screenshot(image_path)
            row["image"] = f"DexMobile/log/scenarios/{image_name}"
            row["explanation"] = explain_status(status)
            scenarios.append(row.copy())

    env.close()

    details_df = pd.DataFrame(rows)
    details_df.to_csv(SUMMARY_CSV, index=False)
    write_scenarios_md(scenarios)

    success_rate = reason_counts["success"] / float(TRIALS)
    print(f"Evaluation complete over {TRIALS} trials")
    print(f"Success rate: {success_rate:.3f}")
    for reason, count in sorted(reason_counts.items()):
        print(f"{reason}: {count}")
    print(f"Detailed per-trial results: {SUMMARY_CSV}")
    print(f"Scenario report: {SCENARIO_MD}")


if __name__ == "__main__":
    evaluate()
