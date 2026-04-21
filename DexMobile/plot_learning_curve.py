import os
import sys

from warnings import filterwarnings
filterwarnings("ignore")

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required to plot learning curves. Install it with `pip install matplotlib`."
    ) from exc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
GRASP_TYPE = "poPmAb25"
CSV_PATH = os.path.join(LOG_DIR, f"{GRASP_TYPE}.csv")
REWARD_PLOT = os.path.join(LOG_DIR, "learning_curve_reward.png")
SUCCESS_PLOT = os.path.join(LOG_DIR, "learning_curve_success.png")
WINDOW = 50


def plot_curves():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Training log not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError(f"Training log is empty: {CSV_PATH}")

    episodes = range(1, len(df) + 1)
    reward_roll = df["r"].rolling(window=WINDOW, min_periods=1).mean()
    success_roll = df["s"].rolling(window=WINDOW, min_periods=1).mean()

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, df["r"], alpha=0.25, label="Episode Reward")
    plt.plot(episodes, reward_roll, linewidth=2.0, label=f"Reward Rolling Mean ({WINDOW})")
    plt.title("Handle-Grasp Training Reward Curve (Object 3381)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REWARD_PLOT, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, success_roll, linewidth=2.0, label=f"Success Rate Rolling Mean ({WINDOW})")
    plt.ylim(-0.05, 1.05)
    plt.title("Handle-Grasp Training Success Curve (Object 3381)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SUCCESS_PLOT, dpi=160)
    plt.close()

    print(f"Saved reward curve: {REWARD_PLOT}")
    print(f"Saved success curve: {SUCCESS_PLOT}")


if __name__ == "__main__":
    try:
        plot_curves()
    except Exception as exc:  # pragma: no cover - script-level reporting
        print(f"Failed to generate learning curves: {exc}")
        sys.exit(1)
