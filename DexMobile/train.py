import os

from Dualenv import Dualenv
from Monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from successRateCallBack import successRateCallBack

from warnings import filterwarnings
filterwarnings("ignore")


STEP_PER_EPISODE = 1024
GRASP_TYPE = "poPmAb25"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_NAME = os.path.join(MODEL_DIR, GRASP_TYPE)
ENV_NAME = os.path.join(MODEL_DIR, f"{GRASP_TYPE}.pkl")
MONITOR_FILE = f"{GRASP_TYPE}.csv"
TOTAL_TIMESTEPS = int(os.environ.get("DEXMOBILE_TOTAL_TIMESTEPS", 1_500_000))


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.environ["DEXMOBILE_MONITOR_FILE"] = MONITOR_FILE

    env = Dualenv(renders=False, is_discrete=False, max_steps=STEP_PER_EPISODE)
    env = Monitor(env, LOG_DIR, log_extension=MONITOR_FILE)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=True)

    callback = successRateCallBack(
        successRates=0.9,
        verbose=1,
        check_freq=STEP_PER_EPISODE * 20,
        path=MODEL_DIR,
        n_eval_episodes=50,
        log_filename=MONITOR_FILE,
        metrics_path=LOG_DIR,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=STEP_PER_EPISODE,
        clip_range=0.2,
        batch_size=64,
    )
    model.learn(TOTAL_TIMESTEPS, callback=callback)
    model.save(MODEL_NAME)
    env.save(ENV_NAME)
    env.close()
    print(f"Training complete. Saved model to {MODEL_NAME}.zip and normalizer to {ENV_NAME}")


if __name__ == "__main__":
    main()
