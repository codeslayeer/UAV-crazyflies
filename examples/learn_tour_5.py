import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.envs.StationTourAviary import (
    StationTourAviary, TargetAugmentWrapper)
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OBS = ObservationType.KIN
DEFAULT_ACT = ActionType.PID
OUTPUT_FOLDER = "results"

NUM_STATIONS = 5
TOTAL_TIMESTEPS = int(2e6)

# ✅ cerveau 2-stations déjà entraîné (dossier renommé en _ns2 à l'étape (a))
PREV_MODEL = r"results\07.21.2026_04.23.25_ns2\best_model.zip"

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    filename = os.path.join(OUTPUT_FOLDER, f"{timestamp}_ns{NUM_STATIONS}")
    os.makedirs(filename, exist_ok=True)

    train_env = make_vec_env(
        StationTourAviary,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT,
                        num_stations=NUM_STATIONS, station_radius=0.30),
        wrapper_class=TargetAugmentWrapper,
        n_envs=4, seed=0,
    )
    eval_env = TargetAugmentWrapper(Monitor(StationTourAviary(
        obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False,
        num_stations=NUM_STATIONS)))

    # ---- WARM-START : on charge le cerveau 2-stations ----
    print('[INFO] Warm-start from', PREV_MODEL)
    model = PPO.load(PREV_MODEL)
    model.set_env(train_env)                              # branché sur 5 stations
    model.tensorboard_log = os.path.join(filename, 'tb')  # logs dans le run 5

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=1000, verbose=1),
        verbose=1,
        best_model_save_path=filename, log_path=filename,
        eval_freq=4000, deterministic=True, render=False,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback, log_interval=50,
                reset_num_timesteps=True)

    model.save(os.path.join(filename, 'final_model'))
    print(f"[INFO] Model saved in {filename}")
