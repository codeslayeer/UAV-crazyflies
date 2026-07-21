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
TOTAL_TIMESTEPS = int(1e6)     # ← 200 000 steps (change ici : 5e5, 2e6...)
NUM_STATIONS = 2

if __name__ == "__main__":

    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    filename = os.path.join(OUTPUT_FOLDER, f"{timestamp}_ns{NUM_STATIONS}")
    os.makedirs(filename, exist_ok=True)

    train_env = make_vec_env(
        StationTourAviary,
        env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT,
                        num_stations=2, station_radius=0.25),
        wrapper_class=TargetAugmentWrapper,
        n_envs=4, seed=0,
    )

    eval_env = TargetAugmentWrapper(Monitor(StationTourAviary(
        obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False)))

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    model = PPO('MlpPolicy', train_env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            gamma=0.99, ent_coef=0.02, verbose=1,           # ← 0.01 → 0.02
            tensorboard_log=os.path.join(filename, 'tb'))

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=1000, verbose=1),
        verbose=1,
        best_model_save_path=filename, log_path=filename,
        eval_freq=4000, deterministic=True, render=False,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback, log_interval=50)

    model.save(os.path.join(filename, 'final_model'))
    print(f"[INFO] Model saved in {filename}")
