import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.StationTourAviary import StationTourAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ✅ Modèle v1 en dur (celui qui plafonnait à reward ~5)
MODEL_PATH = r"results\07.20.2026_23.43.55\best_model.zip"

# ✅ PAS de TargetAugmentWrapper → obs = (1, 57), compatible v1
env = StationTourAviary(
    gui=True,
    record=False,
    obs=ObservationType.KIN,
    act=ActionType.PID,
    num_stations=5,
    station_radius=0.2,
)

model = PPO.load(MODEL_PATH)
print("[INFO] Loaded v1 model:", MODEL_PATH)
print("[INFO] Obs space:", env.observation_space)

obs, _ = env.reset()
for i in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if i % 30 == 0:
        print(f"Step {i:4d} | cible {info['target_idx']}/6 | "
              f"dist {info['dist_to_target']:.2f} | r {reward:.2f}")

    if terminated or truncated:
        print(f"  -> FIN episode (tour complet = {info['tour_complete']})")
        obs, _ = env.reset()

env.close()