import os, glob
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.StationTourAviary import (
    StationTourAviary, TargetAugmentWrapper)
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

NUM_STATIONS = 5     # ← 2 pour rejouer le 2-stations, 5 pour le 5-stations



# 1) modèle pré-entraîné fourni, sinon 2) dernier run local
pre = f"pretrained/best_model_ns{NUM_STATIONS}.zip"
candidates = [pre] if os.path.exists(pre) else []
candidates += sorted(glob.glob(f"results/*_ns{NUM_STATIONS}/best_model.zip"),
                     key=os.path.getmtime, reverse=True)
assert candidates, f"Aucun modèle ns{NUM_STATIONS} (pré-entraîné ou entraîné)."
MODEL_PATH = candidates[0]
print("[INFO] Loading", MODEL_PATH)

# prend le dernier modèle entraîné pour CE nombre de stations (plus de mismatch)
pattern = f"results/*_ns{NUM_STATIONS}/best_model.zip"
candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
assert candidates, f"Aucun modèle *_ns{NUM_STATIONS} trouvé. Entraîne d'abord !"
MODEL_PATH = candidates[0]
print("[INFO] Loading", MODEL_PATH)

env = TargetAugmentWrapper(StationTourAviary(
    gui=True, record=False,
    obs=ObservationType.KIN, act=ActionType.PID,
    num_stations=NUM_STATIONS))

n_wp = env.unwrapped.NUM_WAYPOINTS
model = PPO.load(MODEL_PATH)

obs, _ = env.reset()
for i in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 30 == 0:
        print(f"Step {i:4d} | cible {info['target_idx']}/{n_wp} | "
              f"dist {info['dist_to_target']:.2f} | r {reward:.2f}")
    if terminated or truncated:
        print(f"  -> FIN episode (tour complet = {info['tour_complete']})")
        obs, _ = env.reset()
env.close()
