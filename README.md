# Crazyflie Station-Tour via Reinforcement Learning (PyBullet)

Un drone **Crazyflie** apprend, par **PPO** (reinforcement learning), à
**visiter 5 stations en 3D puis revenir à son point de départ**, dans le
simulateur **PyBullet**. Basé sur

## Fichiers du projet (ajouts)
| Fichier | Rôle |
|---|---|
| `gym_pybullet_drones/envs/StationTourAviary.py` | Environnement Gym custom + `TargetAugmentWrapper` |
| `examples/learn_tour.py`  | Entraînement PPO — 2 stations (curriculum) |
| `examples/learn_tour_5.py`| Warm-start du modèle 2-stations → 5 stations |
| `examples/play_tour.py`   | Visualisation 3D (mettre `NUM_STATIONS = 2` ou `5`) |
| `pretrained/best_model_ns5.zip` | Modèle 5-stations **déjà entraîné** |

```bash
conda create -n drones python=3.10
conda activate drones
pip install -e .
pip install "setuptools<81"   # OBLIGATOIRE : sinon "No module named 'pkg_resources'"
pip install tensorboard       # OBLIGATOIRE pour l'entraînement 
```

## Voir le résultat SANS entraîner (30 s)
```bash
cd gym_pybullet_drones/examples
python play_tour.py           # NUM_STATIONS=5 par défaut -> charge pretrained/
```
Le drone (sphère cyan) fait HOME 🟢 → 5 stations 🟠 → HOME 🟢,
guidé par le trait rouge. ~80 % de tours complets.

## Ré-entraîner (optionnel)
```bash
python learn_tour.py          # 2 stations  (TOTAL_TIMESTEPS dans le fichier)
python learn_tour_5.py        # warm-start -> 5 stations
```
Monitorer : `tensorboard --logdir results`  puis  http://localhost:6006

## Résultat typique (5 stations)
- Taux de tours complets : **~80 %**
- Reward tour parfait (`cible 6/6`) : **+60**
- Méthode : curriculum (2→5) + warm-start + reward shaping + démarrage en vol

## Citation du simulateur de base
```bibtex
@INPROCEEDINGS{panerati2021learning,
  title={Learning to Fly---a Gym Environment with PyBullet Physics for
         Reinforcement Learning of Multi-agent Quadcopter Control},
  author={Panerati, Jacopo and Zheng, Hehui and Zhou, SiQi and Xu, James
          and Prorok, Amanda and Schoellig, Angela P.},
  booktitle={2021 IEEE/RSJ IROS}, year={2021}, pages={7512-7519},
  doi={10.1109/IROS51168.2021.9635857}}
```