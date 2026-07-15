# Simulation Crazyflie — commandes

## Préparation

Depuis le dossier contenant les scripts :

```bash
pip install numpy matplotlib
```

## 1. Simulation 3D simple (capteurs, retour à la base)

```bash
python crazyflie_sensor_mission.py
```

Modifiez les paramètres directement dans la classe `Config` : nombre de capteurs, vitesse, accélération, durée de collecte et graine aléatoire.

## 2. Benchmark sans obstacles

Test rapide :

```bash
python crazyflie_benchmark.py --missions 10 --sensors 6 --output resultats_benchmark
```

Campagne plus robuste :

```bash
python crazyflie_benchmark.py --missions 100 --sensors 8 --output resultats_benchmark
```

Fichiers produits : `benchmark.csv` et `benchmark.png`.

## 3. Comparaison A* / RRT* avec obstacles 3D

Exécution rapide :

```bash
python crazyflie_obstacles_benchmark.py --missions 1 --sensors 5 --obstacles 5 --rrt-iterations 300
```

Exécution intermédiaire :

```bash
python crazyflie_obstacles_benchmark.py --missions 5 --sensors 6 --obstacles 7 --rrt-iterations 600 --output resultats_obstacles
```

Fichiers produits : `benchmark_obstacles.csv` et `mission_3d.png`.

> Plus `--rrt-iterations` est grand, plus RRT* peut améliorer sa route, mais plus le calcul est long.

## 4. Campagne complète pour le rapport

Validation rapide :

```bash
python crazyflie_experiment_suite.py --profile quick --repetitions 2 --output rapport_resultats
```

Campagne recommandée :

```bash
python crazyflie_experiment_suite.py --profile standard --repetitions 5 --output rapport_resultats
```

Campagne lourde :

```bash
python crazyflie_experiment_suite.py --profile full --repetitions 5 --output rapport_resultats
```

Fichiers produits dans `rapport_resultats` :

- `all_results.csv` : valeurs de chaque mission ;
- `summary.csv` : moyennes et écarts-types ;
- `summary.png` : comparaison globale A* / RRT* ;
- `scene_*.png` : une image 3D par mission.

## 5. Rapport Overleaf

1. Créez un projet Overleaf vide.
2. Importez `rapport_crazyflie.tex`.
3. Importez `summary.png` et les images `scene_*.png` de `rapport_resultats`.
4. Compilez avec **pdfLaTeX**.
5. Reportez les valeurs de `summary.csv` dans le tableau du rapport.

## Ordre conseillé

```bash
python crazyflie_sensor_mission.py
python crazyflie_obstacles_benchmark.py --missions 1 --rrt-iterations 300
python crazyflie_experiment_suite.py --profile standard --repetitions 5 --output rapport_resultats
```
