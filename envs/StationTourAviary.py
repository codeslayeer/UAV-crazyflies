import numpy as np
import pybullet as p
import gymnasium as gym

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType
)


class StationTourAviary(BaseRLAviary):
    """1 Crazyflie : visite N stations puis revient au HOME."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.PID,
                 num_stations: int = 5,
                 station_radius: float = 0.30,        # ← v3.2 : 0.25 -> 0.30
                 episode_len_sec: int = 20,
                 ):

        self.STATION_RADIUS = station_radius
        self.EPISODE_LEN_SEC = episode_len_sec

        self.HOME = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        ALL_STATIONS = np.array([
            [ 0.6,  0.0, 0.7],
            [ 0.4,  0.5, 0.6],
            [-0.3,  0.5, 0.8],
            [-0.6,  0.0, 0.6],
            [-0.2, -0.5, 0.7],
        ], dtype=np.float32)
        self.STATIONS = ALL_STATIONS[:num_stations]
        self.WAYPOINTS = np.vstack([self.STATIONS, self.HOME])
        self.NUM_WAYPOINTS = len(self.WAYPOINTS)

        self.current_target_idx = 0
        self.prev_dist = None
        self._wp_bodies = []
        self._drone_marker = None
        self._target_marker = None
        self._line_id = -1

        # ✅ v3.2 : on démarre EN L'AIR, stabilisé au HOME (pas de décollage)
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.5]])
        if initial_rpys is None:
            initial_rpys = np.array([[0.0, 0.0, 0.0]])

        super().__init__(
            drone_model=drone_model, num_drones=1,
            initial_xyzs=initial_xyzs, initial_rpys=initial_rpys,
            physics=physics, pyb_freq=pyb_freq, ctrl_freq=ctrl_freq,
            gui=gui, record=record, obs=obs, act=act,
        )

    def _gui(self):
        return getattr(self, 'GUI', getattr(self, 'gui', False))

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_target_idx = 0
        self.prev_dist = float(np.linalg.norm(self.pos[0] - self.WAYPOINTS[0]))
        if self._gui():
            self._draw_scene()
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        if self._gui():
            self._update_visuals()
        return obs, rew, term, trunc, info

    # ===================== VISUALISATION 3D (v3.1 : sans spam) ========
    def _draw_scene(self):
        self._wp_bodies = []
        for i, wp in enumerate(self.WAYPOINTS):
            is_home = (i == self.NUM_WAYPOINTS - 1)
            col = [0.0, 0.8, 0.0, 0.6] if is_home else [1.0, 0.5, 0.0, 0.6]
            vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.06,
                                     rgbaColor=col, physicsClientId=self.CLIENT)
            bid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs,
                                    basePosition=wp, physicsClientId=self.CLIENT)
            self._wp_bodies.append(bid)

        vs_d = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                   rgbaColor=[0, 1, 1, 0.9],
                                   physicsClientId=self.CLIENT)
        self._drone_marker = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vs_d,
            basePosition=self.pos[0].tolist(), physicsClientId=self.CLIENT)

        idx = min(self.current_target_idx, self.NUM_WAYPOINTS - 1)
        vs_t = p.createVisualShape(p.GEOM_SPHERE, radius=0.08,
                                   rgbaColor=[1, 0, 0, 0.9],
                                   physicsClientId=self.CLIENT)
        self._target_marker = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vs_t,
            basePosition=self.WAYPOINTS[idx].tolist(),
            physicsClientId=self.CLIENT)

        self._line_id = p.addUserDebugLine(
            self.pos[0].tolist(), self.WAYPOINTS[idx].tolist(),
            lineColorRGB=[1, 0, 0], lineWidth=3, physicsClientId=self.CLIENT)

        try:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.6, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.6],
                physicsClientId=self.CLIENT)
        except Exception:
            pass

    def _update_visuals(self):
        try:
            pos = self.pos[0].tolist()
            idx = min(self.current_target_idx, self.NUM_WAYPOINTS - 1)
            tgt = self.WAYPOINTS[idx].tolist()
            p.resetBasePositionAndOrientation(
                self._drone_marker, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
            p.resetBasePositionAndOrientation(
                self._target_marker, tgt, [0, 0, 0, 1], physicsClientId=self.CLIENT)
            self._line_id = p.addUserDebugLine(
                pos, tgt, lineColorRGB=[1, 0, 0], lineWidth=3,
                replaceItemUniqueId=self._line_id, physicsClientId=self.CLIENT)
        except Exception:
            pass

    # ===================== ÉTAT / CRASH ===============================
    def _rpy(self):
        return np.array(p.getEulerFromQuaternion(self.quat[0]))

    def _is_crashed(self):
        pos, rpy = self.pos[0], self._rpy()
        return (pos[2] < 0.05 or pos[2] > 2.0 or
                abs(rpy[0]) > 0.6 or abs(rpy[1]) > 0.6)

    # ===================== REWARD (v3.2 : + filet anti-oscillation) ===
    def _computeReward(self):
        pos, vel = self.pos[0], self.vel[0]
        idx = min(self.current_target_idx, self.NUM_WAYPOINTS - 1)
        target = self.WAYPOINTS[idx]
        dist = float(np.linalg.norm(pos - target))

        r = -0.005                                       # pression temporelle
        if self.prev_dist is not None:                   # se rapprocher = +
            r += 4.0 * (self.prev_dist - dist)
        self.prev_dist = dist

        # ✅ v3.2 : filet anti-oscillation (tes 0.26-0.33 qui ne validaient pas)
        if dist < self.STATION_RADIUS * 1.5:
            r += 0.02

        if dist < self.STATION_RADIUS:                   # ✅ station validée
            r += 10.0
            self.current_target_idx += 1
            if self.current_target_idx >= self.NUM_WAYPOINTS:
                r += 50.0                                # 🎉 tour complet
                self.prev_dist = 0.0
            else:
                self.prev_dist = float(np.linalg.norm(
                    pos - self.WAYPOINTS[self.current_target_idx]))

        r -= 0.0005 * float(np.linalg.norm(vel))
        if self._is_crashed():
            r -= 2.0
        if float(np.linalg.norm(pos)) > 2.5:
            r -= 2.0
        return r

    def _computeTerminated(self):
        return self._is_crashed() or \
               self.current_target_idx >= self.NUM_WAYPOINTS

    def _computeTruncated(self):
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)

    def _computeInfo(self):
        idx = min(self.current_target_idx, self.NUM_WAYPOINTS - 1)
        return {
            "target_idx": self.current_target_idx,
            "dist_to_target": float(np.linalg.norm(
                self.pos[0] - self.WAYPOINTS[idx])),
            "tour_complete": self.current_target_idx >= self.NUM_WAYPOINTS,
        }


# ============== WRAPPER : injecte la CIBLE dans l'obs ================
class TargetAugmentWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n + 4,), dtype=np.float32)

    def observation(self, obs):
        base = self.env.unwrapped
        idx = min(base.current_target_idx, base.NUM_WAYPOINTS - 1)
        rel = base.WAYPOINTS[idx] - base.pos[0]
        idx_n = np.array([idx / max(1, base.NUM_WAYPOINTS - 1)],
                         dtype=np.float32)
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([flat, rel.astype(np.float32), idx_n])
