"""Benchmark radio Monte-Carlo pour la collecte Crazyflie.

 python crazyflie_radio_benchmark.py --trials 1000 --output radio_resultats
Ce modèle doit être calibré avec des RSSI et des pertes mesurés en vol.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RadioConfig:
    frequency_hz: float = 2.4e9
    sensitivity_dbm: float = -85.0       # hypothèse, à remplacer selon le capteur
    fading_margin_db: float = 10.0
    path_loss_exponent: float = 2.2
    shadowing_sigma_db: float = 4.0
    payload_bytes: int = 240
    packet_payload_bytes: int = 60        # limite P2P Bitcraze
    retry_attempts: int = 3
    packet_period_s: float = .01
    dwell_s: float = 2.0


def fspl_1m(cfg: RadioConfig) -> float:
    return 20 * math.log10(4 * math.pi * cfg.frequency_hz / 299792458.0)


def packet_success_probability(rssi_dbm: float, sensitivity_dbm: float) -> float:
    """Courbe logistique, à remplacer après mesure réelle RSSI/PER."""
    return float(1 / (1 + math.exp(-(rssi_dbm - sensitivity_dbm) / 3.0)))


def run(cfg: RadioConfig, trials: int, seed: int) -> list[dict]:
    rng, rows = np.random.default_rng(seed), []
    packets = math.ceil(cfg.payload_bytes / cfg.packet_payload_bytes)
    for power in (-10., 0., 10., 20.):
        for distance in np.arange(1., 51., 1.):
            rssis = power - fspl_1m(cfg) - 10*cfg.path_loss_exponent*np.log10(distance) + rng.normal(0, cfg.shadowing_sigma_db, trials)
            p_single = np.array([packet_success_probability(r, cfg.sensitivity_dbm) for r in rssis])
            p_after_retry = 1 - (1-p_single)**cfg.retry_attempts
            received = rng.binomial(packets, p_after_retry)
            rows.append({"tx_power_dbm": power, "distance_m": distance,
                         "rssi_mean_dbm": float(np.mean(rssis)), "rssi_std_db": float(np.std(rssis)),
                         "packet_delivery_rate": float(np.mean(p_after_retry)),
                         "collection_success_rate": float(np.mean(received == packets)),
                         "transfer_time_s": packets*cfg.packet_period_s,
                         "collection_within_dwell": packets*cfg.packet_period_s <= cfg.dwell_s})
    return rows


def plot(rows: list[dict], cfg: RadioConfig, output: Path) -> None:
    powers = sorted(set(r["tx_power_dbm"] for r in rows)); fig, axes = plt.subplots(2, 2, figsize=(11, 7)); axes=axes.ravel()
    for power in powers:
        r = [x for x in rows if x["tx_power_dbm"] == power]; d=[x["distance_m"] for x in r]
        axes[0].plot(d,[x["rssi_mean_dbm"] for x in r],label=f"{power:.0f} dBm")
        axes[1].plot(d,[100*x["packet_delivery_rate"] for x in r],label=f"{power:.0f} dBm")
        axes[2].plot(d,[100*x["collection_success_rate"] for x in r],label=f"{power:.0f} dBm")
    axes[0].axhline(cfg.sensitivity_dbm, color="tab:red", ls="--", label="sensibilité")
    axes[0].set(ylabel="RSSI moyen (dBm)", xlabel="Distance (m)")
    axes[1].set(ylabel="Livraison de paquet (%)", xlabel="Distance (m)", ylim=(0,102))
    axes[2].set(ylabel="Collecte complète (%)", xlabel="Distance (m)", ylim=(0,102))
    axes[3].axis("off"); axes[3].text(.02,.75, f"Paquets par capteur : {math.ceil(cfg.payload_bytes/cfg.packet_payload_bytes)}\n"
                 f"Retries : {cfg.retry_attempts}\nTemps de transfert : {math.ceil(cfg.payload_bytes/cfg.packet_payload_bytes)*cfg.packet_period_s:.2f} s\n"
                 f"Stationnaire : {cfg.dwell_s:.2f} s\n\nModèle à calibrer : RSSI/PER mesurés", transform=axes[3].transAxes)
    for ax in axes[:3]: ax.grid(alpha=.25); ax.legend(title="Puissance TX")
    fig.suptitle("Benchmark radio 2,4 GHz — modèle Monte-Carlo")
    fig.tight_layout(); fig.savefig(output/"radio_benchmark.png",dpi=180); plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--trials",type=int,default=1000); p.add_argument("--seed",type=int,default=2026); p.add_argument("--output",type=Path,default=Path("radio_resultats")); args=p.parse_args()
    cfg=RadioConfig(); args.output.mkdir(parents=True,exist_ok=True); rows=run(cfg,args.trials,args.seed)
    with (args.output/"radio_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    plot(rows,cfg,args.output)
    print(f"Terminé : {args.output.resolve()} (radio_results.csv, radio_benchmark.png)")


if __name__=="__main__": main()
