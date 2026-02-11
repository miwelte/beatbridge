#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    PoC: Audio Input (beatbridge_poc_audio_in.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


# -----------------------------
# Config
# -----------------------------
DEVICE_INDEX = 0
SAMPLE_RATE = 48_000
CHANNELS = 2
BLOCK_SIZE = 1024

# Kick emphasis
KICK_LP_HZ = 160.0
ENV_LP_HZ = 12.0

# Onset function = positive derivative of envelope
DIFF_LP_HZ = 18.0
DIFF_THRESH_FACTOR = 6.0
DIFF_THRESH_MIN = 1e-4
REFRACTORY_S = 0.120  # slightly longer, avoids double triggers on same kick

# Tempo
BPM_MIN = 70.0
BPM_MAX = 170.0

# PLL
LOCK_WINDOW_S = 0.080
PLL_ALPHA = 0.22
PLL_BETA = 0.02

# Tempo estimator
ONSET_HISTORY_S = 12.0
IOI_MIN_S = 0.18
IOI_MAX_S = 1.80
BPM_BIN_SIZE = 1.0
MIN_ONSETS_FOR_BPM = 4

# UI
METER_WIDTH = 44
PEAK_HOT_THRESHOLD = 0.80
PEAK_HOLD_DECAY_PER_S = 0.175  # long hold


# -----------------------------
# Filters
# -----------------------------
@dataclass
class OnePoleLPF:
    a: float
    y: float = 0.0

    @staticmethod
    def from_cutoff(cutoff_hz: float, sample_rate: float) -> "OnePoleLPF":
        a = float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
        return OnePoleLPF(a=a, y=0.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float32)
        a = self.a
        yy = self.y
        for i in range(x.size):
            yy = (1.0 - a) * x[i] + a * yy
            y[i] = yy
        self.y = float(yy)
        return y


# -----------------------------
# Derivative-based kick onset detector
# -----------------------------
@dataclass
class KickOnsetDetectorDiff:
    sr: float
    kick_lpf: OnePoleLPF
    env_lpf: OnePoleLPF
    diff_lpf: OnePoleLPF
    # adaptive threshold state for diff signal
    diff_floor: float = 0.0
    last_onset_t: float = -1e9
    prev_env_last: float = 0.0
    prev_diff_last: float = 0.0

    @staticmethod
    def create(sample_rate: float) -> "KickOnsetDetectorDiff":
        return KickOnsetDetectorDiff(
            sr=sample_rate,
            kick_lpf=OnePoleLPF.from_cutoff(KICK_LP_HZ, sample_rate),
            env_lpf=OnePoleLPF.from_cutoff(ENV_LP_HZ, sample_rate),
            diff_lpf=OnePoleLPF.from_cutoff(DIFF_LP_HZ, sample_rate),
        )

    def process_block(self, mono: np.ndarray, block_start_time: float) -> List[Tuple[float, float]]:
        # 1) kick emphasis
        low = self.kick_lpf.process(mono.astype(np.float32))

        # 2) envelope
        env = self.env_lpf.process(np.abs(low))

        # 3) positive derivative onset function
        env_ext = np.concatenate(([self.prev_env_last], env))
        diff = env_ext[1:] - env_ext[:-1]
        diff = np.maximum(diff, 0.0).astype(np.float32)

        # 4) smooth diff a bit
        diff_s = self.diff_lpf.process(diff)

        # 5) adaptive floor based on median of diff_s (this is much safer than env median)
        med = float(np.median(diff_s))
        if self.diff_floor <= 0.0:
            self.diff_floor = med
        else:
            # very slow adapt
            self.diff_floor = 0.995 * self.diff_floor + 0.005 * min(self.diff_floor * 1.05, med)

        thr = max(DIFF_THRESH_MIN, self.diff_floor * DIFF_THRESH_FACTOR)

        # 6) peak-pick on diff_s
        diff_ext = np.concatenate(([self.prev_diff_last], diff_s))
        events: List[Tuple[float, float]] = []
        for i in range(diff_s.size - 1):
            d_prev = diff_ext[i]
            d_now = diff_ext[i + 1]
            d_next = diff_ext[i + 2]
            if d_prev < d_now >= d_next and d_now >= thr:
                t = block_start_time + (i / self.sr)
                if (t - self.last_onset_t) >= REFRACTORY_S:
                    self.last_onset_t = t
                    events.append((t, float(d_now)))

        self.prev_env_last = float(env[-1])
        self.prev_diff_last = float(diff_s[-1])
        return events, thr, self.diff_floor


# -----------------------------
# Tempo estimator
# -----------------------------
@dataclass
class TempoEstimator:
    history_s: float = ONSET_HISTORY_S
    onset_times: List[float] = field(default_factory=list)

    def add_onset(self, t: float) -> None:
        self.onset_times.append(t)
        t_min = t - self.history_s
        while self.onset_times and self.onset_times[0] < t_min:
            self.onset_times.pop(0)

    @staticmethod
    def _fold_bpm(bpm: float) -> float:
        while bpm < BPM_MIN:
            bpm *= 2.0
        while bpm > BPM_MAX:
            bpm *= 0.5
        return bpm

    def estimate_bpm(self) -> Optional[float]:
        if len(self.onset_times) < MIN_ONSETS_FOR_BPM:
            return None

        times = self.onset_times[-32:]
        iois: List[float] = []
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if IOI_MIN_S <= dt <= IOI_MAX_S:
                iois.append(dt)
        if len(iois) < 3:
            return None

        votes = {}
        raw_vals = []
        for dt in iois:
            base = 60.0 / dt
            for mult in (0.5, 1.0, 2.0):
                bpm = self._fold_bpm(base * mult)
                raw_vals.append(bpm)
                b = round(bpm / BPM_BIN_SIZE) * BPM_BIN_SIZE
                votes[b] = votes.get(b, 0) + 1

        best_bin = max(votes.items(), key=lambda kv: kv[1])[0]
        close = [b for b in raw_vals if abs(b - best_bin) <= 1.5 * BPM_BIN_SIZE]
        return float(np.median(close)) if len(close) >= 3 else float(best_bin)


# -----------------------------
# PLL
# -----------------------------
@dataclass
class BeatPLL:
    period_s: Optional[float] = None
    next_beat_t: Optional[float] = None
    locks: int = 0

    def set_bpm(self, bpm: float, t_now: float) -> None:
        period = 60.0 / max(1e-6, bpm)
        if self.period_s is None:
            self.period_s = period
            self.next_beat_t = t_now + period
            self.locks = 0
            return
        self.period_s = 0.90 * self.period_s + 0.10 * period
        if self.next_beat_t is None:
            self.next_beat_t = t_now + self.period_s

    def update_from_onset(self, t: float) -> bool:
        if self.period_s is None or self.next_beat_t is None:
            return False
        p = self.period_s
        candidates = [self.next_beat_t - p, self.next_beat_t, self.next_beat_t + p]
        best_bt = min(candidates, key=lambda bt: abs(t - bt))
        err = t - best_bt
        if abs(err) <= LOCK_WINDOW_S:
            self.next_beat_t = best_bt + p + PLL_ALPHA * err
            self.period_s = p + PLL_BETA * err
            self.locks += 1
            return True
        return False

    def stable_bpm(self) -> Optional[float]:
        return None if self.period_s is None else 60.0 / self.period_s


# -----------------------------
# UI
# -----------------------------
def make_meter(value: float, hold: float, width: int, hold_color: str) -> str:
    v = max(0.0, min(1.0, value))
    h = max(0.0, min(1.0, hold))
    filled = int(round(v * width))
    hold_pos = max(0, min(width - 1, int(round(h * width))))
    bar = ["█"] * filled + ["░"] * (width - filled)
    bar[hold_pos] = f"[{hold_color}]▌[/{hold_color}]"
    return "".join(bar)


def build_panel(
    rms: float, peak: float,
    rms_hold: float, peak_hold: float,
    rms_max: float, peak_max: float,
    bpm_est: Optional[float], bpm_pll: Optional[float],
    locks: int,
    diff_floor: float, diff_thr: float,
) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", width=10)
    tbl.add_column()

    hold_color = "green" if peak < PEAK_HOT_THRESHOLD else "red"

    tbl.add_row("RMS",  make_meter(rms, rms_hold, METER_WIDTH, "green") + f"  {rms:0.4f} ({rms_max:0.4f})")
    tbl.add_row("PEAK", make_meter(peak, peak_hold, METER_WIDTH, hold_color) + f"  {peak:0.4f} ({peak_max:0.4f})")

    est_txt = f"{bpm_est:0.2f}" if bpm_est is not None else "--"
    pll_txt = f"{bpm_pll:0.2f}" if bpm_pll is not None else "--"
    tbl.add_row("BPM", f"[bold]est:{est_txt}[/bold]   pll:{pll_txt}   locks:{locks}")

    tbl.add_row("DiffFloor", f"{diff_floor:0.6f}")
    tbl.add_row("DiffThr",   f"{diff_thr:0.6f}")

    return Panel(tbl, title="Beatbridge PoC (Kick Diff-Onset + BPM)", border_style="cyan")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    console = Console()
    detector = KickOnsetDetectorDiff.create(SAMPLE_RATE)
    tempo = TempoEstimator()
    pll = BeatPLL()

    rms_s = 0.0
    peak_s = 0.0
    rms_hold = 0.0
    peak_hold = 0.0
    rms_max = 0.0
    peak_max = 0.0

    last_t = time.time()
    last_print = time.time()

    bpm_est: Optional[float] = None

    with sd.InputStream(
        device=DEVICE_INDEX,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
    ) as stream, Live(console=console, refresh_per_second=12) as live:

        while True:
            block, overflowed = stream.read(BLOCK_SIZE)
            if overflowed:
                console.log("[yellow]WARNING: overflow (try bigger BLOCK_SIZE)[/yellow]")

            mono = block.mean(axis=1)

            # raw levels in 0..1 float domain
            rms_raw = float(np.sqrt(np.mean(mono * mono)))
            peak_raw = float(np.max(np.abs(mono)))

            # Smooth for UI
            rms_s = 0.85 * rms_s + 0.15 * rms_raw
            peak_s = max(peak_s * 0.6, peak_raw)

            rms_max = max(rms_max, rms_raw)
            peak_max = max(peak_max, peak_raw)

            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            rms_hold = max(rms_hold - PEAK_HOLD_DECAY_PER_S * dt, rms_s)
            peak_hold = max(peak_hold - PEAK_HOLD_DECAY_PER_S * dt, peak_s)

            # Onsets
            onsets, diff_thr, diff_floor = detector.process_block(mono, block_start_time=now)
            for (t_evt, strength) in onsets:
                tempo.add_onset(t_evt)
                bpm_est = tempo.estimate_bpm()
                if bpm_est is not None:
                    pll.set_bpm(bpm_est, t_now=t_evt)
                    pll.update_from_onset(t_evt)

            # Always print BPM status periodically
            if now - last_print > 1.5:
                last_print = now
                console.log(f"BPM_EST={bpm_est}  PLL_BPM={pll.stable_bpm()}  locks={pll.locks}")

            panel = build_panel(
                rms=rms_s, peak=peak_s,
                rms_hold=rms_hold, peak_hold=peak_hold,
                rms_max=rms_max, peak_max=peak_max,
                bpm_est=bpm_est, bpm_pll=pll.stable_bpm(),
                locks=pll.locks,
                diff_floor=diff_floor, diff_thr=diff_thr,
            )
            live.update(panel)

            time.sleep(0.005)


if __name__ == "__main__":
    main()
