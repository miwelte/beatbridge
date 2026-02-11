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

# Kick detector
KICK_LP_HZ = 150.0
ENV_LP_HZ = 10.0
THRESH_FACTOR = 6.0
THRESH_MIN = 0.002
REFRACTORY_S = 0.090

# Tempo constraints
BPM_MIN = 70.0
BPM_MAX = 170.0

# PLL behavior
LOCK_WINDOW_S = 0.060          # slightly wider for reliability
PLL_ALPHA = 0.18               # phase correction strength
PLL_BETA = 0.02                # period correction strength
MIN_LOCKS_TO_SHOW = 4

# Tempo estimation (robust)
ONSET_HISTORY_S = 12.0         # how many seconds of onsets to keep
IOI_MIN_S = 0.20               # 300 bpm upper bound-ish (we'll fold anyway)
IOI_MAX_S = 1.20               # 50 bpm lower bound-ish (we'll fold anyway)
BPM_BIN_SIZE = 1.0             # histogram resolution

# UI
METER_WIDTH = 44
PEAK_HOT_THRESHOLD = 0.80      # peak-hold marker becomes red above this
PEAK_HOLD_DECAY_PER_S = 0.175  # doubled hold time (slower decay than before)


# -----------------------------
# Simple 1st order lowpass (streaming)
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
# Kick-focused onset detector
# -----------------------------
@dataclass
class KickOnsetDetector:
    sr: float
    kick_lpf: OnePoleLPF
    env_lpf: OnePoleLPF
    noise_floor: float = 0.0
    last_onset_t: float = -1e9
    prev_env_last: float = 0.0

    @staticmethod
    def create(sample_rate: float, kick_lp_hz: float, env_lp_hz: float) -> "KickOnsetDetector":
        return KickOnsetDetector(
            sr=sample_rate,
            kick_lpf=OnePoleLPF.from_cutoff(kick_lp_hz, sample_rate),
            env_lpf=OnePoleLPF.from_cutoff(env_lp_hz, sample_rate),
        )

    def process_block(self, mono: np.ndarray, block_start_time: float) -> List[Tuple[float, float]]:
        low = self.kick_lpf.process(mono.astype(np.float32))
        rect = np.abs(low)
        env = self.env_lpf.process(rect)

        block_med = float(np.median(env))
        if self.noise_floor <= 0.0:
            self.noise_floor = block_med
        else:
            nf_target = min(self.noise_floor * 1.02, block_med)
            self.noise_floor = 0.995 * self.noise_floor + 0.005 * nf_target

        thr = max(THRESH_MIN, self.noise_floor * THRESH_FACTOR)

        env_ext = np.concatenate(([self.prev_env_last], env))
        events: List[Tuple[float, float]] = []

        for i in range(env.size - 1):
            e_prev = env_ext[i]
            e_now = env_ext[i + 1]
            e_next = env_ext[i + 2]
            if e_prev < e_now >= e_next and e_now >= thr:
                t = block_start_time + (i / self.sr)
                if (t - self.last_onset_t) >= REFRACTORY_S:
                    self.last_onset_t = t
                    events.append((t, float(e_now)))

        self.prev_env_last = float(env[-1])
        return events


# -----------------------------
# Robust tempo estimator from onset history
# -----------------------------
@dataclass
class TempoEstimator:
    history_s: float = ONSET_HISTORY_S
    onset_times: List[float] = field(default_factory=list)

    def add_onset(self, t: float) -> None:
        self.onset_times.append(t)
        self._trim(t)

    def _trim(self, t_now: float) -> None:
        t_min = t_now - self.history_s
        # keep only recent onsets
        while self.onset_times and self.onset_times[0] < t_min:
            self.onset_times.pop(0)

    @staticmethod
    def _fold_bpm(bpm: float) -> float:
        # fold into [BPM_MIN, BPM_MAX]
        while bpm < BPM_MIN:
            bpm *= 2.0
        while bpm > BPM_MAX:
            bpm *= 0.5
        return bpm

    def estimate_bpm(self) -> Optional[float]:
        n = len(self.onset_times)
        if n < 6:
            return None

        # build IOIs between near neighbors (robust, avoids long gaps dominating)
        # use up to last ~24 onsets
        times = self.onset_times[-24:]
        iois: List[float] = []
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if IOI_MIN_S <= dt <= IOI_MAX_S:
                iois.append(dt)

        if len(iois) < 4:
            return None

        # histogram vote in BPM domain, considering x0.5/x1/x2 to address half/double-time
        # Example: if we detect every 2nd beat, IOI doubles -> folding can land wrong; voting fixes it.
        bpm_votes = {}
        for dt in iois:
            base = 60.0 / dt
            for mult in (0.5, 1.0, 2.0):
                bpm = self._fold_bpm(base * mult)
                b = round(bpm / BPM_BIN_SIZE) * BPM_BIN_SIZE
                bpm_votes[b] = bpm_votes.get(b, 0) + 1

        if not bpm_votes:
            return None

        # choose mode bin; then refine with median of contributing values near that bin
        best_bin = max(bpm_votes.items(), key=lambda kv: kv[1])[0]

        close_vals = []
        for dt in iois:
            base = 60.0 / dt
            for mult in (0.5, 1.0, 2.0):
                bpm = self._fold_bpm(base * mult)
                if abs(bpm - best_bin) <= (1.5 * BPM_BIN_SIZE):
                    close_vals.append(bpm)

        if len(close_vals) < 4:
            return float(best_bin)

        return float(np.median(close_vals))


# -----------------------------
# Beat PLL (locks grid to onsets near predicted beats)
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

        # gently steer period towards estimator
        self.period_s = 0.90 * self.period_s + 0.10 * period
        if self.next_beat_t is None:
            self.next_beat_t = t_now + self.period_s

    def update_from_onset(self, t: float) -> bool:
        if self.period_s is None or self.next_beat_t is None:
            return False

        p = self.period_s

        # allow recovery if we're one beat off: check previous/current/next predicted beat
        candidates = [self.next_beat_t - p, self.next_beat_t, self.next_beat_t + p]
        best_bt = min(candidates, key=lambda bt: abs(t - bt))
        phase_err = t - best_bt

        if abs(phase_err) <= LOCK_WINDOW_S:
            # apply PLL corrections
            self.next_beat_t = best_bt + p + PLL_ALPHA * phase_err
            self.period_s = p + PLL_BETA * phase_err
            self.locks += 1
            return True

        return False

    def generate_beats_up_to(self, t_now: float) -> List[float]:
        if self.period_s is None or self.next_beat_t is None:
            return []
        beats = []
        while self.next_beat_t <= t_now:
            beats.append(self.next_beat_t)
            self.next_beat_t += self.period_s
        return beats

    def stable_bpm(self) -> Optional[float]:
        if self.period_s is None:
            return None
        return 60.0 / self.period_s


# -----------------------------
# UI helpers
# -----------------------------
def make_meter(value: float, hold: float, width: int, hold_color: str) -> str:
    """
    value, hold in [0..1]. Draw horizontal bar with a peak-hold marker.
    """
    v = max(0.0, min(1.0, value))
    h = max(0.0, min(1.0, hold))

    filled = int(round(v * width))
    hold_pos = int(round(h * width))
    hold_pos = max(0, min(width - 1, hold_pos))

    bar = ["█"] * filled + ["░"] * (width - filled)
    bar[hold_pos] = f"[{hold_color}]▌[/{hold_color}]"
    return "".join(bar)


def build_panel(
    rms_norm: float,
    peak_norm: float,
    bpm: Optional[float],
    rms_hold_norm: float,
    peak_hold_norm: float,
    rms_raw: float,
    peak_raw: float,
    rms_max: float,
    peak_max: float,
    noise_floor: float,
    thr: float,
    locks: int,
) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", width=12)
    tbl.add_column()

    # Hold marker color rule based on current peak_raw
    hold_color = "green" if peak_raw < PEAK_HOT_THRESHOLD else "red"

    tbl.add_row(
        "RMS",
        make_meter(rms_norm, rms_hold_norm, METER_WIDTH, hold_color="green")
        + f"  {rms_raw:0.4f} ({rms_max:0.4f})"
    )
    tbl.add_row(
        "PEAK",
        make_meter(peak_norm, peak_hold_norm, METER_WIDTH, hold_color=hold_color)
        + f"  {peak_raw:0.4f} ({peak_max:0.4f})"
    )

    bpm_txt = f"{bpm:0.2f}" if bpm is not None and locks >= MIN_LOCKS_TO_SHOW else "--"
    lock_txt = f"{locks}"
    tbl.add_row("BPM", f"[bold]{bpm_txt}[/bold]    locks: {lock_txt}")

    tbl.add_row("NoiseFloor", f"{noise_floor:0.6f}")
    tbl.add_row("Threshold", f"{thr:0.6f}")

    return Panel(tbl, title="Beatbridge PoC (Kick Onset + Robust BPM + PLL)", border_style="cyan")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    console = Console()

    kick = KickOnsetDetector.create(SAMPLE_RATE, KICK_LP_HZ, ENV_LP_HZ)
    tempo = TempoEstimator()
    pll = BeatPLL()

    # Level stats
    rms_val = 0.0
    peak_val = 0.0
    rms_hold = 0.0
    peak_hold = 0.0
    rms_max = 0.0
    peak_max = 0.0
    last_ui_t = time.time()

    # Normalize functions for meters
    def norm_rms(x: float) -> float:
        # 0..0.25 RMS is already quite hot; scale to [0..1]
        return max(0.0, min(1.0, x / 0.25))

    def norm_peak(x: float) -> float:
        return max(0.0, min(1.0, x))

    with sd.InputStream(
        device=DEVICE_INDEX,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
    ) as stream, Live(console=console, refresh_per_second=12) as live:
        t0 = time.time()

        while True:
            block, overflowed = stream.read(BLOCK_SIZE)
            if overflowed:
                console.log("[yellow]WARNING: overflow (try bigger BLOCK_SIZE)[/yellow]")

            mono = block.mean(axis=1)

            # instantaneous block levels
            rms_inst = float(np.sqrt(np.mean(mono * mono)))
            peak_inst = float(np.max(np.abs(mono)))

            # smooth for UI
            rms_val = 0.85 * rms_val + 0.15 * rms_inst
            peak_val = max(peak_val * 0.6, peak_inst)

            # record maxima (raw)
            rms_max = max(rms_max, rms_inst)
            peak_max = max(peak_max, peak_inst)

            # peak-hold decay
            now = time.time()
            dt = max(1e-3, now - last_ui_t)
            last_ui_t = now

            rms_hold = max(rms_hold - PEAK_HOLD_DECAY_PER_S * dt, rms_val)
            peak_hold = max(peak_hold - PEAK_HOLD_DECAY_PER_S * dt, peak_val)

            # onset detection
            onsets = kick.process_block(mono, block_start_time=now)

            # feed tempo estimator + PLL
            for (t_evt, strength) in onsets:
                tempo.add_onset(t_evt)

                bpm_est = tempo.estimate_bpm()
                if bpm_est is not None:
                    pll.set_bpm(bpm_est, t_now=t_evt)

                locked = pll.update_from_onset(t_evt)
                if locked:
                    console.log(f"[green]LOCK[/green] t={t_evt - t0:7.3f}s  bpm≈{(pll.stable_bpm() or 0):0.2f}")

            # emit beat ticks if needed (optional logging; can be noisy)
            # beats = pll.generate_beats_up_to(now)
            # for bt in beats:
            #     bpm = pll.stable_bpm()
            #     if bpm is not None:
            #         console.log(f"BEAT t={bt - t0:7.3f}s  bpm≈{bpm:0.2f}")

            bpm_show = pll.stable_bpm()
            thr = max(THRESH_MIN, kick.noise_floor * THRESH_FACTOR)

            panel = build_panel(
                rms_norm=norm_rms(rms_val),
                peak_norm=norm_peak(peak_val),
                bpm=bpm_show,
                rms_hold_norm=norm_rms(rms_hold),
                peak_hold_norm=norm_peak(peak_hold),
                rms_raw=rms_inst,
                peak_raw=peak_inst,
                rms_max=rms_max,
                peak_max=peak_max,
                noise_floor=kick.noise_floor,
                thr=thr,
                locks=pll.locks,
            )
            live.update(panel)

            time.sleep(0.005)


if __name__ == "__main__":
    main()
