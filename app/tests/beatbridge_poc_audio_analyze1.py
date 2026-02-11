#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
    BEATBRIDGE
    PoC: Audio Input (beatbridge_poc_audio_in.py)
"""
__author__ = "mail@michael.welte.de"
__copyright__ = "Copyright © 2025-2026 by Michael Welte. All rights reserved."


import time
from dataclasses import dataclass
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

# Kick detector (low emphasis)
KICK_LP_HZ = 150.0
ENV_LP_HZ = 10.0
THRESH_FACTOR = 6.0
THRESH_MIN = 0.002
REFRACTORY_S = 0.090

# Beat/tempo tracking (PLL)
BPM_MIN = 70.0
BPM_MAX = 170.0
LOCK_WINDOW_S = 0.055
PLL_ALPHA = 0.12
PLL_BETA = 0.015
BEATS_PER_BAR = 4  # for 1..4 counter only (no bar detection)

# UI
METER_WIDTH = 44
PEAK_HOLD_DECAY_PER_S = 0.35  # how fast peak-hold falls (in normalized units per second)


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
# Beat PLL (locks tempo to onsets near predicted beats)
# -----------------------------
@dataclass
class BeatPLL:
    period_s: Optional[float] = None
    next_beat_t: Optional[float] = None
    beat_idx: int = 0  # increments on each emitted beat tick

    def _fold_bpm(self, bpm: float) -> float:
        while bpm < BPM_MIN:
            bpm *= 2.0
        while bpm > BPM_MAX:
            bpm *= 0.5
        return bpm

    def bootstrap_from_two_onsets(self, t_prev: float, t_now: float) -> Optional[float]:
        ioi = t_now - t_prev
        if 0.25 <= ioi <= 2.0:
            bpm = self._fold_bpm(60.0 / ioi)
            self.period_s = 60.0 / bpm
            self.next_beat_t = t_now + self.period_s
            return bpm
        return None

    def update_from_onset(self, t: float) -> Optional[Tuple[float, bool]]:
        if self.period_s is None or self.next_beat_t is None:
            return None

        phase_err = t - self.next_beat_t
        if abs(phase_err) <= LOCK_WINDOW_S:
            self.next_beat_t += PLL_ALPHA * phase_err
            self.period_s += PLL_BETA * phase_err

            bpm = 60.0 / max(0.25, min(2.0, self.period_s))
            bpm = self._fold_bpm(bpm)
            self.period_s = 60.0 / bpm
            return (bpm, True)

        return None

    def generate_beats_up_to(self, t_now: float) -> List[float]:
        if self.period_s is None or self.next_beat_t is None:
            return []
        beats = []
        while self.next_beat_t <= t_now:
            beats.append(self.next_beat_t)
            self.beat_idx += 1
            self.next_beat_t += self.period_s
        return beats

    def stable_bpm(self) -> Optional[float]:
        if self.period_s is None:
            return None
        return 60.0 / self.period_s

    def beat_in_bar(self) -> Optional[int]:
        # 1..4 counter (no bar detection)
        if self.period_s is None:
            return None
        return (self.beat_idx % BEATS_PER_BAR) + 1


# -----------------------------
# UI helpers
# -----------------------------
def make_meter(value: float, hold: float, width: int) -> str:
    """
    value, hold in [0..1]. Draw horizontal bar with a red peak-hold marker.
    """
    v = max(0.0, min(1.0, value))
    h = max(0.0, min(1.0, hold))
    filled = int(round(v * width))
    hold_pos = int(round(h * width))
    hold_pos = max(0, min(width - 1, hold_pos))

    bar = ["█"] * filled + ["░"] * (width - filled)
    # red peak marker
    bar[hold_pos] = "[red]▌[/red]"
    return "".join(bar)


def build_panel(rms: float, peak: float, bpm: Optional[float], beat_in_bar: Optional[int],
                rms_hold: float, peak_hold: float, noise_floor: float, thr: float) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", width=12)
    tbl.add_column()

    tbl.add_row("RMS",  make_meter(rms, rms_hold, METER_WIDTH) + f"  {rms:0.4f}")
    tbl.add_row("PEAK", make_meter(peak, peak_hold, METER_WIDTH) + f"  {peak:0.4f}")

    bpm_txt = f"{bpm:0.2f}" if bpm is not None else "--"
    beat_txt = f"{beat_in_bar}" if beat_in_bar is not None else "--"
    tbl.add_row("BPM",  f"[bold]{bpm_txt}[/bold]    Beat(1-4): [bold]{beat_txt}[/bold]")

    # Debug-ish but useful (can hide later)
    tbl.add_row("NoiseFloor", f"{noise_floor:0.6f}")
    tbl.add_row("Threshold",  f"{thr:0.6f}")

    return Panel(tbl, title="Beatbridge PoC (Kick Onset + BPM PLL)", border_style="cyan")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    console = Console()

    kick = KickOnsetDetector.create(SAMPLE_RATE, KICK_LP_HZ, ENV_LP_HZ)
    pll = BeatPLL()

    last_onset_t: Optional[float] = None

    # level stats (normalized in float32 domain)
    rms_val = 0.0
    peak_val = 0.0
    rms_hold = 0.0
    peak_hold = 0.0
    last_ui_t = time.time()

    # Convert to normalized [0..1] using a practical scale:
    # 0.0..0.25 is already very hot for RMS; 0.0..1.0 for peak.
    def norm_rms(x: float) -> float:
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

            # compute instantaneous RMS and PEAK for the block
            rms_inst = float(np.sqrt(np.mean(mono * mono)))
            peak_inst = float(np.max(np.abs(mono)))

            # smooth a little so UI is stable
            rms_val = 0.85 * rms_val + 0.15 * rms_inst
            peak_val = max(peak_val * 0.6, peak_inst)  # fast attack, some decay

            # peak-hold with time decay
            now = time.time()
            dt = max(1e-3, now - last_ui_t)
            last_ui_t = now

            rms_hold = max(rms_hold - PEAK_HOLD_DECAY_PER_S * dt, rms_val)
            peak_hold = max(peak_hold - PEAK_HOLD_DECAY_PER_S * dt, peak_val)

            # onset detection
            onsets = kick.process_block(mono, block_start_time=now)

            # bootstrap PLL from first two onsets
            for (t_evt, strength) in onsets:
                console.log(f"ONSET t={t_evt - t0:7.3f}s  strength={strength:0.4f}")
                if last_onset_t is None:
                    last_onset_t = t_evt
                elif pll.period_s is None:
                    bpm0 = pll.bootstrap_from_two_onsets(last_onset_t, t_evt)
                    last_onset_t = t_evt
                    if bpm0 is not None:
                        console.log(f"[green]BOOTSTRAP bpm≈{bpm0:0.2f}[/green]")
                else:
                    upd = pll.update_from_onset(t_evt)
                    if upd is not None:
                        bpm, locked = upd
                        if locked:
                            console.log(f"[green]LOCK bpm≈{bpm:0.2f}[/green]")

            # emit beat ticks (grid)
            beats = pll.generate_beats_up_to(now)
            for bt in beats:
                bpm = pll.stable_bpm()
                beat_pos = pll.beat_in_bar()
                if bpm is not None and beat_pos is not None:
                    console.log(f"BEAT t={bt - t0:7.3f}s  bpm≈{bpm:0.2f}  pos={beat_pos}")

            # show UI
            bpm = pll.stable_bpm()
            beat_pos = pll.beat_in_bar()

            # expose current adaptive threshold (for debugging)
            thr = max(THRESH_MIN, kick.noise_floor * THRESH_FACTOR)

            panel = build_panel(
                rms=norm_rms(rms_val),
                peak=norm_peak(peak_val),
                bpm=bpm,
                beat_in_bar=beat_pos,
                rms_hold=norm_rms(rms_hold),
                peak_hold=norm_peak(peak_hold),
                noise_floor=kick.noise_floor,
                thr=thr,
            )
            live.update(panel)

            time.sleep(0.005)


if __name__ == "__main__":
    main()
