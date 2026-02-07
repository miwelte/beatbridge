# RasPi BPM Hold → MIDI Daemon

> **Club‑tauglicher Ersatz für SBX‑1**
> Audio → BPM‑Analyse → *Hold last BPM* → interner Clock → MIDI Beat → e:cue LAS

Dieses Projekt beschreibt eine **robuste, wartbare und erweiterbare Tempo‑Engine** auf Raspberry Pi‑Basis.
Ziel ist ein **stabiler BPM‑Wert**, der **autonom weiterläuft**, auch wenn das Audiosignal schwankt oder aussetzt.

---

## 1. Ziel & Design‑Prinzipien

**Ziel**

- Stabiler BPM‑State für Club‑Umgebungen
- Audio dient *nur* zur Korrektur, niemals als Taktgeber
- Manueller Override per **Tap‑Button**
- Saubere MIDI‑Events für e:cue LAS
- Redundanzfähig gedacht (Hot‑Standby möglich)

**Leitprinzip**

> **Clock schlägt alles. Audio berät. Tap entscheidet. LAS reagiert.**

---

## 2. Gesamtarchitektur

```
DJ Mixer (REC / BOOTH OUT)
        │  Analog Audio (Line)
        ▼
USB Audio Interface
        │
        ▼
Raspberry Pi
┌───────────────────────────────────────┐
│ BPM Daemon                            │
│                                       │
│  Audio Input Thread                   │
│  → BPM Analysis (aubio)               │
│  → Confidence Gate + Filter           │
│                                       │
│  Tap Input (GPIO / MIDI)              │
│  → BPM Override                       │
│                                       │
│  Internal BPM State (HOLD)            │
│                                       │
│  Clock Generator (monotonic)          │
│                                       │
│  MIDI Output Thread                   │
│  → MIDI Note per Beat                 │
└───────────────────────────────────────┘
        │
        ▼
USB MIDI Interface (DIN)
        │
        ▼
e:cue LAS
```

---

## 3. BPM‑Daemon‑Architektur

### 3.1 Thread‑Modell

Ein **Prozess**, klar getrennte Threads:

```
bpm-daemon
│
├─ Audio Input Thread        (ALSA → Ringbuffer)
├─ BPM Analysis Thread       (aubio + Filter)
├─ Tap Input Thread          (GPIO / MIDI)
├─ Clock Generator Thread    (High‑Res Timer)
└─ MIDI Output Thread        (rtmidi)
```

**Wichtig:**
Kein Thread darf den Clock‑Thread blockieren.

---

### 3.2 BPM‑State‑Maschine

```
STATE: BPM_VALID        (Audio liefert valide BPM)
STATE: BPM_HOLD         (Audio unzuverlässig)
STATE: BPM_TAP_OVERRIDE (manueller Tap)

Priorität:
Tap  >  Audio  >  Hold
```

---

### 3.3 Audio → BPM Analyse

- Audio via ALSA (kleine Frames, z. B. 256–512 Samples)
- aubio liefert:
  - `bpm_candidate`
  - `confidence`

**Confidence Gate**

```
if confidence >= 0.75:
    BPM akzeptieren
else:
    ignorieren
```

---

### 3.4 BPM‑Filterung

Um BPM‑Sprünge zu vermeiden:

1. Median‑Filter (letzte n Werte)
2. Exponential Moving Average (EMA)

```
bpm_filtered = α * bpm_new + (1‑α) * bpm_old
```

- α ≈ 0.15
- Schnelle Tempoänderungen erfolgen **nur über Tap**

---

### 3.5 Clock‑Generator

- Nutzt **monotonic high‑resolution timer**
- Vollständig entkoppelt vom Audio

```
interval = 60 / BPM
next_tick += interval
sleep_until(next_tick)
emit_beat()
```

➡ Wenn Audio & Analyse ausfallen, **läuft der Clock weiter**.

---

### 3.6 MIDI‑Output

- **MIDI Note pro Beat** (empfohlen)
- Optional zweite Note für Downbeat

Warum keine MIDI‑Clock:

- weniger Jitter
- LAS verarbeitet Events robuster
- leichter zu debuggen

---

## 4. Tap‑Button‑Integration (GPIO)

### 4.1 Funktion

- Manueller BPM‑Override
- Sofortige Wirkung
- Audio darf danach nur langsam zurückziehen

### 4.2 Logik

- Zeitstempel mehrerer Taps sammeln
- Median der letzten Intervalle
- Neuer BPM = `60 / interval`

```
Tap > Audio > Hold
```

---

### 4.3 Elektrische Architektur (Variante A)

```
GPIO17 ────────┐
               │
           [ Taster ]
               │
GND    ────────┘
```

- interner Pull‑Up aktiv
- Logik: LOW = Tap
- Leitung > 1 m → **geschirmtes Kabel**, Schirm einseitig auf GND (Pi‑Seite)

---

## 5. Hardware‑Referenz (empfohlen)

| Funktion | Komponente                      |
| -------- | ------------------------------- |
| SBC      | Raspberry Pi 4 (4 GB)         |
| Audio    | Focusrite Scarlett 2i2         |
| MIDI     | iConnectivity mioXC             |
| Tap      | Industrietaster (EAO Serie 19) |
| GPIO     | Screw‑Terminal GPIO HAT        |
| PSU      | Industrie‑Netzteil 5 V        |

---

## 6. systemd‑Integration

Der BPM‑Daemon läuft als **systemd‑Service**:

- Autostart beim Boot
- Restart bei Fehlern
- Optional systemd‑Watchdog
- Gehärtete Service‑Sandbox

Ziel: *kein manuelles Starten, kein UI, kein Eingreifen im Betrieb*.

---

## 7. e:cue LAS – Blueprint

### 7.1 Grundannahmen

- Audio‑Beat in LAS: **aus**
- Eine globale Zeitquelle
- MIDI ist die einzige Beat‑Referenz

### 7.2 MIDI‑Events

| MIDI     | LAS Event              |
| -------- | ---------------------- |
| Note C3  | EV_BEAT                |
| Note C#3 | EV_DOWNBEAT (optional) |

### 7.3 Event‑Logik

**EV_BEAT**

- FX Resync
- Step Advance (optional)

**EV_DOWNBEAT**

- Cuelist Reset
- Accent‑Trigger

### 7.4 Cuelist‑Design

**Empfohlen:** FX‑getriebene Cuelists

- FX referenzieren **Speed Master 1**
- Keine festen Zeiten
- Beat dient nur zur Synchronisation

---

## 8. Redundanz‑Vorbereitung (optional)

Die Architektur ist von Anfang an **redundanzfähig**:

```
        Audio Split
          │     │
         Pi A  Pi B
      (Primary)(Standby)
          │     │
          └── MIDI Merger ──► LAS
```

- Nur ein Pi sendet aktiv MIDI
- Standby übernimmt bei Ausfall
- BPM‑State kann per Netzwerk synchronisiert werden

Die Single‑Pi‑Implementierung bleibt dabei unverändert.

---

## 9. Betrieb & Wartung

- Headless Betrieb (SSH only)
- SD‑Card‑Image sichern
- Austauschbare Standard‑Hardware
- Keine proprietären Abhängigkeiten

**Worst‑Case‑Verhalten**

- Audio weg → BPM Hold
- BPM‑Daemon Restart → Clock weiter
- MIDI weg → Licht friert ein, driftet nicht

---

## 10. Zusammenfassung

Dieses Projekt ersetzt klassische Sync‑Boxen durch eine **offene, kontrollierbare und zukunftssichere Architektur**:

- stabiler BPM‑State
- saubere Trennung von Audio & Clock
- manuelle Kontrolle via Tap
- perfekte Integration in e:cue LAS

> **Tempo ist Infrastruktur. Wenn sie steht, wird alles darüber einfach.**

---

_Ende der Dokumentation._
