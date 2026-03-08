# arc42 Kapitel 3: Kontextabgrenzung - Puzzle Solver V2

## Versionsinformation
- **Dokumentversion:** 1.0
- **Datum:** 2025-12-05
- **Status:** Final

---

## Inhaltsverzeichnis

1. [3.1 Fachlicher Kontext](#31-fachlicher-kontext)
2. [3.2 Technischer Kontext](#32-technischer-kontext)
3. [3.3 Externe Schnittstellen](#33-externe-schnittstellen)

---

## 3.1 Fachlicher Kontext

### Systemübersicht

```
┌─────────────┐
│   Benutzer  │
└──────┬──────┘
       │ Bild (PNG/JPG)
       │ Konfiguration
       ↓
┌─────────────────────────────────────┐
│                                     │
│     Puzzle Solver V2 Simulator      │
│                                     │
│  ┌────────────┐  ┌──────────────┐  │
│  │  Analyse   │→ │ Edge Solver  │  │
│  └────────────┘  └──────┬───────┘  │
│                         ↓           │
│                  ┌─────────────┐   │
│                  │  Assembler  │   │
│                  └──────┬──────┘   │
└─────────────────────────┼──────────┘
                          │
                          ↓
              ┌──────────────────────┐
              │ Ausgabe-Dateien      │
              │ - Visualisierungen   │
              │ - JSON Ergebnisse    │
              │ - Logs               │
              └──────────────────────┘
```

---

### Fachliche Nachbarsysteme

| System/Akteur | Eingabe → System | System → Ausgabe | Beschreibung |
|---------------|------------------|------------------|--------------|
| **Benutzer** | Puzzle-Bild (PNG/JPG)<br>Auswahl: Teil-Anzahl | Visualisierungen<br>Zusammenbau-Ergebnis<br>Fehler-Meldungen | Interaktiver Benutzer über GUI |
| **Dateisystem** | Bild-Dateien<br>Temp-Verzeichnisse | Extrahierte Teile<br>Debug-Bilder<br>JSON-Ergebnisse<br>Logs | Persistierung von Ein-/Ausgaben |
| **Bildaufnahme-System** | Kalibiertes Foto<br>Farbige Rahmen-Markierungen | - | Externes System zur Bild-Erfassung (nicht Teil des Solvers) |

---

### Fachliche Schnittstellen im Detail

#### E1: Benutzer → Simulator

**Kanal:** Qt GUI (PyQt6)

**Daten:**
- Bild-Pfad (absolut)
- Erwartete Teil-Anzahl (Integer)
- Optional: Ausgabe-Verzeichnis

**Formate:**
- Bild: PNG, JPG (RGB, 8-bit pro Kanal)
- Auflösung: 800×600 bis 4096×3072 Pixel empfohlen

**Randbedingungen:**
- Bild muss existieren und lesbar sein
- Unterstützte Teile: 2-16 (optimiert für 4)
- Hintergrund sollte kontrastreich sein

---

#### A1: Simulator → Benutzer

**Kanal:** Qt GUI (PyQt6)

**Daten:**
- Fortschritt (0-100%)
- Schritt-Status (Analyse/Löser/Zusammenbau)
- Visualisierungen (in-memory Bilder)
- Fehler-Meldungen

**Feedback-Arten:**
- ✅ Erfolgreich: Zusammenbau-Visualisierung
- ⚠️ Warnung: Teilweise Lösung, fehlende Verbindungen
- ❌ Fehler: Bild nicht gefunden, zu wenig Teile erkannt

---

#### E2/A2: Simulator ↔ Dateisystem

**Eingabe (E2):**
- Puzzle-Bild: `puzzle.png`
- Konfiguration: `config.json` (optional)

**Ausgabe (A2):**
- Extrahierte Teile: `temp/piece_0.png`, `piece_1.png`, ...
- Segment-Daten: `segment_pairs_*.json`
- Progressive Chains: `progressive_chain_*.json`
- Visualisierungen:
  - `analysis_overview.png`
  - `frame_corners_combined.png`
  - `assembly_steps_combined.png`

**Verzeichnis-Struktur:**
```
output/
├── temp/
│   ├── piece_0.png
│   ├── piece_1.png
│   └── ...
├── segment_pairs_*.json
├── progressive_chain_*.json
└── visualizations/
    ├── analysis_overview.png
    ├── frame_corners_combined.png
    └── assembly_steps_combined.png
```

---

## 3.2 Technischer Kontext

### Technologie-Stack

```
┌─────────────────────────────────────────┐
│           Präsentationsschicht          │
│  ┌─────────────────────────────────┐   │
│  │  PyQt6 GUI (Qt6 Framework)      │   │
│  │  - QMainWindow                   │   │
│  │  - QThread (Worker)              │   │
│  │  - Signals/Slots                 │   │
│  └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │ Qt Signals
┌─────────────────┴───────────────────────┐
│         Geschäftslogik-Schicht          │
│  ┌─────────────────────────────────┐   │
│  │  Python 3.10+ Module            │   │
│  │  - analyze.py                    │   │
│  │  - edge_solver_v2.py             │   │
│  │  - assembler.py                  │   │
│  │  - chain_matcher.py              │   │
│  └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │ NumPy Arrays
┌─────────────────┴───────────────────────┐
│          Computer Vision Layer          │
│  ┌─────────────────────────────────┐   │
│  │  OpenCV 4.x                      │   │
│  │  - cv2.findContours()            │   │
│  │  - cv2.approxPolyDP()            │   │
│  │  - cv2.goodFeaturesToTrack()     │   │
│  └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │ Pixel-Daten
┌─────────────────┴───────────────────────┐
│        Datenhaltungs-Schicht            │
│  ┌─────────────────────────────────┐   │
│  │  Dateisystem (OS)                │   │
│  │  - PNG/JPG (PIL/cv2)             │   │
│  │  - JSON (Python json)            │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

### Technische Schnittstellen

#### T1: GUI ↔ Pipeline Worker

**Technologie:** Qt Signals/Slots (thread-sicher)

**Protokoll:**
```python
# GUI → Worker
start_pipeline(image_path: str, num_pieces: int)
stop_pipeline()

# Worker → GUI
fortschritt_aktualisiert: pyqtSignal(int)           # 0-100
schritt_beendet: pyqtSignal(str)                    # "Analyse", "Löser", "Zusammenbau"
pipeline_beendet: pyqtSignal(object)                # AssemblyResult
fehler_aufgetreten: pyqtSignal(str)                 # Error message
```

**Datenformat:**
```python
@dataclass
class AssemblyResult:
    pieces: List[PieceData]
    connections: List[Connection]
    transformations: Dict[int, Transformation]
    visualization: np.ndarray  # RGB Image
```

---

#### T2: Python ↔ OpenCV

**Technologie:** OpenCV Python Bindings (cv2)

**Datenformat:**
- Bilder: NumPy Arrays `(H, W, 3)` uint8 (BGR)
- Konturen: `List[np.ndarray]` mit `(N, 1, 2)` int32
- Punkte: `np.ndarray` mit `(N, 2)` float32

**Wichtige Operationen:**
```python
# Bild einlesen
image = cv2.imread(path)  # BGR format

# Kontur-Erkennung
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Polygon-Approximation
approx = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)

# Ecken-Erkennung
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
```

---

#### T3: Python ↔ Dateisystem

**Technologie:**
- Standard Library: `os`, `pathlib`, `json`
- Bilder: `cv2.imwrite()`, `PIL.Image`

**Dateiformate:**

| Typ | Format | Kodierung | Verwendung |
|-----|--------|-----------|------------|
| **Eingabe-Bild** | PNG/JPG | RGB, 8-bit | Original Puzzle-Foto |
| **Teil-Bilder** | PNG | RGBA, 8-bit | Extrahierte Teile (Alpha für Maske) |
| **Segment-Daten** | JSON | UTF-8 | SegmentMatch-Listen |
| **Chain-Daten** | JSON | UTF-8 | ChainMatch-Listen |
| **Visualisierungen** | PNG | RGB, 8-bit | Debug/Ausgabe-Bilder |

**JSON-Schema (Beispiel):**
```json
{
  "segment_pairs": [
    {
      "piece1_id": 0,
      "piece2_id": 1,
      "segment1_id": 2,
      "segment2_id": 0,
      "length_score": 92.5,
      "shape_score": 85.3,
      "direction_score": 100.0
    }
  ]
}
```

---

## 3.3 Externe Schnittstellen

### Abhängigkeiten

| Bibliothek | Version | Lizenz | Verwendung |
|------------|---------|--------|------------|
| **Python** | ≥3.10 | PSF | Runtime |
| **PyQt6** | ≥6.4 | GPL/Commercial | GUI Framework |
| **OpenCV** | ≥4.5 | Apache 2.0 | Computer Vision |
| **NumPy** | ≥1.23 | BSD | Numerische Operationen |
| **SciPy** | ≥1.9 | BSD | Spatial Transformationen |
| **Pillow** | ≥9.0 | HPND | Bild-I/O |

---

### Deployment-Kontext

```
┌─────────────────────────────────────────┐
│      Entwickler-Workstation             │
│  ┌─────────────────────────────────┐   │
│  │  Python 3.10+ Virtual Env       │   │
│  │  - pip install -r requirements  │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Puzzle Solver V2               │   │
│  │  - Source Code (Python)         │   │
│  │  - Konfiguration                │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Betriebssystem                 │   │
│  │  - Windows 10/11                │   │
│  │  - Linux (Ubuntu 20.04+)        │   │
│  │  - macOS (12+)                  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Hardware-Anforderungen:**
- **CPU:** Dual-Core, 2+ GHz (Quad-Core empfohlen)
- **RAM:** 4 GB (8 GB empfohlen)
- **Speicher:** 500 MB (für Dependencies + temporäre Dateien)
- **Display:** 1280×720 Pixel minimal

---

### Externe Daten-Quellen

#### Bildaufnahme-System (optional, extern)

**Beschreibung:**
- Kamera-Setup zur Puzzle-Erfassung
- Nicht Teil des Puzzle Solvers
- Liefert kalibrierte Bilder

**Anforderungen an Bilder:**
- **Auflösung:** ≥1920×1080 empfohlen
- **Beleuchtung:** Gleichmäßig, diffus
- **Hintergrund:** Einfarbig, kontrastreich zu Teilen
- **Perspektive:** Orthogonal (von oben)
- **Farbige Rahmen:** Gut sichtbar für Analyse

**Schnittstelle:**
- Dateibasiert (keine direkte Integration)
- Übertragung via Dateisystem oder Netzwerk

---

## Abgrenzung: Was ist NICHT Teil des Systems?

### Außerhalb des Scopes

❌ **Bildaufnahme:**
- Keine Kamera-Steuerung
- Keine Bild-Kalibrierung
- Keine Echtzeit-Video-Verarbeitung

❌ **Robotik-Integration:**
- Keine Steuerung von physischen Robotern
- Keine Bewegungsplanung
- Keine Greifer-Ansteuerung

❌ **Online-Dienste:**
- Keine Cloud-Verarbeitung
- Keine Web-API
- Keine Mehrbenutzerfähigkeit

❌ **Puzzle-Generierung:**
- Keine Erstellung von Puzzle-Vorlagen
- Keine Schnittmuster-Generierung

❌ **3D-Puzzles:**
- Nur 2D Jigsaw Puzzles
- Keine räumlichen Puzzle (Würfel, etc.)

---

## Zusammenfassung

### Kontext-Diagramm (kompakt)

```
        [Benutzer]
             ↕ Bild, Konfiguration / Visualisierung
    ┌────────────────────┐
    │  Puzzle Solver V2  │
    └────────────────────┘
             ↕ PNG/JPG, JSON
       [Dateisystem]
```

### Wichtigste Schnittstellen

| ID | Schnittstelle | Technologie | Kritikalität |
|----|---------------|-------------|--------------|
| **E1** | Benutzer → System | Qt GUI | 🔴 HOCH |
| **A1** | System → Benutzer | Qt GUI | 🔴 HOCH |
| **T1** | GUI ↔ Worker | Qt Signals | 🔴 HOCH |
| **T2** | Python ↔ OpenCV | cv2 API | 🔴 HOCH |
| **T3** | Python ↔ Dateisystem | Standard I/O | ⚠️ MITTEL |

### Externe Abhängigkeiten

- **Kritisch:** Python 3.10+, PyQt6, OpenCV
- **Standard:** NumPy, SciPy, Pillow
- **Optional:** Keine

---