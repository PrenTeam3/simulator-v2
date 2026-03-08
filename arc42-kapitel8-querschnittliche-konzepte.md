# arc42 Kapitel 8: Querschnittliche Konzepte - Puzzle Solver V2

## Versionsinformation
- **Dokumentversion:** 1.0
- **Datum:** 2025-12-05
- **Status:** Final

---

## Inhaltsverzeichnis

1. [8.1 Koordinatensysteme](#81-koordinatensysteme)
2. [8.2 Bewertungssystem](#82-bewertungssystem)
3. [8.3 Adaptive Algorithmen](#83-adaptive-algorithmen)
4. [8.4 Datenstrukturen](#84-datenstrukturen)
5. [8.5 Threading-Modell](#85-threading-modell)
6. [8.6 Fehlerbehandlung](#86-fehlerbehandlung)

---

## 8.1 Koordinatensysteme

### Bildraum (Analyse)
- Ursprung: Oben-Links (0, 0)
- Y-Achse: ↓ (unten positiv)
- Verwendung: OpenCV, Ecken-Erkennung

### Zusammenbau-Raum (Lösung)
- Ursprung: Zentrum (0, 0)
- Y-Achse: ↑ (oben positiv)
- Verwendung: Teil-Platzierung, Rotation

### Transformationen
```python
# Verschiebung
verschoben = punkte + vektor

# Rotation um Punkt
rotiert = rotiere_um_punkt(punkte, drehpunkt, winkel)
```

**Quellcode:** `geometry_utils.py`, `assembler.py`

---

## 8.2 Bewertungssystem

### Kombinierter Score
```
kombiniert = 0,3 × länge + 0,5 × form + 0,2 × richtung
```

### Komponenten

| Metrik | Berechnung | Gewicht |
|--------|------------|---------|
| **Länge** | `100 - (längen_differenz / durchschnitt) × 100` | 30% |
| **Form** | `100 - (RMSD / durchschnitt) × 100` | 50% |
| **Richtung** | `100 - abs(180° - winkel)` | 20% |

### Schwellenwerte
- Länge: ≥ 80
- Form: ≥ 80
- Richtung: ≥ 60
- Kombiniert: ≥ 75

**Quellcode:** `similarity_calculator.py`, `chain_matcher.py`

---

## 8.3 Adaptive Algorithmen

### Adaptive Striktheit (Rahmen-Ecken-Erkennung)

**Striktheitsstufen:**

| Stufe | winkel_toleranz | max_abweichung | min_geradheit |
|-------|----------------|---------------|------------------|
| streng | 12° | 1.0 | 0.90 |
| ausgewogen | 15° | 2.0 | 0.85 |
| locker | 18° | 3.0 | 0.80 |

**Logik:**
```
gefunden < ziel → reduziere Striktheit (streng → ausgewogen → locker)
gefunden > ziel → erhöhe Striktheit (locker → ausgewogen → streng)
gefunden = ziel → fertig ✓
```

Maximal 5 Iterationen, dann Best-Effort.

### Greedy Verbindungsauswahl

**Einschränkungen:**
- Maximal 2 Verbindungen pro Teil
- Maximal 1 Verbindung pro Teil-Paar
- Keine überlappenden Segmente

**Ranking:** `0,8 × chain_länge + 0,2 × qualität`

**Quellcode:** `corner_detector.py`, `connection_selector.py`

---

## 8.4 Datenstrukturen

### Kern-Klassen

```python
@dataclass
class SegmentMatch:
    teil1_id: int
    teil2_id: int
    seg1_id: int
    seg2_id: int
    längen_score: float
    form_score: float
    richtungs_score: float

@dataclass
class ChainMatch:
    teil1_id: int
    teil2_id: int
    segment_ids_t1: List[int]
    segment_ids_t2: List[int]
    chain_länge: int
    scores: Dict[str, float]
    blauer_punkt_t1: np.ndarray  # Rahmen-Verbindung
    roter_punkt_t1: np.ndarray   # Innenrichtung
    blauer_punkt_t2: np.ndarray
    roter_punkt_t2: np.ndarray

@dataclass
class Transformation:
    verschiebung: np.ndarray
    rotation: float
    drehpunkt: np.ndarray
```

**Quellcode:** `chain_matcher.py`, `models.py`

---

## 8.5 Threading-Modell

### GUI-Threading (Qt)

```python
class PipelineWorker(QThread):
    schritt_beendet = pyqtSignal(str)
    fortschritt_aktualisiert = pyqtSignal(int)
    pipeline_beendet = pyqtSignal(object)

    def run(self):
        # Phase 1: Analyse (25%)
        ergebnis = führe_analyse_aus(bild_pfad)
        self.schritt_beendet.emit("Analyse")

        # Phase 2: Lösung (50%)
        ergebnis = löse_puzzle(temp_ordner)
        self.schritt_beendet.emit("Löser")

        # Phase 3: Zusammenbau (25%)
        ergebnis = baue_puzzle_zusammen(ergebnis)
        self.pipeline_beendet.emit(ergebnis)
```

**Kommunikation:** Qt Signals/Slots (thread-sicher)

**Quellcode:** `simulator/main.py`

---

## 8.6 Fehlerbehandlung

### Strategie

**Ebene 1: Validierung**
```python
if not os.path.exists(bild_pfad):
    raise FileNotFoundError(f"Bild nicht gefunden: {bild_pfad}")
```

**Ebene 2: Sanfte Degradierung**
```python
try:
    ecken = erkenne_adaptiv(teil)
except Exception:
    ecken = erkenne_fallback(teil)  # Verwende Standard-Striktheit
```

**Ebene 3: Teilweise Ergebnisse**
```python
# Setze Zusammenbau fort, auch wenn einige Teile fehlschlagen
for teil in teile:
    try:
        platziere_teil(teil)
    except PlatzierungsFehler:
        fehlgeschlagene_teile.append(teil)
        continue  # Fahre mit anderen fort
```

### Robustheitsmuster

| Muster | Anwendung |
|---------|-----------|
| **Schwellenwert-Fallback** | Gelockerte Schwellenwerte bei wenigen Matches |
| **Iterative Verfeinerung** | Maximal 5 Iterationen für Rahmen-Ecken-Erkennung |
| **Plausibilitätsprüfungen** | Validiere Zusammenbau (Überlappungen, Zusammenhang) |

**Quellcode:** Alle Module

---

## Zusammenfassung

| Konzept | Kernpunkte | Referenz |
|---------|------------|----------|
| **Koordinaten** | 2 Systeme (Bild/Zusammenbau), Transformationen | `geometry_utils.py` |
| **Bewertung** | 3 Metriken, Kombiniert 30/50/20, Schwellenwerte | `similarity_calculator.py` |
| **Adaptiv** | 5 Striktheitsstufen, Greedy-Auswahl | `corner_detector.py` |
| **Daten** | SegmentMatch, ChainMatch, Transformation | `chain_matcher.py` |
| **Threading** | Qt Worker, Signals, 3 Phasen | `simulator/main.py` |
| **Fehler** | 3-Ebenen-Strategie, Teilweise Ergebnisse | Alle Module |

---