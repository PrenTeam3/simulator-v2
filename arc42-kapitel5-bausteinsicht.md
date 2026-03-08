# arc42 Kapitel 5: Bausteinsicht - Puzzle Solver V2

## Versionsinformation
- **Dokumentversion:** 1.0
- **Datum:** 2025-12-11
- **Status:** Final

---

## Inhaltsverzeichnis

1. [5.1 Ebene 1: Übersicht über die Bausteine](#51-ebene-1-übersicht-über-die-bausteine)
2. [5.2 Ebene 2: Detaillierte Bausteine - solver-v2](#52-ebene-2-detaillierte-bausteine---solver-v2)
3. [5.2.1 simulator-gui: Detaillierte Bausteine](#521-simulator-gui-detaillierte-bausteine)
4. [5.3 Bausteine im Detail](#53-bausteine-im-detail)

---

## 5.1 Ebene 1: Übersicht über die Bausteine

### Übersicht

Die Bausteinsicht zeigt die Hauptkomponente **Simulator** mit ihren drei Untermodulen: **simulator-gui**, **puzzleSolver** (Legacy) und **solver-v2** (Hauptkomponente). Der Simulator orchestriert die komplette Puzzle-Lösungspipeline von der Bildanalyse bis zum virtuellen Zusammenbau.

### Level 1 Diagramm

![Bausteinsicht Ebene 1](diagrams/05_01_bausteinsicht_ebene1.puml)

### Hauptkomponente: Simulator

| Baustein | Verantwortlichkeit | Untermodule |
|----------|-------------------|-------------|
| **Simulator** | Orchestriert die Puzzle-Lösungspipeline, Benutzerinteraktion, Pipeline-Ausführung | simulator-gui, puzzleSolver, solver-v2 |

### Untermodule des Simulators

| Modul | Verantwortlichkeit | Verwendung |
|-------|-------------------|------------|
| **simulator-gui** | GUI für Visualisierung, Pipeline-Orchestrierung über PipelineWorker | Primär, wird von PuzzleSimulatorWindow genutzt |
| **puzzleSolver** | Legacy Puzzle Solver (alternative Implementierung) | Alternativ, kann über GUI ausgewählt werden |
| **solver-v2** | Hauptkomponente für Analyse, Matching, Assembly (drei Phasen) | Primär, standardmäßig verwendet |

### Schnittstellen zwischen den Modulen

#### S1: simulator-gui → solver-v2
- **Phase 1 - Analyse:** `analyze.run_analysis()` wird aufgerufen
  - **Datenfluss:** Bildpfad (string), Analyse-Parameter
  - **Rückgabe:** Temp-Folder-Name (string), Analyse-Daten (via Dateisystem)
- **Phase 2+3 - Lösen & Assembly:** `solver.solve_puzzle()` wird aufgerufen
  - **Datenfluss:** Temp-Folder-Name (string), Solver-Algorithmus ('edge_v2')
  - **Rückgabe:** Solver-Ergebnisse (via Dateisystem und Rückgabe-Objekte)

#### S2: solver-v2 interne Schnittstellen
- **Analyzer → Solver:** Analyse-Daten (JSON, SVG, PNG) via Dateisystem
- **Format:** Strukturierte Daten in `temp/analysis_TIMESTAMP/` Verzeichnis

### Abhängigkeiten

- **simulator-gui** hängt von **solver-v2** ab (importiert `analyze` und `puzzle_solver.solver`)
- **solver-v2** ist in sich abgeschlossen und nutzt Dateisystem für Datenübergabe zwischen Phasen

---

## 5.2 Ebene 2: Detaillierte Bausteine - solver-v2

### Übersicht

Ebene 2 zeigt die detaillierte Struktur der drei Hauptkomponenten von **solver-v2**: **Analyzer**, **Solver** und **Assembly** mit ihren Untermodulen. Der Fokus liegt auf der Pipeline-Abfolge entsprechend der Laufzeitsicht (Analyse → Matching → Assembly).

### Level 2 Diagramm

![Bausteinsicht Ebene 2](diagrams/05_02_bausteinsicht_ebene2.puml)

### Pipeline-Ablauf (entsprechend Laufzeitsicht Szenario 1)

Die Pipeline folgt diesem Ablauf:

1. **Phase 1 - Analyzer** (Schritte 1-12 in Laufzeitsicht):
   - `PuzzleAnalyzer` erkennt Teile, ermittelt Ecken, erzeugt geglättete Konturen
   - Temp-Folder und Analysis-Data werden zurückgegeben

2. **Phase 2 - Solver** (Schritte 13-15 in Laufzeitsicht):
   - `PuzzlePreparer` bereitet Daten vor
   - `EdgeSolver` berechnet passende Kantenverbindungen

3. **Phase 3 - Assembly** (Schritte 16-21 in Laufzeitsicht):
   - `AssemblySolver` platziert Puzzleteile virtuell
   - Resultate werden zurückgegeben

### solver-v2: Analyzer (Phase 1) - Untermodule

| Baustein | Verantwortlichkeit |
|----------|-------------------|
| **PuzzleAnalyzer (Core)** | Orchestrierung der Analyse-Phase |
| **CornerDetector** | Ecken-Erkennung, Frame-Corner-Identifikation, Adaptive Strictness |
| **SVGVisualizer** | SVG-Generierung aus Konturen |
| **SVGSmoother** | Kontur-Glättung für bessere Ecken-Erkennung |
| **SVGCornerDrawer** | Ecken-Markierung in SVG |
| **CornerVisualizer** | Visualisierung der erkannten Ecken |

**Verantwortlichkeiten:**
- Puzzle-Teile-Erkennung (Contour Detection)
- Kontur-Extraktion und Glättung
- Ecken-Erkennung und Klassifikation (Frame/Inner)
- SVG-Generierung für präzise geometrische Analyse
- JSON/PNG-Export für nachfolgende Phasen

### solver-v2: Solver (Phase 2) - Untermodule

Der Solver besteht aus zwei Teilen:

#### Preparation

| Baustein | Verantwortlichkeit |
|----------|-------------------|
| **PuzzlePreparer** | Orchestrierung der Vorbereitung |
| **PuzzleDataLoader** | Laden von Analyse-Daten aus JSON/SVG |
| **SVGDataLoader** | Parsen von SVG-Dateien |
| **ContourSegmenter** | Segmentierung von Konturen in Segmente |
| **SolverVisualizer** | Initiale Visualisierung |

#### Edge Solver V2

**Edge Solver V2** (primär verwendeter Algorithmus, entspricht "Matching-Komponente" in Laufzeitsicht):

| Baustein | Verantwortlichkeit |
|----------|-------------------|
| **EdgeSolver** | Orchestrierung des Edge-Solving-Prozesses |
| **SegmentFinder** | Identifikation von Frame-adjacent Segmenten |
| **ChainMatcher** | Erweiterung zu Progressive Chains, Chain-Scoring |
| **ConnectionSelector** | Auswahl der besten Verbindungen (Greedy) |
| **GeometryUtils** | Geometrische Transformationen (ICP, RMSD) |
| **EdgeSolverV2Visualizer** | Segment-Paar-Vergleich, Score-Berechnung, Visualisierung |
| **ConnectionVisualizer** | Visualisierung von Connections |
| **ChainVisualizer** | Visualisierung von Chains |
| **SegmentVisualizer** | Visualisierung von Segmenten |

### solver-v2: Assembly (Phase 3) - Untermodule

**Assembly** (entspricht "Assembly" in Laufzeitsicht):

| Baustein | Verantwortlichkeit |
|----------|-------------------|
| **AssemblySolver** | Orchestrierung des Zusammenbaus |
| **AssemblyVisualizer** | Visualisierung der Assembly-Schritte |

**Verantwortlichkeiten:**
- Transformation der Pieces (Translation, Rotation)
- Anker-Auswahl (bestes Piece als Referenz)
- Iterative Platzierung basierend auf Chains und Blue/Red Dots
- 8-Schritt-Visualisierung des Zusammenbau-Prozesses

---

## 5.2.1 simulator-gui: Detaillierte Bausteine

### Übersicht

simulator-gui ist die Benutzeroberfläche, die die Pipeline orchestriert und die Ergebnisse visualisiert.

### simulator-gui Diagramm

![Bausteinsicht simulator-gui](diagrams/05_03_bausteinsicht_simulator_gui.puml)

### simulator-gui - Untermodule

| Baustein | Verantwortlichkeit |
|----------|-------------------|
| **PuzzleSimulatorWindow** | GUI-Hauptfenster, Steuerung, Visualisierung |
| **PipelineWorker** | Thread-basierte Ausführung der Pipeline, Signal-Kommunikation |

**Verantwortlichkeiten:**
- Start/Stop der Pipeline
- Fortschrittsanzeige (0-100%)
- Bild-Visualisierung (Zwischenschritte und Endergebnis)
- Fehlerbehandlung und Logging
- Ruft `analyze.run_analysis()` und `solver.solve_puzzle()` auf

**Schnittstellen zu solver-v2:**
- **Phase 1:** Ruft `analyze.run_analysis()` auf → schreibt Analyse-Daten ins Dateisystem
- **Phase 2+3:** Ruft `solver.solve_puzzle()` auf → liest Analyse-Daten, schreibt Ergebnisse

---

## 5.3 Bausteine im Detail

### 5.3.1 simulator-gui

**Verantwortlichkeit:**
Die simulator-gui stellt die Benutzeroberfläche bereit und orchestriert die gesamte Pipeline über den PipelineWorker. Sie kapselt die komplexe Logik der Analyse und Lösung in einer benutzerfreundlichen Oberfläche.

**Schnittstellen:**
- **Eingabe:** Bild-Pfad (Datei-Auswahl), Solver-Algorithmus (Dropdown)
- **Ausgabe:** Fortschrittsanzeige, Visualisierungen, Logs, Fehlermeldungen

**Nutzung von solver-v2:**
- `PipelineWorker.run()` ruft `analyze.run_analysis()` auf (Phase 1)
- `PipelineWorker.run()` ruft `solver.solve_puzzle()` auf (Phase 2+3)
- Nutzt Dateisystem für Ergebnis-Export und Visualisierung

---

### 5.3.2 solver-v2: Analyzer (Phase 1)

**Verantwortlichkeit:**
Der Puzzle Analyzer erkennt Puzzleteile in einem Eingabebild und analysiert ihre geometrischen Eigenschaften. Er extrahiert Konturen, glättet sie, erkennt Ecken und klassifiziert sie in Frame- und Inner-Ecken.

**Kernfunktionen:**
1. **Piece Detection:** Findet alle Puzzleteile im Bild mittels Contour Detection
2. **Contour Smoothing:** Glättet Konturen für präzisere Analyse
3. **Corner Detection:** Erkennt Ecken mit adaptiver Strictness-Anpassung
4. **Frame Corner Identification:** Identifiziert Rahmen-Ecken (90°-Ecken am Puzzle-Rand)
5. **SVG Export:** Generiert SVG-Dateien für präzise geometrische Analyse

**Schnittstellen:**
- **Eingabe:** Bild-Pfad (string), Debug-Flag (bool), Target Frame Corners (int)
- **Ausgabe:** Temp-Folder-Name (string), Analyse-Daten (JSON, SVG, PNG im Dateisystem)

**Abhängigkeiten:**
- OpenCV für Bildverarbeitung
- NumPy für numerische Operationen
- JSON für Daten-Persistierung

---

### 5.3.3 solver-v2: Solver (Phase 2)

Der Solver besteht aus zwei Teilen: Preparation und Edge Solver V2.

#### Preparation (Phase 2a)

**Verantwortlichkeit:**
Die Preparation-Phase lädt die Analyse-Daten, konvertiert SVG zu Bildern und segmentiert die Konturen in einzelne Segmente für das Matching.

**Kernfunktionen:**
1. **Data Loading:** Lädt Analyse-Daten aus JSON und SVG-Dateien
2. **SVG Conversion:** Konvertiert SVG zu OpenCV-Bildern
3. **Contour Segmentation:** Unterteilt Konturen in Segmente (zwischen Ecken)
4. **Initial Visualization:** Erstellt erste Visualisierung der vorbereiteten Daten

**Schnittstellen:**
- **Eingabe:** Temp-Folder-Name (string)
- **Ausgabe:** PreparedData (dict) mit Analyse-Daten, Bildern, Segmenten

---

#### Edge Solver V2 (Phase 2b)

**Entspricht "Matching-Komponente" in Laufzeitsicht Szenario 1 (Schritte 13-15)**

**Verantwortlichkeit:**
Edge Solver V2 ist der primäre Matching-Algorithmus. Er vergleicht Segmente zwischen allen Pieces, bewertet Passgenauigkeit und erstellt Progressive Chains für den Zusammenbau.

**Kernfunktionen:**
1. **Segment Finding:** Identifiziert Frame-adjacent Segmente pro Piece
2. **Segment Matching:** Vergleicht alle Segment-Paare zwischen Pieces (via Visualizer)
3. **Score Calculation:** Berechnet Composite Scores (Length 30%, Shape 50%, Direction 20%)
4. **Chain Building:** Erweitert Matches zu längeren Chains (Progressive Chains)
5. **Connection Selection:** Wählt beste Verbindungen (Greedy-Algorithmus, Top-2 pro Piece)
6. **Geometry Utilities:** ICP-Alignment, RMSD-Berechnung, Transformationen

**Algorithmus:**
- Für jedes Piece-Paar und jedes Segment-Paar wird ein Match-Score berechnet
- Composite Score kombiniert Längen-Ähnlichkeit, Form-Ähnlichkeit (RMSD) und Richtungskompatibilität
- ConnectionSelector wählt Top-2-Verbindungen pro Piece
- ChainMatcher erweitert Matches bidirektional zu Chains
- Blue/Red Dots werden für Assembly bestimmt

**Schnittstellen:**
- **Eingabe:** PreparedData (dict)
- **Ausgabe:** EdgeSolverResults (dict) mit Matches, Chains, Connections

---

### 5.3.4 solver-v2: Assembly (Phase 3)

**Entspricht "Assembly" in Laufzeitsicht Szenario 1 (Schritte 16-21)**

**Verantwortlichkeit:**
Der Assembly Solver platziert alle Puzzleteile virtuell im Zusammenbau-Raum basierend auf den erkannten Verbindungen und Chains.

**Kernfunktionen:**
1. **Anchor Selection:** Wählt Piece mit höchstem Connection-Score als Anker
2. **Orientation Normalization:** Rotiert alle Pieces so, dass Blue Dots nach oben zeigen
3. **Iterative Placement:** Platziert Pieces schrittweise:
   - Translation: Blue Dot des neuen Pieces auf Blue Dot des verbundenen Pieces
   - Rotation: Red Dots werden ausgerichtet (Richtungsvektoren)
4. **Visualization:** Erstellt 8-Schritt-Visualisierung des Zusammenbau-Prozesses

**Transformationen:**
- **Bildraum → Zusammenbau-Raum:** Koordinatensystem-Transformation
- **Translation:** Verschiebung entlang Verbindungsvektor
- **Rotation:** Drehung um Blue Dot (Ankerpunkt)

**Schnittstellen:**
- **Eingabe:** EdgeSolverResults (dict), PreparedData (dict)
- **Ausgabe:** AssemblyResults (dict) mit Transformations-Matrix, Visualisierungen

---

---

## Zusammenfassung der Bausteine

### Hauptkomponenten (Ebene 1)

| Komponente | Untermodule | Hauptverantwortlichkeit |
|------------|------------|-------------------------|
| **Simulator** | simulator-gui, puzzleSolver, solver-v2 | Orchestriert Puzzle-Lösungspipeline |

### Komponentenhierarchie

```
Simulator
├── simulator-gui
│   ├── PuzzleSimulatorWindow
│   └── PipelineWorker
├── puzzleSolver (Legacy, alternativ)
└── solver-v2 (Hauptkomponente)
    ├── Analyzer (Phase 1)
    │   ├── PuzzleAnalyzer (Core)
    │   ├── CornerDetector
    │   ├── SVGVisualizer
    │   ├── SVGSmoother
    │   ├── SVGCornerDrawer
    │   └── CornerVisualizer
    │
    ├── Solver (Phase 2)
    │   ├── Preparation
    │   │   ├── PuzzlePreparer
    │   │   ├── PuzzleDataLoader
    │   │   ├── SVGDataLoader
    │   │   ├── ContourSegmenter
    │   │   └── SolverVisualizer
    │   └── Edge Solver V2
    │       ├── EdgeSolver
    │       ├── SegmentFinder
    │       ├── ChainMatcher
    │       ├── ConnectionSelector
    │       ├── GeometryUtils
    │       └── Visualizers (4 Module)
    │   ├── EdgeSolver
    │   ├── SegmentFinder
    │   ├── ChainMatcher
    │   ├── ConnectionSelector
    │   ├── GeometryUtils
    │   └── Visualizers (4 Module)
    │
    └── Assembly (Phase 3)
        ├── AssemblySolver
        └── AssemblyVisualizer
```

### Pipeline-Ablauf (entsprechend Laufzeitsicht)

```
simulator-gui (PipelineWorker)
    │
    ├─→ Phase 1: solver-v2/analyze.py
    │       └─→ Puzzle Analyzer
    │           └─→ temp/analysis_TIMESTAMP/ (JSON, SVG, PNG)
    │
    └─→ Phase 2+3: solver-v2/puzzle_solver/solver.py
            ├─→ Preparation (lädt Analyse-Daten)
            ├─→ Edge Solver V2 (Matching)
            └─→ Assembly Solver (Zusammenbau)
```

### Kritische Abhängigkeiten

- **simulator-gui** → **solver-v2**: Ruft `analyze.run_analysis()` und `solver.solve_puzzle()` auf
- **Preparation** → **Analyzer**: Liest Analyse-Ergebnisse aus Dateisystem
- **Edge Solver V2** → **Preparation**: Nutzt PreparedData
- **Assembly** → **Edge Solver V2**: Verwendet Chains und Connections
- **Alle solver-v2 Module** → **Common**: Nutzen gemeinsame Datenstrukturen

---
