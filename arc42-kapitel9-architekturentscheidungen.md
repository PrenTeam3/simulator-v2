# arc42 Kapitel 9: Architekturentscheidungen - Puzzle Solver V2

## Versionsinformation
- **Dokumentversion:** 1.0
- **Datum:** 2025-12-05
- **Status:** Final

---

## Inhaltsverzeichnis

1. [AE-001: Edge Solver V2 mit Progressive Chains](#ae-001-edge-solver-v2-mit-progressive-chains)
2. [AE-002: RMSD-basierte Form-Ähnlichkeit](#ae-002-rmsd-basierte-form-ähnlichkeit)
3. [AE-003: Adaptive Striktheit für Rahmen-Ecken-Erkennung](#ae-003-adaptive-striktheit-für-rahmen-ecken-erkennung)
4. [AE-004: Blaue/Rote Punkt-Markierungen für Zusammenbau](#ae-004-blauerotenote-punkt-markierungen-für-zusammenbau)
5. [AE-005: Qt Threading mit Signals/Slots](#ae-005-qt-threading-mit-signalsslots)
6. [AE-006: Greedy Verbindungsauswahl statt Optimierung](#ae-006-greedy-verbindungsauswahl-statt-optimierung)
7. [AE-007: OpenCV für Computer Vision](#ae-007-opencv-für-computer-vision)

---

## AE-001: Edge Solver V2 mit Progressive Chains

**Status:** ✅ Akzeptiert

**Kontext:**
- Edge Solver V1 verwendete einzelne Segment-Matches
- Keine Information über sequenzielle Verbindungen
- Zusammenbau hatte keine klare Orientierungsinformation

**Entscheidung:**
Implementierung von **Progressive Chains** - erweiterte Segment-Matches über mehrere aufeinanderfolgende Segmente.

**Begründung:**
- **Bessere Qualität:** Längere Chains = stabilere Verbindungen
- **Orientierung:** Blaue/Rote Punkte geben klare Richtung für Zusammenbau
- **Robustheit:** Mehr Punkte für Ausrichtung = genaueres Matching

**Konsequenzen:**
- ✅ Höhere Zusammenbau-Qualität
- ✅ Klarere Orientierungsinformation
- ✅ Besseres Form-Scoring durch mehr Datenpunkte
- ⚠️ Erhöhte Komplexität bei Chain-Erweiterung
- ⚠️ Mehr Rechenaufwand für kombiniertes Scoring

**Verworfene Alternativen:**
- **ICP-basiertes vollständiges Kontur-Matching:** Zu langsam, nicht robust
- **Feature-basiertes Matching (SIFT/ORB):** Für glattes Puzzle ungeeignet

**Verwandte Themen:**
- Szenario 4: Erweiterung eines Segment-Paars zur Chain
- `chain_matcher.py`, `PROGRESSIVE_CHAIN_MATCHING.md`

---

## AE-002: RMSD-basierte Form-Ähnlichkeit

**Status:** ✅ Akzeptiert

**Kontext:**
- Verschiedene Form-Matching-Ansätze verfügbar
- Notwendigkeit für schnelle, robuste Ähnlichkeitsbewertung

**Entscheidung:**
Verwendung von **Root Mean Square Deviation (RMSD)** als primäre Form-Ähnlichkeits-Metrik.

**Begründung:**
```
form_score = 100 - (rmsd / durchschnittliche_länge) × 100
```

**Vorteile:**
- **Einfach:** Leicht verständlich und implementierbar
- **Schnell:** O(N×M) für N und M Punkte
- **Robust:** Funktioniert gut mit ausgerichteten Chains
- **Normalisiert:** Prozentual zur durchschnittlichen Länge

**Konsequenzen:**
- ✅ Gute Balance zwischen Geschwindigkeit und Genauigkeit
- ✅ Funktioniert gut für Puzzle-Teile
- ⚠️ Erfordert präzise Ausrichtung vor Berechnung
- ⚠️ Empfindlich bei sehr unterschiedlichen Segmentlängen

**Verworfene Alternativen:**
- **Hausdorff-Distanz:** Zu konservativ (maximale Distanz dominiert)
- **Chamfer-Distanz:** Ähnlich zu RMSD, aber asymmetrisch
- **ICP-Fehler:** Zu langsam für Online-Berechnung

**Implementation:**
- Gewicht: 50% des kombinierten Scores
- Schwellenwert: ≥ 80 für valide Matches

**Verwandte Themen:**
- Kapitel 8.2: Scoring-System
- `similarity_calculator.py`

---

## AE-003: Adaptive Striktheit für Rahmen-Ecken-Erkennung

**Status:** ✅ Akzeptiert

**Kontext:**
- Statische Parameter funktionieren nicht für alle Puzzle-Qualitäten
- Manuelles Tuning unpraktisch
- Unter-/Über-Erkennung problematisch für Zusammenbau

**Entscheidung:**
Implementierung eines **adaptiven Algorithmus** mit 5 Striktheitsstufen und automatischer Anpassung.

**Begründung:**
```
gefunden < ziel → reduziere Striktheit
gefunden > ziel → erhöhe Striktheit
gefunden = ziel → fertig ✓
```

**Striktheitsstufen:**
```
ultra_streng → streng → ausgewogen → locker → ultra_locker
```

**Konsequenzen:**
- ✅ Robust gegenüber unterschiedlichen Bildqualitäten
- ✅ Keine manuelle Parameter-Justierung nötig
- ✅ Konvergiert meist in 2-3 Iterationen
- ⚠️ Maximal 5 Iterationen als Obergrenze
- ⚠️ Best-Effort falls Ziel nicht erreicht wird

**Verworfene Alternativen:**
- **Machine Learning:** Überdimensioniert, benötigt Trainingsdaten
- **Statische Parameter:** Nicht robust genug
- **Binäre Suche:** Zu langsam, nicht smooth genug

**Konfiguration:**
- Maximale Iterationen: 5
- Initiale Stufe: ausgewogen
- Ziel: 4 Rahmen-Ecken (für 4-Teile-Puzzle)

**Verwandte Themen:**
- Szenario 2: Erkennung von Rahmen-Ecken
- `corner_detector.py`

---

## AE-004: Blaue/Rote Punkt-Markierungen für Zusammenbau

**Status:** ✅ Akzeptiert

**Kontext:**
- Zusammenbau benötigt klare Orientierungsinformation
- Rahmen-Verbindung vs. Innenrichtung unklar
- Rotations-Ausrichtung schwierig ohne Referenzpunkte

**Entscheidung:**
Verwendung von **Blauen/Roten Punkt-Markierungen** an Chain-Endpunkten:
- **Blauer Punkt:** Rahmen-Verbindung (näher am Schwerpunkt)
- **Roter Punkt:** Innenrichtung (weiter vom Schwerpunkt)

**Begründung:**
```
Ausrichtungsprozess:
1. Verschiebung: blauer_punkt_neu → blauer_punkt_anker
2. Rotation um blauen_punkt: roter_punkt_neu → roter_punkt_anker Richtung
```

**Konsequenzen:**
- ✅ Eindeutige Orientierung für Zusammenbau
- ✅ Robustes Ausrichtungsverfahren
- ✅ Visuell nachvollziehbar beim Debugging
- ⚠️ Zusätzliche Endpunkt-Berechnung nötig
- ⚠️ Bestimmung basiert auf Rahmen-Nachbarschaft

**Verworfene Alternativen:**
- **Schwerpunkt-basiert:** Nicht präzise genug für Rotation
- **Multiple Referenzpunkte:** Unnötig komplex
- **Nur Normalenvektoren:** Mehrdeutig bei symmetrischen Teilen

**Implementation:**
- Berechnung in Chain-Erweiterung (Szenario 4)
- Verwendung im Zusammenbau (Szenario 5)

**Verwandte Themen:**
- Szenario 4: Chain-Erweiterung
- Szenario 5: Zusammenbau
- `chain_matcher.py`, `assembler.py`

---

## AE-005: Qt Threading mit Signals/Slots

**Status:** ✅ Akzeptiert

**Kontext:**
- GUI muss während Pipeline-Ausführung responsiv bleiben
- Thread-sichere Kommunikation zwischen Worker und GUI
- Fortschritts-Updates und Abbruch-Unterstützung benötigt

**Entscheidung:**
Verwendung von **Qt QThread mit Signals/Slots** für Pipeline-Ausführung.

**Architektur:**
```python
PipelineWorker(QThread) → Signals → GUI (Haupt-Thread)
- schritt_beendet
- fortschritt_aktualisiert
- pipeline_beendet
- fehler_aufgetreten
```

**Begründung:**
- **Thread-Sicherheit:** Qt handled Marshalling automatisch
- **Einfach:** Deklarative Signal-Verbindungen
- **Robust:** Bewährtes Pattern in Qt
- **Abbruch:** Sanfter Stopp via Flag

**Konsequenzen:**
- ✅ Responsive GUI während langer Berechnungen
- ✅ Thread-sicher ohne manuelles Locking
- ✅ Einfach zu testen (Mock Signals)
- ⚠️ Qt-Abhängigkeit (aber bereits für GUI verwendet)
- ⚠️ Kein direkter Rückgabewert (via Signal)

**Verworfene Alternativen:**
- **Python Threading:** Manuelles Locking, komplexer
- **asyncio:** Nicht gut für CPU-bound Tasks
- **ProcessPoolExecutor:** Overhead, Serialisierung problematisch

**Verwandte Themen:**
- Kapitel 8.5: Threading-Modell
- `simulator/main.py`

---

## AE-006: Greedy Verbindungsauswahl statt Optimierung

**Status:** ✅ Akzeptiert

**Kontext:**
- Optimales Matching ist NP-schwer (Zuordnungsproblem)
- Brute-Force zu langsam für >4 Teile
- Heuristik benötigt für praktische Performance

**Entscheidung:**
Verwendung eines **Greedy-Algorithmus** mit Ranking und Konflikt-Vermeidung.

**Algorithmus:**
```
1. Sortiere Chains nach: 0,8 × länge + 0,2 × qualität
2. Für jede Chain (beste zuerst):
   - Prüfe Constraints (max 2 pro Teil, keine Überlappung)
   - Akzeptiere wenn gültig
   - Markiere als verwendet
3. Stoppe wenn alle Teile 2 Verbindungen haben
```

**Begründung:**
- **Performance:** O(N log N) statt exponentiell
- **Qualität:** Ranking bevorzugt längere, bessere Chains
- **Praktisch:** Funktioniert gut für Puzzle-Domäne

**Konsequenzen:**
- ✅ Schnell genug für Echtzeit
- ✅ Meist gute Ergebnisse (lokal optimal)
- ✅ Einfach zu verstehen und debuggen
- ⚠️ Nicht global optimal
- ⚠️ Reihenfolge-abhängig

**Verworfene Alternativen:**
- **Ungarischer Algorithmus:** Zu restriktiv (perfektes Matching)
- **Simulated Annealing:** Zu langsam, nicht deterministisch
- **ILP-Solver:** Überdimensioniert, externe Abhängigkeit

**Kompromisse:**
- Akzeptiere lokal optimale Lösung für schnelle Ausführung
- Ranking sorgt dafür, dass beste Verbindungen zuerst gewählt werden

**Verwandte Themen:**
- Kapitel 8.3: Adaptive Algorithmen
- `connection_selector.py`

---

## AE-007: OpenCV für Computer Vision

**Status:** ✅ Akzeptiert

**Kontext:**
- Computer-Vision-Operationen benötigt (Konturen, Ecken, etc.)
- Verschiedene Bibliotheken verfügbar

**Entscheidung:**
Verwendung von **OpenCV** als primäre CV-Bibliothek.

**Begründung:**
- **Umfassend:** Alle benötigten Funktionen vorhanden
  - Kontur-Erkennung
  - Ecken-Erkennung (Harris, Shi-Tomasi)
  - Konvexe Hülle
  - Bildverarbeitung
- **Performant:** C++ Backend, optimiert
- **Standard:** De-facto Standard für CV in Python
- **Dokumentation:** Exzellent, viele Beispiele

**Konsequenzen:**
- ✅ Alle CV-Operationen aus einer Bibliothek
- ✅ Gute Performance für Bild-Operationen
- ✅ Große Community, viel Support
- ⚠️ Große Abhängigkeit (~100MB)
- ⚠️ API manchmal unintuitiv

**Verworfene Alternativen:**
- **scikit-image:** Pythonischer, aber langsamer
- **PIL/Pillow:** Zu basic, keine CV-Algorithmen
- **Eigene Implementation:** Rad neu erfinden, langsam

**Verwendung:**
- Kontur-Erkennung: `cv2.findContours()`
- Approximation: `cv2.approxPolyDP()`
- Konvexe Hülle: `cv2.convexHull()`
- Ecken-Erkennung: `cv2.goodFeaturesToTrack()`

**Verwandte Themen:**
- `analyze.py`, `corner_detector.py`

---

## Zusammenfassung

| AE | Entscheidung | Begründung | Kompromisse |
|-----|----------|------------|------------|
| **001** | Progressive Chains | Bessere Qualität, Orientierung | Mehr Komplexität |
| **002** | RMSD Form-Scoring | Schnell, robust, normalisiert | Ausrichtungs-abhängig |
| **003** | Adaptive Striktheit | Robustheit ohne Tuning | Max 5 Iterationen |
| **004** | Blaue/Rote Punkte | Eindeutige Orientierung | Zusätzliche Berechnung |
| **005** | Qt Threading | Thread-Sicherheit, Responsive GUI | Qt-Abhängigkeit |
| **006** | Greedy-Auswahl | Performance über Optimalität | Lokal optimal |
| **007** | OpenCV | Standard, umfassend, schnell | Große Abhängigkeit |

**Übergreifende Prinzipien:**
- ✅ **Pragmatismus:** Praktische Lösungen über theoretische Optimalität
- ✅ **Performance:** Echtzeitfähigkeit wichtiger als perfekte Genauigkeit
- ✅ **Robustheit:** Adaptive Algorithmen über statische Parameter
- ✅ **Wartbarkeit:** Einfache, verständliche Algorithmen

---