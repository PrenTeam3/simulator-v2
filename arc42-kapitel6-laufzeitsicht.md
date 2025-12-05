# arc42 Kapitel 6: Laufzeitsicht - Puzzle Solver V2

## Versionsinformation
- **Dokumentversion:** 2.0
- **Datum:** 2025-12-05
- **Status:** Final

---

## Inhaltsverzeichnis

1. [6.1 Szenario 1: Komplette Pipeline über GUI](#61-szenario-1-komplette-pipeline-über-gui)
2. [6.2 Szenario 2: Erkennung von Rahmen-Ecken](#62-szenario-2-erkennung-von-rahmen-ecken)
3. [6.3 Szenario 3: Segment-Matching und Verbindungsauswahl](#63-szenario-3-segment-matching-und-verbindungsauswahl)
4. [6.4 Szenario 4: Erweiterung eines Segment-Paars zur Chain](#64-szenario-4-erweiterung-eines-segment-paars-zur-chain)
5. [6.5 Szenario 5: Zusammenbau der Puzzleteile](#65-szenario-5-zusammenbau-der-puzzleteile)

---

## 6.1 Szenario 1: Komplette Pipeline über GUI

**Beschreibung:** User startet die GUI, wählt ein Bild aus und lässt es mit dem Edge Solver V2 analysieren und lösen.

**Akteure:** User, Simulator (GUI), PipelineWorker (Thread), Analyzer, Solver, Assembly

**Ablauf:**

![Szenario 1](diagrams/szenario1-komplette-pipeline.puml)

**Phasen:**

### Phase 1: Analysis (Schritte 1-12)
- User startet GUI und wählt Bild und Solver aus
- Simulator startet PipelineWorker Thread
- Analyzer führt Piece Detection, Corner Analysis und SVG-Erstellung durch
- Temporary folder und analysis_data werden zurückgegeben
- UI wird aktualisiert (Step 1 ✓)

### Phase 2: Solving (Schritte 13-16)
- Solver bereitet Puzzle-Daten vor
- Edge Solver V2 führt Segment-Matching durch
- EdgeSolverResults werden an Assembly übergeben

### Phase 3: Assembly (Schritte 17-19)
- Assembly platziert Puzzleteile basierend auf Chains
- Assembly Results werden zurückgegeben

### Phase 4: Finalization (Schritte 20-21)
- PipelineWorker meldet Abschluss
- Simulator zeigt alle Ergebnisse an (all 4 steps ✓)

**Ausgaben:**
- `temp/analysis_TIMESTAMP/output.png` - Detected pieces
- `temp/analysis_TIMESTAMP/analysis_data.json` - Metadata
- `temp/puzzle_connections_v2.png` - Connections visualization
- `temp/assembly_steps_combined.png` - Assembly steps

**Quellcode-Referenzen:**
- Simulator: `simulator/main.py:247` (PuzzleSimulatorApp)
- PipelineWorker: `simulator/main.py:63` (run method)
- Analyzer: `solver-v2/analyze.py:run_analysis()`
- Solver: `solver-v2/puzzle_solver/solver.py:solve_puzzle()`

---

## 6.2 Szenario 2: Erkennung von Rahmen-Ecken

**Beschreibung:** Der CornerDetector analysiert ein Puzzleteil und identifiziert Frame Corners (90°-Ecken) mit adaptiver Strictness-Anpassung.

**Akteure:** PuzzleAnalyzer, CornerDetector, Validation

**Ablauf:**

![Szenario 2](diagrams/szenario2-frame-corner-detection.puml)

**Adaptive Strictness Mechanismus:**

Der CornerDetector verwendet einen iterativen Ansatz mit bis zu 5 Iterationen:

1. **Initiale Contour Approximation** (Schritt 2)
2. **Iterative Corner Detection mit Validation** (Schritte 3-10):
   - Detect corners mit aktuellem strictness level
   - Validiere jeden Corner anhand 4 Kriterien
   - Zähle gefundene Frame Corners
3. **Adaptive Anpassung** (Schritte 11-12):
   - Falls `frame_corners == target`: ✓ Fertig
   - Falls `frame_corners < target`: Reduziere Strictness (strict → balanced → loose)
   - Falls `frame_corners > target`: Erhöhe Strictness (loose → balanced → strict)

**Validierungskriterien für Frame Corners:**

```python
# Criteria 1: Both adjacent edges are straight
if not (both_edges_straight):
    return False

# Criteria 2: Angle ≈ 90° (±15° tolerance)
if not (75° < angle < 105°):
    return False

# Criteria 3: Inward Arrow Test
# Vector to centroid must fall within 90° opening
if not (vector_points_inward):
    return False

# Criteria 4: Convexity
# Corner point must lie on convex hull
if corner not in convex_hull:
    return False

return True  # Valid frame corner
```

**Strictness Levels:**

| Level | max_deviation | angle_tolerance | min_straightness |
|-------|---------------|-----------------|------------------|
| ultra_strict | 0.5 | 10° | 0.95 |
| strict | 1.0 | 12° | 0.90 |
| balanced | 2.0 | 15° | 0.85 |
| loose | 3.0 | 18° | 0.80 |
| ultra_loose | 5.0 | 20° | 0.75 |

**Beispiel-Iteration:**

| Iteration | Strictness | Frame Corners | Action |
|-----------|------------|---------------|--------|
| 1 | balanced | 2 | ❌ Too few → reduce to `loose` |
| 2 | loose | 6 | ❌ Too many → increase to `balanced` |
| 3 | balanced (angle_tol=12°) | 4 | ✓ Target reached |

**Quellcode-Referenzen:**
- `solver-v2/puzzle_analyzer/corner_detector.py:is_frame_corner()`
- `solver-v2/puzzle_analyzer/core.py:analyze_piece_corners()`

---

## 6.3 Szenario 3: Segment-Matching und Verbindungsauswahl

**Beschreibung:** EdgeSolverV2 findet die besten Verbindungen zwischen allen Puzzleteilen durch Segment-Matching und erstellt Progressive Chains.

**Akteure:** EdgeSolverV2, MatchingEngine, SimilarityCalculator, ConnectionSelector

**Ablauf:**

![Szenario 3](diagrams/szenario3-edge-solver-v2.puml)

**Phase 1: Vorbereitung (Schritte 1-3)**
- Frame Corner Validation
- Frame Segment Identification

**Phase 2: Segment Matching (Schritte 4-11)**

Für jedes Piece-Paar (i, j) und jedes Segment-Paar:

1. **Length Similarity** (Schritt 6)
   ```
   length_score = max(0, 100 - (length_diff / avg_length) * 100)
   ```

2. **Shape Similarity** (Schritt 7) - ICP-basiert
   - Iterative Closest Point Alignment
   - RMSD-basierte Bewertung

3. **Direction Compatibility** (Schritt 8)
   - Normale Vektoren müssen entgegengesetzt sein (~180°)
   - Prüft ob Segmente zueinander passen

4. **Composite Score** (Schritt 9)
   ```
   composite_score = 0.3 × length_score + 0.5 × shape_score + 0.2 × direction_score
   ```

**Berechnungsbeispiel:**

```
Piece 0, Segment 2 ↔ Piece 1, Segment 0:
  Length Similarity:  0.92  (very similar)
  Shape Similarity:   0.85  (ICP-based)
  Direction Compat.:  1.00  (perfectly compatible)

  Composite Score = 0.3 × 0.92 + 0.5 × 0.85 + 0.2 × 1.00
                  = 0.276 + 0.425 + 0.200
                  = 0.901 ✓ (excellent connection)
```

**Phase 3: Connection Selection (Schritte 12-15)**

ConnectionSelector verwendet folgende Strategie:
- Wähle für jedes Piece die top 2 Verbindungen
- Erstelle Progressive Chains aus den Matches
- Berücksichtige Chain-Qualität und -Länge

**Progressive Chains Konzept:**

Nach dem Matching werden Verbindungen zu Chains organisiert:

```
Chain 1: Piece 0 ──[Seg 2-0]── Piece 1 ──[Seg 3-1]── Piece 2
         (Score: 0.901)        (Score: 0.874)

Chain 2: Piece 3 ──[Seg 1-2]── Piece 0
         (Score: 0.856)
```

Diese Chains werden in Szenario 4 erweitert und in Szenario 5 für die Assembly verwendet.

**Quellcode-Referenzen:**
- `solver-v2/puzzle_solver/edge_solver_v2/solver.py:solve_with_edges()`
- `solver-v2/puzzle_solver/edge_solver_v2/matching.py`
- `solver-v2/puzzle_solver/matrix_solver/similarity_calculator.py`

---

## 6.4 Szenario 4: Erweiterung eines Segment-Paars zur Chain

**Beschreibung:** Der ChainMatcher erweitert ein initiales Segment-Paar (Segment Match) zu einer längeren Sequential Chain durch forward/backward Extension.

**Akteure:** ChainMatcher, Extension, Scoring, Endpoints, ChainMatch

**Ablauf:**

![Szenario 4](diagrams/szenario4-progressive-chain-building.puml)

**Input:**
```
Initial Segment Match:
  Piece 0 Segment 2 ↔ Piece 1 Segment 0
  Scores: length=92, shape=85, direction=100
```

**Phase 1: Initialization (Schritte 1-2)**
- Start mit initialem Segment Match als Chain-Basis
- Chain State: P0:[Seg 2] ↔ P1:[Seg 0], Length: 1

**Phase 2: Bidirectional Extension (Schritte 2-7)**

Die Extension läuft in beide Richtungen:

### Forward Extension (Schritt 3-4):
```python
next_seg_id = (current_seg_id + 1) % num_segments  # Wraparound
```
- Piece 0: Segment 3 (2+1)
- Piece 1: Segment 1 (0+1)

### Backward Extension:
```python
prev_seg_id = (current_seg_id - 1) % num_segments  # Wraparound
```
- Piece 0: Segment 1 (2-1)
- Piece 1: Segment 3 (0-1)

**Scoring der Combined Chain (Schritt 4):**

```python
def calculate_combined_chain_scores(chain_segments_p1, chain_segments_p2):
    # 1. Combined Length Score
    total_length_p1 = sum(segment_lengths_p1)
    total_length_p2 = sum(segment_lengths_p2)
    length_score = 100 - (abs(total_length_p1 - total_length_p2) / avg_length) * 100

    # 2. Average Direction Score
    avg_normal_p1 = mean(normals_p1)
    avg_normal_p2 = mean(normals_p2)
    angle_between = arccos(dot(avg_normal_p1, avg_normal_p2))
    direction_score = 100 - abs(180° - angle_between)

    # 3. Combined Shape Score (RMSD)
    rmsd = sqrt(mean(point_distances²))
    shape_score = 100 - (rmsd / avg_length) * 100

    return length_score, direction_score, shape_score
```

**Stopping Criteria (Schritte 5-6):**
- Extension stoppt wenn **alle scores < thresholds**
- Verhindert Qualitätsverlust der Chain

**Phase 3: Endpoint Determination (Schritte 8-9)**

Blue/Red Dot Markierung für Assembly:

```python
if first_segment_adjacent_to_frame:
    blue_dot = first_endpoint   # Frame connection
    red_dot = last_endpoint     # Interior direction
else:
    blue_dot = last_endpoint
    red_dot = first_endpoint
```

**Output:**
```
Extended Chain:
  Piece 0 [Seg 1-2-3] ═══ Piece 1 [Seg 3-0-1]
  Length: 3, Score: 89.7

ChainMatch Object:
  - piece1_id: 0, piece2_id: 1
  - segment_ids_p1: [1, 2, 3]
  - segment_ids_p2: [3, 0, 1]
  - chain_length: 3
  - scores: length=89, shape=82, direction=98
  - blue_dot_p1, red_dot_p1 (endpoints für Assembly)
  - blue_dot_p2, red_dot_p2
```

**Quellcode-Referenzen:**
- `solver-v2/puzzle_solver/edge_solver_v2/chain_matcher.py:build_chains_from_matches()`
- `solver-v2/puzzle_solver/edge_solver_v2/geometry_utils.py:align_chains()`

---

## 6.5 Szenario 5: Zusammenbau der Puzzleteile

**Beschreibung:** AssemblySolver platziert Puzzleteile schrittweise basierend auf den Progressive Chains und Blue/Red Dot Markierungen.

**Akteure:** Input, Preparation, Anchor Selection, Piece Placement, Visualization, Output

**Ablauf:**

![Szenario 5](diagrams/szenario5-assembly.puml)

**Phase 1: Preparation (Schritte 1-3)**

1. **Calculate Centroids** (Schritt 1)
   - Berechne geometrischen Mittelpunkt jedes Pieces

2. **Normalize Orientations** (Schritt 2)
   ```python
   # Rotate all pieces so blue dots point upward (standard orientation)
   for piece in pieces:
       rotation = calculate_rotation_to_vertical(piece.blue_dot_vector)
       piece.apply_rotation(rotation)
   ```

**Phase 2: Anchor Selection & Placement (Schritte 3-4)**

```python
def select_anchor_piece(pieces, connections):
    """Select piece with highest total connection score"""
    anchor = max(pieces, key=lambda p: sum(conn.score for conn in p.connections))
    return anchor

# Place anchor at origin
T₀ = {
    position: (0, 0),
    rotation: calculated_from_normalization
}
```

**Phase 3: Iterative Piece Placement (Schritte 5-7)**

Für jedes verbleibende Piece (1 bis N-1):

### Step 5: Find Best Connection
```python
def find_best_connection(piece, placed_pieces):
    """Find highest-scoring connection to already placed pieces"""
    best_conn = max(
        (conn for conn in piece.connections if conn.target in placed_pieces),
        key=lambda c: c.score
    )
    return best_conn.anchor_piece_id, best_conn.chain
```

### Step 6: Translation
```python
# Align blue dots (frame connection points)
translation_vector = anchor_blue_dot - piece_blue_dot
piece.translate(translation_vector)
```

### Step 7: Rotation around Blue Dot
```python
# Align red dots (interior direction markers)
current_vector = piece_red_dot - piece_blue_dot
target_vector = anchor_red_dot - anchor_blue_dot

rotation_angle = angle_between(current_vector, target_vector)
piece.rotate_around_point(piece_blue_dot, rotation_angle)

# Store transformation
Tᵢ = {
    position: translation_vector,
    rotation: rotation_angle,
    pivot: piece_blue_dot
}
```

**Mathematische Transformationen:**

```
Piece N Placement (N ≥ 1):

Given:
  - Anchor Piece A (already placed)
  - Connection: Piece N Segment S_N ↔ Piece A Segment S_A
  - Blue Dot N: B_N (frame connection endpoint)
  - Red Dot N:  R_N (interior direction endpoint)
  - Blue Dot A: B_A (on anchor piece)
  - Red Dot A:  R_A (on anchor piece)

Step 1 - Translation:
  T = B_A - B_N
  Piece N' = Piece N + T

Step 2 - Rotation around B_A:
  v_current = R_N' - B_A  (vector from blue to red on piece N)
  v_target  = R_A - B_A   (vector from blue to red on anchor)
  θ = angle_between(v_current, v_target)

  Rotate Piece N' by θ around B_A

Final Transformation:
  T_N = {translation: T, rotation: θ, pivot: B_A}
```

**Phase 4: Visualization (Schritt 8)**

Generiere 8 Visualisierungsschritte:

1. **Step 1**: Centroids - Zeigt berechnete Mittelpunkte
2. **Step 2**: Orientations - Blue/Red Dots nach Normalisierung
3. **Step 3**: Anchor Selection - Hervorgehobenes Anchor Piece
4. **Step 4**: Anchor Placement - Anchor am Ursprung
5. **Step 5**: 2nd Piece Placement - Erstes verbundenes Piece
6. **Step 6**: 3rd Piece Placement
7. **Step 7**: 4th Piece Placement
8. **Step 8**: Complete Assembly - Finales Ergebnis

**Blue Dot / Red Dot Konzept:**

Jedes Chain-Segment hat zwei markierte Endpunkte:
- **Blue Dot:** Endpoint näher am Piece-Zentroid (Ankerpunkt für Frame-Verbindung)
- **Red Dot:** Endpoint weiter vom Zentroid (Richtungspunkt für Innenbereich)

Diese Markierungen dienen als Referenzpunkte für die Chain Alignment Transformation bei der Assembly.

**Ausgabe:**
- Einzelne Visualisierungsschritte (steps 1-8)
- `assembly_steps_combined.png` - Alle Schritte kombiniert

**Quellcode-Referenzen:**
- `solver-v2/puzzle_solver/assembly_solver/assembler.py:assemble_puzzle()`
- `solver-v2/puzzle_solver/assembly_solver/assembly_visualizer.py`
- `solver-v2/puzzle_solver/edge_solver_v2/geometry_utils.py`

---

## Zusammenfassung der Szenarien

| Szenario | Fokus | Key Components |
|----------|-------|----------------|
| 1 | End-to-End Pipeline | GUI → Analyzer → Solver → Assembly |
| 2 | Frame Corner Detection | Adaptive Strictness, 4 Validation Criteria |
| 3 | Segment Matching | Similarity Scores, Connection Selection |
| 4 | Chain Extension | Forward/Backward Extension, Blue/Red Dots |
| 5 | Puzzle Assembly | Anchor Selection, Piece Placement, Transformations |

**Workflow-Reihenfolge:**
1. **User-Interaktion** (Szenario 1)
2. **Piece-Analyse** mit Frame Corner Detection (Szenario 2)
3. **Segment-Matching** für alle Piece-Paare (Szenario 3)
4. **Chain-Erweiterung** zu längeren Sequenzen (Szenario 4)
5. **Assembly** basierend auf Chains (Szenario 5)

---

## Technische Hinweise

### Coordinate Systems
- **Image Space**: Origin top-left, Y-axis down
- **Assembly Space**: Origin center, Y-axis up (nach Normalisierung)

### Performance Considerations
- Segment Matching: O(N² × M²) für N Pieces, M Segments
- Chain Extension: Max iterations begrenzt auf reasonable values
- Assembly: O(N) für N Pieces (greedy placement)

### Error Handling
- Frame Corner Detection: Max 5 adaptive iterations, dann best effort
- Segment Matching: Threshold-basiertes Filtering
- Chain Extension: Quality-basierter Abbruch
- Assembly: Fallback zu partial assembly wenn nicht alle Pieces platzierbar

---
