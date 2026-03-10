# Puzzle Solver — Algorithmus Arbeitsdokument

> Dieses Dokument wird schrittweise erarbeitet und ergaenzt.
> Status-Legende: ✅ Fertig | 🔄 In Arbeit | ⬜ Noch offen

---

## Uebersicht

**Ziel:** Aus den erkannten Aussenlinien und Ecken jedes Puzzleteils die korrekte Position und Rotation im 190x128mm Rahmen bestimmen.

**Rahmen:** Fix im Querformat — 190mm breit, 128mm hoch.

**Input:** Puzzleteile mit klassifizierten Konturen
- Gerade Aussenlinien (`is_outside=True`, gelb) = Rahmen-Kanten
- Ecken (gruen) = 90°-Verbindungspunkte zwischen zwei geraden Kanten

**Output:** Pro Teil — Position (x, y), Rotation (Winkel), zugewiesene Rahmenecke/-seite

---

## Schritt 1 — Rahmen definieren ⬜

Der Rahmen ist fix im Querformat:
- Oben / Unten: 190mm
- Links / Rechts: 128mm

**Pixel-zu-mm Umrechnung:**
```
px_per_mm = a4_image_width_px / 297.0
length_mm = segment_length_px / px_per_mm
```

**Notizen:**
- [ ] `a4_image_width_px` kommt direkt aus den Inputdaten

---

## Schritt 2 — Border Info extrahieren ⬜

Pro Teil alle Segmente mit `is_outside=True` sammeln und in mm umrechnen.

Jedes Teil enthaelt:
- `piece_idx` — Index des Teils
- `outside_segments` — Liste von `{seg_id, piece_idx, length_mm, p1, p2}`
- `centroid_px` — Pixel-Schwerpunkt der Kontur
- `px_per_mm` — Skalierungsfaktor

**Notizen:**
- [ ] Input sind sowohl das Bild als auch alle Segmentdaten bereits als Variablen gespeichert
- [ ] Segmentlaengen sind bereits berechnet — muessen nicht neu ermittelt werden
- [ ] `corner_pairs` und `straight_edges` sind direkt aus den Inputvariablen lesbar

---

## Schritt 3 — Varianten pro Teil generieren ⬜

Pro Teil werden alle sinnvollen Verwendungsmoeglichkeiten aus den **bereits vorhandenen Inputdaten** gelesen. Der Analyzer hat die Ecken und geraden Aussenlinien bereits klassifiziert — das muss hier nicht neu berechnet werden.

**Input pro Teil (bereits bekannt):**
- `corner_pairs` — Liste von Kanten-Paaren die eine Ecke bilden (z.B. [(a,b)])
- `straight_edges` — Liste aller geraden Aussenlinien (z.B. [a, b, c])

**Beispiel:** Teil mit Kanten a, b (bereits als Ecke markiert) und Kante c (separat gerade):
```
Variante 1: [a+b]   → als Eckteil (aus corner_pairs gelesen)
Variante 2: [a]     → als Randteil
Variante 3: [b]     → als Randteil
Variante 4: [c]     → als Randteil
```

**Pseudo-Code:**
```
for each piece:
    variants = []

    # Ecken-Einheiten direkt aus Inputdaten lesen
    for (a, b) in piece.corner_pairs:
        variants.append({ type: 'corner', edges: [a, b] })

    # Jede einzelne gerade Kante als separate Variante
    for edge in piece.straight_edges:
        variants.append({ type: 'edge', edges: [edge] })

    piece.variants = variants
```

**Notizen:**
- [ ] Vorerst werden nur einzelne Kanten und zusammenhaengende Ecken-Paare beruecksichtigt
- [ ] ⚠️ SPAETER AUFNEHMEN: Kombinationen von nicht-zusammenhaengenden Kanten (z.B. [a+b, c] oder [a, c]) wurden bewusst weggelassen da geometrisch komplexer — muss spaeter ergaenzt werden

---

## Schritt 4 — Aehnlichkeitsanalyse zwischen Teilen ⬜

Direkt nach der Variantengenerierung werden alle Teile paarweise verglichen um potenzielle Gleichteile zu erkennen. Kein hartes Ja/Nein — nur ein Aehnlichkeits-Score wird mitgefuehrt.

**Pseudo-Code:**
```
for each pair (piece_A, piece_B):

    # Schritt 1: Flaeche vergleichen (billig)
    flaechen_diff = abs(flaeche_A - flaeche_B) / max(flaeche_A, flaeche_B)
    if flaechen_diff > 0.05:  # ±5% Toleranz
        pair.similarity = 0
        continue  # sicher nicht gleich

    # Schritt 2: Kontur vergleichen via Hu Moments (rotationsinvariant)
    shape_score = cv2.matchShapes(kontur_A, kontur_B, cv2.CONTOURS_MATCH_I1, 0)

    # Schritt 3: Score speichern
    pair.similarity_score = shape_score        # tief = sehr aehnlich
    pair.likely_duplicate = shape_score < 0.10 # 10% Toleranz
```

**Notizen:**
- [ ] `likely_duplicate` ist ein Hinweis, kein hartes Urteil
- [ ] Flaechentoleranz: ±5%
- [ ] Form-Toleranz (Hu Moments Schwellwert): 0.10 (10%) — muss experimentell validiert werden
- [ ] Score wird in Schritt 7 (Tree-based) verwendet um symmetrische Kombinationen zu ueberspringen

---

## Schritt 5 — Constraints definieren ⬜

Hier werden alle Constraints definiert die spaeter (Schritt 6) auf die Kombinationen aller Teile angewendet werden. Nur Kombinationen die **alle** Constraints erfuellen kommen weiter.

**Constraint 1 — Genau 4 Rahmenecken besetzt**
```
corners_used = [v for v in combination if v.type == 'corner']
assert len(corners_used) == 4
assert alle positionen {TL, TR, BR, BL} genau einmal besetzt
```

**Constraint 2 — Jede Rahmenseite vollstaendig abgedeckt**
```
oben   = sum(kanten die oben zugewiesen sind)   → 190mm ± 15mm (minus ~1mm/Verbindung)
unten  = sum(kanten die unten zugewiesen sind)  → 190mm ± 15mm (minus ~1mm/Verbindung)
links  = sum(kanten die links zugewiesen sind)  → 128mm ± 15mm (minus ~1mm/Verbindung)
rechts = sum(kanten die rechts zugewiesen sind) → 128mm ± 15mm (minus ~1mm/Verbindung)
```

**Constraint 3 — Jedes Teil genau einmal verwendet**
```
assert len(combination) == anzahl_teile
assert keine duplikate (jedes piece_idx genau einmal)
```

**Constraint 4 — Jedes Teil beruehrt den Rahmen**
```
for each piece in combination:
    assert piece.variant.edges ist nicht leer
    assert mindestens eine Kante ist einer Rahmenseite zugewiesen
```

**Constraint 5 — Zentroid innen**
```
for each piece in combination:
    rotated_centroid = rotate(piece.centroid, assigned_rotation)
    assert inside(rotated_centroid, frame_190x128)
```

**Notizen:**
- [ ] Constraints werden in Schritt 6 in dieser Reihenfolge angewendet (billigste zuerst, teuerste zuletzt)
- [ ] Constraint 1-4 sind schnell zu berechnen → fruehzeitig filtern
- [ ] Constraint 5 (Zentroid) benoetigt bereits eine Rotation → etwas teurer, wird zuletzt angewendet
- [ ] Zentroid = Pixelschwerpunkt der Masse des Puzzleteils
- [ ] Toleranz ±15mm ist vorerst statisch — muss experimentell validiert werden, lieber zu grosszuegig

---

## Schritt 6 — Kombination der Teile (Tree-based) ⬜

Startet mit einem Eckteil in TL in 2 Orientierungen (lange Kante oben oder lange Kante links). Arbeitet sich dann entlang des Rahmens vor:

```
TL →[oben]→ TR →[rechts]→ BR →[unten]→ BL →[links]→ TL (geschlossen)
```

Pro Position wird jedes verbleibende Teil / jede Variante probiert:

```
aktuell_laenge += kandidat.kanten_laenge

if aktuell_laenge < seiten_ziel:
    # Noch Platz → Teil passt, naechstes Teil auf dieser Seite probieren
    weiter im Baum

elif aktuell_laenge ≈ seiten_ziel AND kandidat.type == 'corner':
    # Seite genau gefuellt UND Teil ist Ecke → Ecke erreicht ✅
    weiter zur naechsten Seite

else:
    # Zu lang oder Seite voll aber kein Eckteil → Ast abschneiden ❌
```

Zusaetzlich bei jedem platzierten Teil:
- Zentroid-Check → Teil muss nach innen zeigen

Iteration laeuft bis man wieder bei TL ankommt. Abschluss-Check:
- Alle Teile verwendet? (4 oder 6)
- Letzte Kante (links) schliesst den Rahmen korrekt — Laenge stimmt exakt (±Toleranz)?

→ Wenn ja: gueltige Loesung gefunden ✅

**Maximale Loesungsanzahl begrenzen:**
```
MAX_SOLUTIONS = 10  # konfigurierbar

if len(gueltige_loesungen) >= MAX_SOLUTIONS:
    baum_aufbau_abbrechen()
```

**Notizen:**
- [ ] Jedes Teil wird einmal als Startecke probiert
- [ ] Pro Startecke 2 Orientierungen (lange Kante oben oder links)
- [ ] Funktioniert fuer 4 und 6 Teile — bei 6 Teilen kommen einfach mehr Teile pro Seite
- [ ] MAX_SOLUTIONS ist konfigurierbar — Standardwert 10
- [ ] Es wird davon ausgegangen dass mindestens 4 Eckteile vorhanden sind
- [ ] ⚠️ Es ist moeglich dass mehr oder weniger als 4 Ecken erkannt werden — Verhalten in diesem Fall noch ungeklaert

---

## Schritt 7 — Geometrische Platzierung, Visualisierung & Overlap-Check ⬜

Fuer jede gueltige Kombination aus Schritt 6 wird das Ergebnis geometrisch platziert, visuell ausgegeben und auf Ueberschneidungen geprueft.

**7.1 — Geometrische Platzierung**
Pro Teil:
- Frame-Corner finden (gemeinsamer Endpunkt von horiz und vert Kante)
- Rotation berechnen damit Kanten zur zugewiesenen Rahmenseite ausgerichtet sind
- Kontur in mm umrechnen, zentrieren, rotieren und auf 190x128 Canvas platzieren

**7.2 — Visuelle Ausgabe**
Pro Loesung wird ein separates Bild gespeichert mit:
- Jedes Teil in einer anderen Farbe gezeichnet
- Rahmen (190x128) als Referenzrechteck eingezeichnet
- Ueberschneidungen speziell markiert (z.B. rot)
- Beschriftung pro Teil: piece_idx, zugewiesene Position, Rotation in Grad

So kann der Entwickler alle Kandidaten visuell pruefen und vergleichen.

**7.3 — Overlap-Check**
- Alle Teile auf Pixel-Canvas zeichnen (5px/mm → 950x640px)
- Ueberlappende Pixel zaehlen → Umrechnung in mm²
- Toleranz: 50mm²
- Innerhalb Toleranz → gueltige Loesung ✅
- Ueber Toleranz → Loesung verwerfen ❌

**Notizen:**
- [ ] Canvas-Koordinatenursprung: oben-links = (0, 0)
- [ ] Pro Kandidat aus Schritt 6 ein separates Bild ausgeben
- [ ] Bilder mit Loesung-Index benennen (z.B. solution_01.png, solution_02.png)
- [ ] Frame-Corner wird vorerst 1:1 aus Inputdaten uebernommen — Endpunkte werden als korrekt angenommen
- [ ] ⚠️ SPAETER: Robustheit verbessern falls Endpunkte der corner_pairs durch Messungenauigkeit leicht versetzt sind

---

## Schritt 8 — Finale Loesung ausgeben ⬜

Aus allen gueltigen Loesungen aus Schritt 7 wird die beste ausgewaehlt und ausgegeben.

**8.1 — Beste Loesung waehlen (Score-basiert)**
```
beste_loesung = min(gueltige_loesungen, key=lambda l: l.overlap_mm2)
```
Falls mehrere Loesungen denselben Overlap haben → gleichmaessigere Verteilung der Teile als Tiebreaker.

**8.2 — Output pro Teil**
- Position (x, y) im Rahmen in mm
- Rotation in Grad
- Zugewiesene Rahmenposition (TL, TR, BR, BL)

**8.3 — Finales Bild**
- Beste Loesung sauber visualisiert als Referenzbild (solution_final.png)

**Notizen:**
- [ ] Vorerst Testphase — kein direkter Roboter-Output
- [ ] Pro gueltige Loesung wird eine .txt Datei generiert
- [ ] Zusaetzlich wird ein Ranking aller Loesungen erstellt (sortiert nach Overlap)
- [ ] Null, eine oder mehrere Loesungen sind moeglich

---

## Offene Fragen / TODO

- [ ] ⚠️ Schritt 7: Robustheit des Frame-Corner Findens verbessern — was passiert wenn Endpunkte der corner_pairs durch Messungenauigkeit leicht versetzt sind? Vorerst wird auf korrekte Inputdaten vertraut.
- [ ] Toleranzwerte experimentell validieren: ±15mm (Laenge), 50mm² (Overlap), 0.10 (Hu Moments), ±5% (Flaeche)
- [ ] ⚠️ Schritt 4: Kombinationen von nicht-zusammenhaengenden Kanten noch nicht beruecksichtigt — spaeter aufnehmen

---

## Aenderungshistorie

| Datum | Aenderung |
|---|---|
| 10.03.2026 | Initiales Dokument erstellt |
| 10.03.2026 | Schritt 3 neu eingefuegt: Aehnlichkeitsanalyse. Alle nachfolgenden Schritte um 1 erhoeht |
| 10.03.2026 | Offene Punkte aus Entwickler-Review in Schritte 2, 3, 5, 7, 8 eingetragen |
