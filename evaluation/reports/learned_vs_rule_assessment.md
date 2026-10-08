## Bewertung

Alle Aussagen gelten nur für die **Simulation** auf dem Test-Split (46 Aufgaben, nie im
Training). Hyperparameter wurden vor dem ersten Lauf festgelegt und danach nicht verändert –
auch nicht nach Sichtung der Testergebnisse.

**Nachgewiesen (in dieser Simulation):**

* Weniger Fehlgriffe nach unten: kritische Capability-Verletzungen 1 → 0 (vis-013: Screenshot +
  CSS landet beim Learned Router auf dem multimodalen Modell mit Coding-Fähigkeit), schwere
  Fehler „zu schwaches Modell“ 12 → 1, simulierter Verifikationserfolg 60 % → 87 %
  (`all_available`) bzw. 56 % → 80 % (`degraded`).
* Die harten Regeln hielten: 0 nicht verfügbare Modelle, 0 unberechtigte Fehlschläge,
  0 verworfene Vorschläge des Wächters (der Ranker bekommt nur gültige Kandidaten zu sehen).

**Nicht besser bzw. schlechter:**

* Exakte Routing-Genauigkeit sinkt (62 % → 49 %, `degraded` 67 % → 42 %): Der Learned Router
  überdimensioniert systematisch (17 statt 2 Fälle) – FAST- und GENERAL-Aufgaben gehen an große,
  langsamere Modelle. Simulierte Latenz +14 s bzw. +26 s im Mittel.
* Höhere Fallback-Rate im Ausfallszenario (24 % → 44 %): Er bevorzugt genau die großen Modelle,
  die in `degraded` ausfallen.
* Routing-Zeit 0.5 → 2–3 ms (Merkmalsberechnung) – für lokale Inferenz vernachlässigbar, aber
  messbar höher.
* Paargenauigkeit 87 % (Training) vs. 69 % (Test): deutliche Überanpassung bei 103
  Trainingsaufgaben.

**Ursachen (aus Gewichten und Entscheidungen abgeleitet):**

1. *Konstruktionsfehler, im Lauf entdeckt:* Paare entstehen erst ab Nutzenabstand 0.01
   (`pair_margin`), Latenzunterschiede gleich guter Modelle kosten aber nur 0.001–0.004. Der
   Ranker sieht daher nie ein Paar „schneller ist besser“ und lernt nur „stärker ist sicherer“.
   Korrektur für den nächsten Lauf: `pair_margin` unter die Latenzauflösung senken bzw. Latenz
   stärker gewichten – Auswahl per Kreuzvalidierung **auf dem Trainings-Split**.
2. Der Klassifikator schätzt Komplexität oft zu niedrig (Baseline); der Ranker kompensiert das,
   indem er Textmerkmale (Länge, Zahlen, Reasoning-/Code-Signale) mit Modellstärke verknüpft –
   und übertreibt dabei bei echten FAST-Aufgaben (`x_cat_fast*coding` +0.84).

**Gesamturteil:** Der Learned Router ist in dieser Simulation **sicherer** (weniger zu schwache
und ungeeignete Modelle), aber **langsamer und weniger präzise** in der Wahl des *passenden*
Modells. Ein „besser“ im Sinne der Zielfunktion (bestes Ergebnis bei vertretbarer Latenz) ist
**nicht** nachgewiesen. Er sollte nicht als Standard-Router aktiviert werden, bevor (a) der
Konstruktionsfehler behoben, (b) per Kreuzvalidierung auf dem Training neu bewertet und
(c) mit echten Routing-Log-Daten bestätigt ist.
