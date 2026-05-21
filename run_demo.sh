#!/bin/bash
# ─────────────────────────────────────────────────────────────
# ExciPick Demo Runner — Mentor Presentation
# All APIs sourced directly from oral_only_cleaned.csv
# Usage: bash run_demo.sh
# ─────────────────────────────────────────────────────────────

THRESHOLD=0.3
DOSE_TOL=10

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         ExciPick — Formulation Recommendation Demo       ║"
echo "║     HetGNN + Incompatibility Engine  |  Oral Tablets     ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Demo 1 ─────────────────────────────────────────────────────
# Low dose tablet — known 4/4 perfect match
# Best opener: clean output, perfect ground truth
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DEMO 1 — Low Dose Oral Tablet (25mg)"
echo "  API: U3H27498KS | Known 4/4 ground truth match"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python predict.py \
    --api U3H27498KS \
    --dose 25 \
    --unit 1 \
    --route ORAL \
    --form TABLET \
    --threshold $THRESHOLD \
    --show-gt \
    --dose-tol $DOSE_TOL

echo ""
sleep 1

# ── Demo 2 ─────────────────────────────────────────────────────
# Film coated tablet — shows form awareness
# Coating excipients appear: TiO2, HPMC, PEG
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DEMO 2 — Film Coated Tablet (25mg)"
echo "  API: 3ST302B24A | Coating excipients: TiO2, HPMC, PEG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python predict.py \
    --api 3ST302B24A \
    --dose 25 \
    --unit 1 \
    --route ORAL \
    --form "TABLET, FILM COATED" \
    --threshold $THRESHOLD \
    --show-gt \
    --dose-tol $DOSE_TOL

echo ""
sleep 1

# ── Demo 3 ─────────────────────────────────────────────────────
# Mid dose — incompatibility engine flags visible
# Magnesium stearate + croscarmellose pairwise clash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DEMO 3 — Mid Dose Tablet (30mg) + Incompatibility Flags"
echo "  API: 423D2T571U | Pairwise clash warnings visible"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python predict.py \
    --api 423D2T571U \
    --dose 30 \
    --unit 1 \
    --route ORAL \
    --form TABLET \
    --threshold $THRESHOLD \
    --show-gt \
    --dose-tol $DOSE_TOL

echo ""
sleep 1

# ── Demo 4 ─────────────────────────────────────────────────────
# High dose tablet — 250mg, different excipient profile
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DEMO 4 — High Dose Tablet (250mg)"
echo "  API: 0P6C6ZOP5U | High dose shifts excipient ratios"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python predict.py \
    --api 0P6C6ZOP5U \
    --dose 250 \
    --unit 1 \
    --route ORAL \
    --form TABLET \
    --threshold $THRESHOLD \
    --show-gt \
    --dose-tol $DOSE_TOL

echo ""
sleep 1

# ── Demo 5 ─────────────────────────────────────────────────────
# Extended release — matrix formers expected
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DEMO 5 — Extended Release Tablet (30mg)"
echo "  API: LX1OH63030 | ER-specific excipients (matrix formers)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python predict.py \
    --api LX1OH63030 \
    --dose 30 \
    --unit 1 \
    --route ORAL \
    --form "TABLET, EXTENDED RELEASE" \
    --threshold $THRESHOLD \
    --show-gt \
    --dose-tol $DOSE_TOL

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Demo Complete ✓                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
