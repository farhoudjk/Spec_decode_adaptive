#!/usr/bin/env bash
# Recommended order: cheap go/no-go tests first, full RQ battery only after.
set -euo pipefail
W="${WORKERS:-}"
ARGS=${W:+--workers $W}

echo "== stage 1: premise tests (run these before anything else) =="
python -m specloop.cli pilot     --seeds 8 $ARGS
python -m specloop.cli regime    --seeds 4 $ARGS
python -m specloop.cli mechanism --seeds 6 $ARGS

echo "== stage 2: full RQ battery =="
python -m specloop.cli rq1 --seeds 24 $ARGS
python -m specloop.cli rq2 --seeds 5  $ARGS
python -m specloop.cli rq3 --seeds 10 $ARGS
python -m specloop.cli rq4 --seeds 10 $ARGS
python -m specloop.cli rq5 --seeds 6  $ARGS

echo "== stage 3: ablations =="
python -m specloop.cli abl-controller --seeds 8 $ARGS
python -m specloop.cli abl-loops      --seeds 8 $ARGS
python -m specloop.cli abl-slo        --seeds 6 $ARGS
