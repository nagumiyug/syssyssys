# Evaluation Summary

- Input data: `data/processed/freecad/combined.csv`
- 80 sessions
- 1788239 events

## Sequence-only model
- Accuracy: 0.604
- FPR: 0.008

## Sequence + context model
- Accuracy: 0.666
- FPR: 0.015

## Delta
- Accuracy gain: 0.062
- FPR drop: -0.008
