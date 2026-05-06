#!/usr/bin/env bash
set -e
PY=/root/miniconda3/envs/syscall-anomaly-linux/bin/python
cd /root/s428
OUT_BASE=models/case_study/probe_normal
mkdir -p "$OUT_BASE"

for R in r1 r6 r12 r18; do
  echo "=== probing freecad_normal_${R} ==="
  $PY linux/scripts/case_study_extract.py \
      --raw-dir data/raw/freecad_only \
      --normal-case "freecad_normal_${R}" \
      --abnormal-cases freecad_bulk_export_r5,freecad_source_copy_r5,freecad_project_scan_r5 \
      --output-dir "${OUT_BASE}/${R}" \
      2>&1 | tail -20
  echo "--- summary for ${R} ---"
  cat "${OUT_BASE}/${R}/session_vote_summary.json" | python -c "import json,sys; d=json.load(sys.stdin); [print(f\"  {x[\"case_stem\"]:<32} n={x[\"n_chunks\"]:>3}  min={x[\"min_proba_abnormal\"]:.3f} max={x[\"max_proba_abnormal\"]:.3f} mean={x[\"mean_proba_abnormal\"]:.3f} pred_abn={x[\"n_predicted_abnormal_chunks\"]}/{x[\"n_chunks\"]}  vote={x[\"final_session_pred\"]} correct={x[\"correct\"]}\") for x in d]"
done
echo "all done"
