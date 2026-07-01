# Validation Dashboard

The post-hoc dashboard is an additive reporting tool. It does not change runtime
behavior and it does not require any new launch arguments.

## Inputs

The dashboard consumes the artifacts that the packaged runtime already produces:

- JSONL evaluation logs from `JsonlEvaluationLogger`
- image manifests plus PNG frames from `ImageFrameLogger`
- the `.vnsdb` reference database used during the run

## Usage

```bash
python3 simulation/scripts/generate_visualization_dashboard.py \
  --log logs/ground_truth_*.jsonl \
  --manifest logs/images/manifest_*.jsonl \
  --database simulation/database/qau_campus.vnsdb \
  --output reports/dashboard
```

The script writes:

- `reports/dashboard/index.html`
- `reports/dashboard/overlays/*.png`

Each overlay reconstructs feature correspondences offline from the saved frame
and the matched reference image recorded in the log.

## Notes

- This is a best-effort offline reconstruction. It does not request extra
  runtime fields and it does not modify the main ROS node.
- If a saved frame, reference image, or `matched_ref_id` is missing, the sample
  still appears in the HTML table with a note instead of an overlay.
