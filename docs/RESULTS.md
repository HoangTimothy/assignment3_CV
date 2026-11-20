# Verified results

Frozen SmartDoc test (native-image Polygon IoU; NCE uses the GT bounding-box diagonal):

| Method | Frames | Valid | IoU, all | NCE, valid | NCE median, valid | Runtime ms |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 120 | 55 | 0.423271 | 0.057760 | 0.004222 | 71.862613 |
| refined | 120 | 55 | 0.423889 | 0.057602 | 0.003536 | 111.071496 |

IoU includes failures as zero. NCE is conditional on valid predictions. Detection failures remain in the denominator; valid geometry is not necessarily correct localization.

Frozen FUNSD test: 50 source scans, 150 derived views, five branches. Primary metric: **ROI-assisted OCR (dataset text boxes provided)**. E4 is an oracle and is excluded from automatic-system success rates.

| condition | method | entities | failed_entities | CER | WER |
| --- | --- | --- | --- | --- | --- |
| frontal | E0_raw | 2270 | 0 | 0.174574 | 0.452629 |
| frontal | E1_crop | 2270 | 0 | 0.176929 | 0.452400 |
| frontal | E2_basic | 2270 | 0 | 0.194129 | 0.478634 |
| frontal | E3_refined | 2270 | 0 | 0.190507 | 0.477260 |
| frontal | E4_oracle | 2270 | 0 | 0.179146 | 0.459732 |
| mild | E0_raw | 2270 | 0 | 0.489757 | 0.826097 |
| mild | E1_crop | 2270 | 0 | 0.489737 | 0.826097 |
| mild | E2_basic | 2270 | 0 | 0.210815 | 0.502234 |
| mild | E3_refined | 2270 | 0 | 0.212181 | 0.502234 |
| mild | E4_oracle | 2270 | 0 | 0.203294 | 0.498912 |
| strong | E0_raw | 2270 | 0 | 0.659323 | 1.043418 |
| strong | E1_crop | 2270 | 0 | 0.659323 | 1.043418 |
| strong | E2_basic | 2270 | 0 | 0.236011 | 0.540955 |
| strong | E3_refined | 2270 | 0 | 0.238862 | 0.545538 |
| strong | E4_oracle | 2270 | 0 | 0.242167 | 0.546340 |

Complete-sample scores include empty predictions for failed regions. Conditional metrics, document-pooled summaries, paired changes and region-size strata are retained in the run. Views share source documents; no significance claim is made. These controlled scan results do not establish real-camera OCR gains.

Exact paths, hashes and independent audits: [run index](../report/artifacts/run_index.json). See [report](../report/report.md) and [limitations](LIMITATIONS.md).
