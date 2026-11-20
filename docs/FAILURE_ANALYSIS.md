# Observed failure cases

The frozen SmartDoc test has 65 `no_candidate` rows and 55 geometrically valid outputs per method. Validity alone is not object-localization correctness. A reviewed refined prediction on `background05/magazine005/frame_0035.jpeg` selected the background book: native-image IoU 0, NCE 0.899005. This remains a valid-output row with poor accuracy, rather than being relabeled as successful localization.

A deterministic local review sheet compares one miss per background plus the worst-NCE valid output: `results/20261119T172258558113Z_geometry_refined_test/failure_review.jpg`, with exact rows in `failure_review.json`. These are qualitative observations, not new dataset annotations or causal measurements:

| Source frame | Visible condition | Recorded outcome |
| --- | --- | --- |
| `background01/datasheet005/frame_0141.jpeg` | Faint lower page boundary on wood | No valid contour candidate |
| `background02/letter005/frame_0131.jpeg` | Small page in a broad tabletop view | No valid contour candidate |
| `background03/datasheet005/frame_0084.jpeg` | Dimmer, uneven illumination | No valid contour candidate |
| `background04/datasheet005/frame_0029.jpeg` | Sideways printed page on a pale surface | No valid contour candidate |
| `background05/datasheet005/frame_0017.jpeg` | Cable/objects cross the page | No valid contour candidate |
| `background05/magazine005/frame_0035.jpeg` | Competing rectangular printed book | Wrong-object valid prediction |

Edge maps, candidates, masks and montages for every detection failure are retained in the run's `examples/` directory. Line refinement cannot recover a page when contour detection returns no candidate. These cases motivate future development-data work, without changing this frozen test experiment.

OCR degradation examples are selected from measured document-level E3-minus-E0 CER in each completed run's `worsened_documents.csv` and `error_case_gallery.csv`. Their raw/basic/refined contact sheets remain local, under `error_cases/`. The report states known ROI assistance and keeps oracle results separate. A degrading score does not by itself establish the physical cause; interpolation, source quality, handwriting, ROI clipping or corner error require case-specific review.
