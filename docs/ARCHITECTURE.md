# Architecture

ColonyNet is a local two-model inference application.

1. A YOLO detector selects the most plausible Petri-dish region from a raw photograph.
2. The region is expanded by a small margin and normalized to 736 × 736 pixels.
3. A YOLO segmentation model returns per-colony masks and confidence values.
4. Masks are cleaned, smoothed for analysis, and made non-overlapping.
5. The feature layer measures geometry, colour, texture, local background, neighbour differences, density, and segmentation reliability.
6. Robust scaling, local outlier methods, clustering, independent evidence groups, and perturbation stability produce review priorities.
7. Results are split into selected anomalies, segmentation-review candidates, and technical warnings.

The desktop UI runs inference on a worker thread and sends status/results to Tk through a queue. Each run receives a unique output directory, so a previous report is never silently overwritten. Files with duplicate case-insensitive stems are rejected before inference because per-image output folders use those stems.

The production path lives entirely in `src/colonynet`. Build metadata, examples, tests, and documentation remain separate from runtime code.
