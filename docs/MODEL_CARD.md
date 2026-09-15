# Model card and limitations

## Intended use

ColonyNet assists visual review of Petri-dish photographs by segmenting colony-like objects and ranking unusual instances. It is designed for controlled laboratory exploration and portfolio demonstration.

## Inputs and outputs

The application accepts common raster image formats. It produces a normalized dish crop, instance masks, tabular features, ranked candidates, technical warnings, and visual overlays.

## Interpretation

The anomaly score is relative to other valid objects on the same plate. It is not a calibrated probability, species identification, contamination diagnosis, or antimicrobial-susceptibility result. A highlighted object requires human review.

## Known limitations

- Lighting, reflections, condensation, dish edges, labels, and image compression can affect detection and segmentation.
- Rare but valid morphotypes can receive high anomaly scores.
- Poor masks are separated into review/technical groups where possible, but automatic filtering is imperfect.
- Performance claims require a locked test set split by physical plate or experiment. Random image-level splits can overestimate generalisation.
- The validated checkpoints are not distributed with this repository; results depend on the local weights supplied by the user.

## Privacy

Inference runs locally. Reports can reproduce source pixels and filenames, so they should be handled with the same controls as the original laboratory images.
