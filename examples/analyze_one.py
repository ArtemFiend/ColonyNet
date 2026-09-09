"""Minimal programmatic example for one local image."""
from pathlib import Path

from colonynet.pipeline import (
    load_pipeline_models,
    make_full_pipeline_config,
    run_single_image_pipeline,
)


image = Path("sample.jpg")
config = make_full_pipeline_config("outputs/example")
segmenter, petri_detector = load_pipeline_models(config)
result = run_single_image_pipeline(image, segmenter, config, petri_detector)

print(f"Detected colonies: {len(result['scores'])}")
print(f"Selected anomalies: {len(result['selected'])}")
