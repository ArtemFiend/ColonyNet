# ColonyNet (SegFormer MiT-B2/B3 + sem/center/boundary + watershed)

ColonyNet trains instance segmentation with 3 heads:
- `sem`: foreground (colony/cell) vs background
- `center`: instance center heatmap
- `boundary`: boundaries between instances

Instances are recovered with marker-controlled watershed.

## Requirements
```bash
pip install -r requirements.txt
```

## Expected Dataset Format
Each dataset must be a pair of folders:
- `images/*` (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`)
- `instances/*.png` (integer instance ids)

Mask format:
- `0` = background
- `1..N` = instance IDs
- `uint16` PNG is supported and recommended for many objects

## Training
Run training with a config:
```bash
python train.py --config configs/train_trainable_pool_mit_b3.yaml
```

Most used configs:
- `configs/train_trainable_pool_mit_b2.yaml`
- `configs/train_trainable_pool_mit_b3.yaml`
- `configs/train_unified_mit_b2.yaml`
- `configs/train_unified_mit_b3.yaml`
- `configs/train_unified_petri_mit_b2.yaml`
- `configs/train_unified_petri_mit_b3.yaml`
- `configs/train_cups_mit_b2.yaml`
- `configs/train_cups_mit_b3.yaml`

For `train_trainable_pool_*` configs, training/validation use:
- `trainable_pool/images`
- `trainable_pool/instances`

Validation split is controlled by `data.val_split` in config.

## Inference
```bash
python infer.py --ckpt runs/<run_name>/best.pt --input_dir trainable_pool/images --out_dir runs/demo_preds
```

## Dataset Conversion Utilities
- DSB2018 -> instance masks:
```bash
python tools/convert_dsb2018.py --dsb_root /path/to/stage1_train --out_root data/dsb2018
```
- Build merged training pool:
```bash
python tools/build_trainable_pool.py --data_root data --all_root all_datasets --stage1_root stage1_train --out_root trainable_pool
```

## Notes
- Model heads predict at `1 / out_stride` resolution (`out_stride=4` by default).
- Ground-truth instances are downscaled with nearest-neighbor interpolation before target building.
- Watershed postprocessing thresholds are configured in each training YAML under `post`.
