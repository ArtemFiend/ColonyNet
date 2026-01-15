# ColonyNet v1 (SegFormer MiT-B2/B3 + sem/center/boundary + watershed)

This is a minimal, practical starter for training **instance segmentation via watershed**
using **SegFormer MiT backbones** and **3 supervision heads**:
- sem: colony vs background
- center: instance centers heatmap
- boundary: boundaries between instances

The repo expects datasets in a simple format:
- image: RGB file
- instances: PNG with integer ids (0=background, 1..N=instances)

## Install
```bash
pip install -r requirements.txt
```

## Prepare Data Science Bowl 2018 (DSB2018 nuclei) into instances.png format
Download the dataset from Kaggle and point to the extracted `stage1_train/` folder.

Example structure (per image id):
- stage1_train/<id>/images/<id>.png
- stage1_train/<id>/masks/*.png

Run:
```bash
python tools/convert_dsb2018.py --dsb_root /path/to/stage1_train --out_root data/dsb2018
```

It will create:
- data/dsb2018/images/<id>.png
- data/dsb2018/instances/<id>.png

## Train
```bash
python train.py --config configs/train_mit_b2.yaml
```

## Inference / visualization
```bash
python infer.py --ckpt runs/colony_mit_b2/best.pt --input_dir data/dsb2018/images --out_dir runs/demo_preds
```

## Notes
- Default output stride is 4 (heads predict at 1/4 resolution).
- Postprocessing uses marker-controlled watershed.
