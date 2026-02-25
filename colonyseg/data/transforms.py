import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_train_tf(img_size: int):
    # Instances is a label mask -> keep nearest interpolation (default for mask)
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-10, 10),
                border_mode=0,
                p=0.35,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.GaussNoise(p=0.1),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"instances": "mask"},
    )

def build_val_tf(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"instances": "mask"},
    )
