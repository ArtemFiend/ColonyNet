import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_train_tf(img_size: int):
    # Instances is a label mask -> keep nearest interpolation (default for mask)
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.10, rotate_limit=15, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
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
