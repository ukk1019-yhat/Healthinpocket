import albumentations as A
from typing import Dict


def get_train_transform(cfg: Dict) -> A.Compose:
    aug_cfg = cfg["augmentation"]["train"]
    img_size = cfg["data"]["img_size"]
    transforms = [
        A.Resize(img_size, img_size),
    ]
    if aug_cfg.get("horizontal_flip"):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_cfg.get("vertical_flip"):
        transforms.append(A.VerticalFlip(p=0.5))
    rotation = aug_cfg.get("rotation_limit", 0)
    if rotation:
        transforms.append(A.Rotate(limit=rotation, p=0.5))
    bc = aug_cfg.get("brightness_contrast")
    if bc:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc[0] - 1, contrast_limit=bc[1] - 1, p=0.5
            )
        )
    hs = aug_cfg.get("hue_saturation")
    if hs:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=hs[0] * 255,
                sat_shift_limit=hs[1] * 255,
                val_shift_limit=0,
                p=0.3,
            )
        )
    scale = aug_cfg.get("scale_limit", 0)
    if scale:
        transforms.append(A.RandomScale(scale_limit=scale, p=0.3))
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return A.Compose(transforms)


def get_eval_transform(cfg: Dict) -> A.Compose:
    img_size = cfg["data"]["img_size"]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class RetinopathyDataset:
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import cv2
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot load {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row["diagnosis"])
        augmented = self.transform(image=image)
        return augmented["image"].transpose(2, 0, 1).astype(np.float32), label
