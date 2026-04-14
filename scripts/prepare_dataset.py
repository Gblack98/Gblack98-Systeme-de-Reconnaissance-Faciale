"""
Prépare le dataset YOLO à partir du dataset LFW.

Sélectionne les N personnes avec le plus d'images, divise en train/val,
et génère les annotations YOLO (bounding box pleine image, les photos LFW
étant déjà centrées sur le visage).

Usage:
    python scripts/prepare_dataset.py \
        --lfw_dir "/path/to/lfw-deepfunneled" \
        --output_dir data/dataset \
        --n_classes 20 \
        --val_split 0.2
"""

import os
import shutil
import random
import argparse
import yaml
from pathlib import Path


def get_top_n_persons(lfw_dir: str, n: int) -> list[tuple[str, list[str]]]:
    """Retourne les N personnes avec le plus d'images, triées par classe."""
    counts = {}
    for person in os.listdir(lfw_dir):
        person_dir = os.path.join(lfw_dir, person)
        if os.path.isdir(person_dir):
            images = [
                os.path.join(person_dir, f)
                for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if images:
                counts[person] = images

    sorted_persons = sorted(counts.items(), key=lambda x: len(x[1]), reverse=True)
    return sorted_persons[:n]


def write_yolo_label(label_path: str, class_id: int):
    """
    Écrit l'annotation YOLO.
    Pour les images LFW (visage centré), le bbox couvre toute l'image :
    class_id cx cy w h = class_id 0.5 0.5 1.0 1.0
    """
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def prepare(lfw_dir: str, output_dir: str, n_classes: int, val_split: float, seed: int = 42):
    random.seed(seed)
    output_dir = Path(output_dir)

    persons = get_top_n_persons(lfw_dir, n_classes)
    class_names = [name for name, _ in persons]

    print(f"\n{'─'*50}")
    print(f"  {n_classes} classes sélectionnées")
    print(f"{'─'*50}")
    for i, (name, imgs) in enumerate(persons):
        print(f"  {i:2}. {name:<35} {len(imgs)} images")

    # Créer les dossiers
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0}

    for class_id, (person_name, images) in enumerate(persons):
        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_split))
        splits = {"train": images[:split_idx], "val": images[split_idx:]}

        for split, split_images in splits.items():
            for img_path in split_images:
                stem = f"{person_name}_{Path(img_path).stem}"
                dest_img = output_dir / "images" / split / f"{stem}.jpg"
                dest_lbl = output_dir / "labels" / split / f"{stem}.txt"

                shutil.copy2(img_path, dest_img)
                write_yolo_label(str(dest_lbl), class_id)
                stats[split] += 1

    # data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": n_classes,
        "names": class_names,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n{'─'*50}")
    print(f"  Dataset prêt : {stats['train']} train | {stats['val']} val")
    print(f"  data.yaml    : {yaml_path}")
    print(f"{'─'*50}\n")
    return str(yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lfw_dir",
        default="/home/gblack98/Projets/python/travail_IGD/archive (1)/lfw-deepfunneled/lfw-deepfunneled",
    )
    parser.add_argument("--output_dir", default="data/dataset")
    parser.add_argument("--n_classes", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare(args.lfw_dir, args.output_dir, args.n_classes, args.val_split, args.seed)
