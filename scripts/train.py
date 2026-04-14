"""
Entraîne un modèle YOLOv8 pour la détection et reconnaissance faciale.

Le modèle détecte les visages ET identifie la personne en un seul passage.

Usage:
    python scripts/train.py \
        --data data/dataset/data.yaml \
        --model yolov8n.pt \
        --epochs 80 \
        --imgsz 224 \
        --batch 32 \
        --output models/

Ou enchaîner avec la préparation :
    python scripts/prepare_dataset.py && python scripts/train.py
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    output_dir: str,
    name: str,
):
    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        # Augmentation — essentielle pour la reconnaissance faciale
        flipud=0.0,       # Ne pas retourner verticalement (visages)
        fliplr=0.5,       # Miroir horizontal OK
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=10.0,     # Légère rotation
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
        mixup=0.1,
        # Optimisation
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        patience=20,      # Early stopping
        # Sauvegarde
        project=output_dir,
        name=name,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # Copier le meilleur modèle à la racine de models/
    best_pt = Path(output_dir) / name / "weights" / "best.pt"
    dest = Path(output_dir) / "face_yolo.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, dest)
        print(f"\n✅ Meilleur modèle sauvegardé : {dest}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/dataset/data.yaml")
    parser.add_argument("--model",     default="yolov8n.pt",
                        help="Modèle de base : yolov8n.pt | yolov8s.pt | yolov8m.pt")
    parser.add_argument("--epochs",    type=int, default=80)
    parser.add_argument("--imgsz",     type=int, default=224,
                        help="Taille image (224 suffit pour des visages centrés)")
    parser.add_argument("--batch",     type=int, default=32)
    parser.add_argument("--output",    default="models")
    parser.add_argument("--name",      default="face_recognition_v1")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.output, args.name)
