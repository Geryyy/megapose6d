import argparse
import time
import warnings
import torch
import os
import shutil
from pathlib import Path
from ultralytics import YOLO, __version__ as ultralytics_version
from sklearn.model_selection import KFold, train_test_split
import yaml
import numpy as np
from packaging import version

# Suppress multiprocessing cleanup warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.multiprocessing")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

def read_data_yaml(data_path: Path):
    """Read class names from data.yaml."""
    try:
        with open(data_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if not names or not isinstance(names, list):
            raise ValueError("data.yaml must contain a 'names' field with a list of class names (e.g., ['Block'])")
        return names
    except Exception as e:
        raise ValueError(f"Failed to read data.yaml: {str(e)}")

def check_label_file(label_path: Path, num_classes: int):
    """Check if a label file contains valid YOLO-format annotations."""
    if not label_path.exists():
        return False
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            class_id, x, y, w, h = map(float, parts)
            if not (0 <= class_id < num_classes and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                return False
        return len(lines) > 0 and any(line.strip() for line in lines)
    except Exception:
        return False

def create_split_folders(dataset_dir: Path, split_ratio: float, seed: int = 42):
    """Split train folder into train and valid sets, ensuring valid labels."""
    image_dir = dataset_dir / "train" / "images"
    label_dir = dataset_dir / "train" / "labels"
    
    images = sorted([f for f in image_dir.glob("*.jpg") if f.is_file()])
    labels = [label_dir / img.with_suffix(".txt").name for img in images]
    
    # Ensure images have corresponding labels with valid annotations
    num_classes = len(read_data_yaml(dataset_dir / "data.yaml"))
    valid_pairs = [(img, lbl) for img, lbl in zip(images, labels) if lbl.exists() and check_label_file(lbl, num_classes)]
    if not valid_pairs:
        raise ValueError("No images with valid label annotations found in dataset. Check label files in {}".format(label_dir))
    
    images, labels = zip(*valid_pairs)
    images, labels = list(images), list(labels)
    print(f"Found {len(images)} images with valid annotations")
    
    # Split into train and valid
    train_idx, valid_idx = train_test_split(
        range(len(images)), train_size=split_ratio, random_state=seed, shuffle=True
    )
    
    return [(images[i], labels[i]) for i in train_idx], [(images[i], labels[i]) for i in valid_idx]

def setup_fold_folders(dataset_dir: Path, train_files: list, valid_files: list, fold: int):
    """Create temporary train and valid folders for a fold."""
    fold_dir = dataset_dir / f"fold_{fold}"
    train_img_dir = fold_dir / "train" / "images"
    train_lbl_dir = fold_dir / "train" / "labels"
    valid_img_dir = fold_dir / "valid" / "images"
    valid_lbl_dir = fold_dir / "valid" / "labels"
    
    for d in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Copy train files
    for img, lbl in train_files:
        shutil.copy(img, train_img_dir / img.name)
        shutil.copy(lbl, train_lbl_dir / lbl.name)
    
    # Copy valid files
    for img, lbl in valid_files:
        shutil.copy(img, valid_img_dir / img.name)
        shutil.copy(lbl, valid_lbl_dir / lbl.name)
    
    # Create data.yaml for this fold with absolute paths
    data_yaml = fold_dir / "data.yaml"
    yaml_content = {
        "train": str(train_img_dir.absolute()),
        "val": str(valid_img_dir.absolute()),
        "nc": len(read_data_yaml(dataset_dir / "data.yaml")),
        "names": read_data_yaml(dataset_dir / "data.yaml")
    }
    with open(data_yaml, "w") as f:
        yaml.dump(yaml_content, f)
    
    # Check training and validation labels
    num_classes = len(read_data_yaml(dataset_dir / "data.yaml"))
    train_instances = sum(1 for img, lbl in train_files if check_label_file(lbl, num_classes))
    valid_instances = sum(1 for img, lbl in valid_files if check_label_file(lbl, num_classes))
    print(f"Fold {fold} training set: {len(train_files)} images, {train_instances} images with annotations")
    print(f"Fold {fold} validation set: {len(valid_files)} images, {valid_instances} images with annotations")
    
    if train_instances == 0:
        raise ValueError(f"Fold {fold} training set has no images with annotations. Check dataset or reduce n_folds.")
    if valid_instances == 0:
        print(f"WARNING: Fold {fold} validation set has no images with annotations. Metrics will be zero.")
    
    return data_yaml

def finetune_yolov8(
    model_path: str,
    data_path: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    patience: int = 20,
    workers: int = 1,
    fliplr: float = 0.5,
    flipud: float = 0.5,
    degrees: float = 45.0,
    hsv_v: float = 0.4,
    mosaic: float = 0.8,
    scale: float = 0.2,
    n_folds: int = 5,
    split_ratio: float = 0.8,
    project: str = "runs/train",
    name: str = "exp",
    seed: int = 42,
    cumulative: bool = True
):
    """Fine-tune YOLOv8 with K-fold cross-validation and optional cumulative learning."""
    try:
        # Check Ultralytics version
        required_version = "8.3.0"
        if version.parse(ultralytics_version) < version.parse(required_version):
            raise ValueError(f"Ultralytics version {ultralytics_version} is outdated. Please upgrade to {required_version} or later: pip install --upgrade ultralytics")
        
        dataset_dir = Path(data_path).parent
        class_names = read_data_yaml(data_path)
        results = []
        best_map50 = 0.0
        best_model_path = None
        previous_best_path = None  # Track the best model from the previous fold
        
        if n_folds > 1:
            print(f"Running {n_folds}-fold cross-validation{' with cumulative learning' if cumulative else ''}...")
            images = sorted([f for f in (dataset_dir / "train" / "images").glob("*.jpg") if f.is_file()])
            labels = [dataset_dir / "train" / "labels" / img.with_suffix(".txt").name for img in images]
            images = [img for img, lbl in zip(images, labels) if lbl.exists()]
            labels = [lbl for lbl in labels if lbl.exists()]
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold, (train_idx, valid_idx) in enumerate(kf.split(images), 1):
                print(f"\nTraining Fold {fold}/{n_folds}...")
                
                # Load model
                if fold == 1 or not cumulative or not previous_best_path or not os.path.exists(previous_best_path):
                    print(f"Initializing with pretrained model: {model_path}")
                    model = YOLO(model_path)  # Use pretrained model (e.g., yolov8s.pt)
                else:
                    print(f"Loading fine-tuned model from: {previous_best_path}")
                    model = YOLO(previous_best_path)  # Use previous fold's best.pt
                
                # Setup fold folders and data.yaml
                train_files = [(images[i], labels[i]) for i in train_idx]
                valid_files = [(images[i], labels[i]) for i in valid_idx]
                
                if len(valid_files) < 5:
                    print(f"WARNING: Fold {fold} has only {len(valid_files)} validation images. Consider reducing n_folds.")
                
                data_yaml = setup_fold_folders(dataset_dir, train_files, valid_files, fold)
                
                # Train
                start_time = time.time()
                model.train(
                    data=data_yaml,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    patience=patience,
                    workers=workers,
                    fliplr=fliplr,
                    flipud=flipud,
                    degrees=degrees,
                    hsv_v=hsv_v,
                    mosaic=mosaic,
                    scale=scale,
                    project=project,
                    name=f"{name}/fold_{fold}",
                    verbose=True
                )
                training_time = time.time() - start_time
                
                # Evaluate
                try:
                    metrics = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device, workers=workers)
                    map50 = metrics.box.map50 if metrics.box.map50 is not None else 0.0
                    map5095 = metrics.box.map if metrics.box.map is not None else 0.0
                except Exception as e:
                    print(f"Validation failed for Fold {fold}: {str(e)}. Setting metrics to 0.")
                    map50, map5095 = 0.0, 0.0
                
                results.append({"fold": fold, "map50": map50, "map5095": map5095, "time": training_time})
                
                # Save model if best
                model_path_fold = Path(project) / name / f"fold_{fold}" / "weights" / "best.pt"
                if map50 > best_map50:
                    best_map50 = map50
                    best_model_path = model_path_fold
                    dest_path = Path(project) / name / "weights" / "best.pt"
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy(model_path_fold, dest_path)
                
                # Update previous_best_path for the next fold
                previous_best_path = model_path_fold
                if cumulative:
                    print(f"Setting previous_best_path for Fold {fold+1}: {previous_best_path}")
                
                # Clean up fold folders
                shutil.rmtree(dataset_dir / f"fold_{fold}")
                
                print(f"Fold {fold} completed in {training_time / 60:.2f} minutes. mAP@50 = {map50:.4f}, mAP@50:95 = {map5095:.4f}")
        else:
            print("Running single split training...")
            train_files, valid_files = create_split_folders(dataset_dir, split_ratio, seed)
            if len(valid_files) < 5:
                print(f"WARNING: Validation set has only {len(valid_files)} images. Consider collecting more images.")
            
            data_yaml = setup_fold_folders(dataset_dir, train_files, valid_files, 0)
            print(f"Initializing with pretrained model: {model_path}")
            model = YOLO(model_path)
            
            start_time = time.time()
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                patience=patience,
                workers=workers,
                fliplr=fliplr,
                flipud=flipud,
                degrees=degrees,
                hsv_v=hsv_v,
                mosaic=mosaic,
                scale=scale,
                project=project,
                name=name,
                verbose=True
            )
            training_time = time.time() - start_time
            
            try:
                metrics = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device, workers=workers)
                map50 = metrics.box.map50 if metrics.box.map50 is not None else 0.0
                map5095 = metrics.box.map if metrics.box.map is not None else 0.0
            except Exception as e:
                print(f"Validation failed: {str(e)}. Setting metrics to 0.")
                map50, map5095 = 0.0, 0.0
            
            results.append({"fold": 0, "map50": map50, "map5095": map5095, "time": training_time})
            best_model_path = Path(project) / name / "weights" / "best.pt"
            
            shutil.rmtree(dataset_dir / "fold_0")
            
            print(f"Training completed in {training_time / 60:.2f} minutes. mAP@50 = {map50:.4f}, mAP@50:95 = {map5095:.4f}")
        
        # Save cross-validation results
        mean_map50 = np.mean([r["map50"] for r in results])
        mean_map5095 = np.mean([r["map5095"] for r in results])
        total_time = sum(r["time"] for r in results)
        
        summary = f"Cross-Validation Summary ({n_folds} folds):\n"
        summary += f"Mean mAP@50: {mean_map50:.4f}\n"
        summary += f"Mean mAP@50:95: {mean_map5095:.4f}\n"
        summary += f"Total training time: {total_time / 60:.2f} minutes\n"
        summary += f"Best model (mAP@50 = {best_map50:.4f}) saved to {best_model_path}\n"
        for r in results:
            summary += f"Fold {r['fold']}: mAP@50 = {r['map50']:.4f}, mAP@50:95 = {r['map5095']:.4f}, Time = {r['time'] / 60:.2f} minutes\n"
        
        summary_path = Path(project) / name / "cv_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        print(summary)
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_model_path
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        print("Ensure model path, data.yaml, and dependencies are correct.")
        print("Try updating Ultralytics: pip install --upgrade ultralytics")
        raise
    finally:
        # Ensure CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main(
    model_path: str,
    data_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    patience: int,
    workers: int,
    fliplr: float,
    flipud: float,
    degrees: float,
    hsv_v: float,
    mosaic: float,
    scale: float,
    n_folds: int,
    split_ratio: float,
    project: str,
    name: str,
    seed: int,
    cumulative: bool
):
    """Run fine-tuning for YOLOv8."""
    finetune_yolov8(
        model_path=model_path,
        data_path=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        workers=workers,
        fliplr=fliplr,
        flipud=flipud,
        degrees=degrees,
        hsv_v=hsv_v,
        mosaic=mosaic,
        scale=scale,
        n_folds=n_folds,
        split_ratio=split_ratio,
        project=project,
        name=name,
        seed=seed,
        cumulative=cumulative
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 Model with Cross-Validation")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Path to YOLOv8 model (.pt file, e.g., yolov8s.pt)")
    parser.add_argument("--data", type=str, default="/home/geraldebmer/Pictures/dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (e.g., '0' for GPU, 'cpu' for CPU)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    parser.add_argument("--workers", type=int, default=1, help="Number of DataLoader workers")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--flipud", type=float, default=0.5, help="Vertical flip probability")
    parser.add_argument("--degrees", type=float, default=45.0, help="Rotation angle (degrees)")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="Brightness/contrast adjustment")
    parser.add_argument("--mosaic", type=float, default=0.8, help="Mosaic augmentation probability")
    parser.add_argument("--scale", type=float, default=0.2, help="Scale augmentation factor")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of cross-validation folds (1 to disable)")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train split ratio (e.g., 0.8 for 80% train)")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cumulative", action="store_true", help="Enable cumulative learning across folds")
    args = parser.parse_args()
    
    main(
        model_path=args.model,
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        fliplr=args.fliplr,
        flipud=args.flipud,
        degrees=args.degrees,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        scale=args.scale,
        n_folds=args.n_folds,
        split_ratio=args.split_ratio,
        project=args.project,
        name=args.name,
        seed=args.seed,
        cumulative=args.cumulative
    )