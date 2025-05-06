import argparse
import time
from pathlib import Path
from ultralytics import YOLO

def finetune_yolo_world(
    model_path: str,
    data_path: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    patience: int = 20,
    project: str = "runs/train",
    name: str = "exp"
):
    """Fine-tune YOLO-World v2 model on a custom dataset."""
    try:
        # Load model
        print(f"Loading YOLO-World v2 model: {model_path}...")
        model = YOLO(model_path)
        
        # Start timing
        start_time = time.time()
        
        # Fine-tune
        print(f"Starting fine-tuning on dataset: {data_path}")
        model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=patience,
            project=project,
            name=name,
            verbose=True
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time / 60:.2f} minutes")
        
        # Evaluate on validation set
        print("Evaluating fine-tuned model on validation set...")
        metrics = model.val(data=data_path, imgsz=imgsz, batch=batch, device=device)
        print(f"Validation metrics: mAP@50 = {metrics.box.map50:.4f}, mAP@50:95 = {metrics.box.map:.4f}")
        
        # Save model
        output_path = Path(project) / name / "weights" / "best.pt"
        print(f"Fine-tuned model saved to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        print("Ensure model path, data.yaml, and dependencies are correct.")
        raise

def main(
    model_path: str,
    data_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    patience: int,
    project: str,
    name: str
):
    """Run fine-tuning for YOLO-World v2."""
    finetune_yolo_world(
        model_path=model_path,
        data_path=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=project,
        name=name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO-World v2 Model")
    parser.add_argument("--model", type=str, default="yolov8s-worldv2.pt", help="Path to YOLO-World v2 model (.pt file)")
    parser.add_argument("--data", type=str, default="/home/geraldebmer/Pictures/dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (e.g., '0' for GPU, 'cpu' for CPU)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    args = parser.parse_args()
    
    main(
        model_path=args.model,
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name
    )