import cv2
import random
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import argparse

def run_yolo_detection(model_path: str, source: str, conf: float = 0.3, save: bool = True):
    """Run YOLOv8 detection, save bounding boxes as JSON, copy input image, and return output paths."""
    try:
        # Load YOLOv8 model
        model = YOLO(model_path)
        
        # Run prediction
        results = model.predict(
            source=source,
            conf=conf,
            save=save,
            project="runs/detect",
            name="predict",
            exist_ok=True
        )
        
        # Get the output directory and image paths
        output_dir = Path("runs/detect/predict")
        output_image = output_dir / Path(source).name
        input_image_copy = output_dir / f"input_{Path(source).name}"
        
        if not output_image.exists():
            raise FileNotFoundError(f"Output image not found at {output_image}")
        
        # Copy input image to output directory
        shutil.copy(source, input_image_copy)
        if not input_image_copy.exists():
            raise FileNotFoundError(f"Failed to copy input image to {input_image_copy}")
        
        # Extract bounding boxes, labels, and scores
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] in xyxy format
        scores = results[0].boxes.conf.cpu().numpy()  # [N]
        labels = ["block"] * len(boxes)  # Single class 'block'
        
        # Create JSON data
        bboxes_json = [
            {
                "label": label,
                "bbox_modal": [int(x1), int(y1), int(x2), int(y2)]
            }
            for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores)
        ]
        
        # Save JSON
        json_output = output_dir / f"{Path(source).stem}_bboxes.json"
        with open(json_output, 'w') as f:
            json.dump(bboxes_json, f, indent=2)
        
        return str(input_image_copy), str(json_output), str(output_image)
    except Exception as e:
        print(f"Error during YOLO detection: {str(e)}")
        raise

def display_image(image_path: str, window_name: str = "YOLO Detection"):
    """Display an image using OpenCV."""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image at {image_path}")
        
        # Display the image
        cv2.imshow(window_name, img)
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error displaying image: {str(e)}")
        raise

def random_image_detection(model_path: str, dataset_dir: str, conf: float = 0.3, save: bool = True):
    """Select a random image from the dataset, run YOLO detection, and display the result."""
    try:
        # Define the image directory
        image_dir = Path(dataset_dir) / "train" / "images"
        if not image_dir.exists():
            raise ValueError(f"Image directory not found at {image_dir}")
        
        # Get list of image files
        image_files = list(image_dir.glob("*.jpg"))
        if not image_files:
            raise ValueError(f"No .jpg images found in {image_dir}")
        
        # Select a random image
        random_image = random.choice(image_files)
        print(f"Selected random image: {random_image}")
        
        # Run YOLO detection
        input_image, json_output, output_image = run_yolo_detection(model_path, str(random_image), conf, save)
        
        # Display the result
        display_image(output_image, window_name=f"YOLO Detection - {random_image.name}")
        
        return input_image, json_output, output_image
    except Exception as e:
        print(f"Error in random image detection: {str(e)}")
        raise

def main(
    model_path: str,
    source: str,
    dataset_dir: str,
    conf: float,
    save: bool,
    random_mode: bool
):
    """Main function to run YOLO detection and display results."""
    try:
        if random_mode:
            # Run detection on a random image from the dataset
            input_image, json_output, output_image = random_image_detection(model_path, dataset_dir, conf, save)
            print(f"Input image: {input_image}")
            print(f"Bounding boxes saved to: {json_output}")
            print(f"Detected image saved to: {output_image}")
        else:
            # Run detection on the specified image
            input_image, json_output, output_image = run_yolo_detection(model_path, source, conf, save)
            # Display the result
            display_image(output_image)
            print(f"Input image: {input_image}")
            print(f"Bounding boxes saved to: {json_output}")
            print(f"Detected image saved to: {output_image}")
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection and display results with OpenCV")
    parser.add_argument("--model", type=str, default="src/detection/runs/train/exp/weights/best.pt", 
                        help="Path to YOLOv8 model (.pt file)")
    parser.add_argument("--source", type=str, default="/home/geraldebmer/Documents/Megapose/examples/legoblock/image.png", 
                        help="Path to input image")
    parser.add_argument("--dataset-dir", type=str, default="/home/geraldebmer/Pictures/dataset", 
                        help="Path to dataset directory containing train/images/")
    parser.add_argument("--conf", type=float, default=0.3, 
                        help="Confidence threshold for detection")
    parser.add_argument("--save", action="store_true", default=True, 
                        help="Save detection results")
    parser.add_argument("--random", action="store_true", 
                        help="Run detection on a random image from the dataset")
    args = parser.parse_args()
    
    main(
        model_path=args.model,
        source=args.source,
        dataset_dir=args.dataset_dir,
        conf=args.conf,
        save=args.save,
        random_mode=args.random
    )