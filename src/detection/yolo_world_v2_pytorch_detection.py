import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

def load_yolo_world_model(model_path: str = "yolov8s-worldv2.pt"):
    """Load YOLO-World v2 model from a .pt file."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        print("Ensure the .pt file exists and ultralytics is installed (`pip install ultralytics`).")
        raise

def detect_objects_yolo_world(
    image: np.ndarray,
    model,
    prompt: str,
    confidence_threshold: float = 0.3
) -> tuple[np.ndarray, np.ndarray, list]:
    """Detect objects using YOLO-World v2 with a text prompt and measure inference time."""
    # Resize image to 640x640 for faster inference
    # For even faster inference, change to (320, 320) and adjust scale_x, scale_y
    orig_h, orig_w = image.shape[:2]
    resized_image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    # Set custom classes for open-vocabulary detection
    model.set_classes([prompt])
    
    # Measure inference time
    start_time = time.time()
    results = model.predict(
        resized_image,
        conf=confidence_threshold,
        verbose=False
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.3f} seconds")
    
    # Extract detections
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] in xyxy format
    scores = results[0].boxes.conf.cpu().numpy()  # [N]
    
    # Use prompt as label
    labels = [prompt] * len(boxes)
    
    # Scale boxes back to original image size
    scale_x, scale_y = orig_w / 640, orig_h / 640
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    
    return boxes, scores, labels

def visualize_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: list) -> np.ndarray:
    """Visualize detected bounding boxes on the image."""
    vis_image = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(vis_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return vis_image

def main(image_path: Path, prompt: str, model_path: str = "yolov8s-worldv2.pt", confidence_threshold: float = 0.3):
    """Run YOLO-World v2 detection and visualize results."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Load model
    print(f"Loading YOLO-World v2 model: {model_path}...")
    model = load_yolo_world_model(model_path)
    
    # Run detection
    print(f"Running detection with prompt: {prompt}")
    boxes, scores, labels = detect_objects_yolo_world(image, model, prompt, confidence_threshold)
    
    # Visualize results
    vis_image = visualize_detections(image, boxes, scores, labels)
    
    # Display result
    cv2.imshow("YOLO-World V2 Detections", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = image_path.parent / f"{image_path.stem}_yolo_world_detections.jpg"
    cv2.imwrite(str(output_path), vis_image)
    print(f"Saved detection visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-World V2 Object Detection")
    parser.add_argument("--image", type=Path, required=True, help="Path to input RGB image")
    parser.add_argument("--prompt", type=str, default="block", help="Text prompt for detection (e.g., block)")
    parser.add_argument("--model", type=str, default="yolov8s-worldv2.pt", help="Path to YOLO-World v2 model (.pt file)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold for detections")
    args = parser.parse_args()
    
    main(args.image, args.prompt, args.model, args.confidence)