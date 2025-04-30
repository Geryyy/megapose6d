# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import cv2
import numpy as np
import torch
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoProcessor

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)

def load_yolo_world_model():
    """Load YOLO-World v2 model and processor from Hugging Face."""
    model_id = "stevengrove/YOLO-World"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id).cuda()
    return processor, model

def detect_objects_yolo_world(
    image: np.ndarray,
    prompt: str,
    processor,
    model,
    confidence_threshold: float = 0.3,
) -> DetectionsType:
    """Detect objects using YOLO-World with a text prompt."""
    # Convert image to RGB (YOLO-World expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare inputs for YOLO-World
    inputs = processor(images=image_rgb, text=[prompt], return_tensors="pt").to("cuda")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract bounding boxes and scores
    boxes = outputs.pred_boxes[0].cpu().numpy()  # [N, 4] in xyxy format
    scores = outputs.scores[0].cpu().numpy()  # [N]
    
    # Filter detections by confidence
    valid_idx = scores >= confidence_threshold
    boxes = boxes[valid_idx]
    scores = scores[valid_idx]
    
    # Convert to MegaPose DetectionsType
    labels = ["block"] * len(boxes)  # Single label for all blocks
    bboxes = torch.tensor(boxes, dtype=torch.float32).cuda()
    infos = {"label": labels, "score": scores.tolist()}
    
    class Detections:
        def __init__(self, bboxes, infos):
            self.bboxes = bboxes
            self.infos = infos
        
        def cuda(self):
            return self
    
    detections = Detections(bboxes=bboxes, infos=infos)
    return detections

def load_observation(
    image_path: Path,
    camera_data: CameraData = None,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    """Load an RGB image and optional camera data."""
    rgb = np.array(Image.open(image_path), dtype=np.uint8)
    
    if camera_data is None:
        # Default camera parameters (adjust based on your setup)
        camera_data = CameraData(
            resolution=rgb.shape[:2],
            K=np.array([[800, 0, rgb.shape[1]/2],
                       [0, 800, rgb.shape[0]/2],
                       [0, 0, 1]], dtype=np.float32),  # Example intrinsic matrix
        )
    
    depth = None
    if load_depth:
        raise NotImplementedError("Depth loading not implemented in this example")
    
    return rgb, depth, camera_data

def load_observation_tensor(
    image_path: Path,
    camera_data: CameraData = None,
    load_depth: bool = False,
) -> ObservationTensor:
    """Convert image to ObservationTensor for MegaPose."""
    rgb, depth, camera_data = load_observation(image_path, camera_data, load_depth)
    
    # Convert RGB to grayscale for color-agnostic processing
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_gray = cv2.cvtColor(rgb_gray, cv2.COLOR_GRAY2RGB)  # Convert back to 3 channels
    
    observation = ObservationTensor.from_numpy(rgb_gray, depth, camera_data.K)
    return observation

def make_object_dataset(ply_path: Path) -> RigidObjectDataset:
    """Create a RigidObjectDataset from a single PLY file."""
    label = "block"
    mesh_units = "mm"  # Ensure PLY is in millimeters
    rigid_objects = [RigidObject(label=label, mesh_path=ply_path, mesh_units=mesh_units)]
    return RigidObjectDataset(rigid_objects)

def save_predictions(
    output_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    """Save estimated poses to a JSON file."""
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data], indent=2)
    output_fn = output_dir / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")

def make_visualizations(
    image_path: Path,
    ply_path: Path,
    output_dir: Path,
    pose_estimates: PoseEstimatesType,
    detections: DetectionsType,
    camera_data: CameraData,
) -> None:
    """Generate visualizations for detections and pose estimates."""
    rgb, _, _ = load_observation(image_path, camera_data, load_depth=False)
    object_dataset = make_object_dataset(ply_path)
    object_datas = [
        ObjectData(label=label, TWO=Transform(pose))
        for label, pose in zip(pose_estimates.infos["label"], pose_estimates.poses.cpu().numpy())
    ]
    
    # Initialize renderer
    renderer = Panda3dSceneRenderer(object_dataset)
    
    # Convert to Panda3D format
    camera_data.TWC = Transform(np.eye(4))
    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    
    # Render scene
    light_datas = [Panda3dLightData(light_type="ambient", color=((1.0, 1.0, 1.0, 1)))]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]
    
    # Create visualizations
    plotter = BokehPlotter()
    
    # Detections
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    
    # Pose overlays
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    
    # Combine visualizations
    fig_all = gridplot([[fig_rgb, fig_det, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    
    # Save visualizations
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    export_png(fig_det, filename=vis_dir / "detections.png")
    export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
    export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
    export_png(fig_all, filename=vis_dir / "all_results.png")
    logger.info(f"Wrote visualizations to {vis_dir}")

def run_pose_estimation(
    image_path: Path,
    ply_path: Path,
    output_dir: Path,
    prompt: str,
    model_name: str = "megapose-1.0-RGB",
    camera_data: CameraData = None,
) -> None:
    """Run 6D pose estimation using YOLO-World for detection and MegaPose for pose estimation."""
    # Load YOLO-World
    logger.info("Loading YOLO-World model.")
    processor, yolo_model = load_yolo_world_model()
    
    # Load image
    rgb, _, camera_data = load_observation(image_path, camera_data)
    
    # Detect objects
    logger.info(f"Running YOLO-World detection with prompt: {prompt}")
    detections = detect_objects_yolo_world(rgb, prompt, processor, yolo_model)
    
    # Load observation tensor (grayscale for color-agnostic processing)
    observation = load_observation_tensor(image_path, camera_data, load_depth=False).cuda()
    
    # Create object dataset
    object_dataset = make_object_dataset(ply_path)
    
    # Load MegaPose model
    logger.info(f"Loading MegaPose model: {model_name}")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()
    
    # Run inference
    logger.info("Running MegaPose inference.")
    model_info = NAMED_MODELS[model_name]
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )
    
    # Save predictions
    save_predictions(output_dir, output)
    
    # Generate visualizations
    make_visualizations(image_path, ply_path, output_dir, output, detections, camera_data)

if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser(description="6D Pose Estimation with YOLO-World and MegaPose")
    parser.add_argument("--image", type=Path, required=True, help="Path to input RGB image")
    parser.add_argument("--ply", type=Path, required=True, help="Path to PLY file")
    parser.add_argument("--prompt", type=str, default="block", help="Text prompt for YOLO-World")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_DATA_DIR / "outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB", help="MegaPose model name")
    args = parser.parse_args()
    
    run_pose_estimation(
        image_path=args.image,
        ply_path=args.ply,
        output_dir=args.output_dir,
        prompt=args.prompt,
        model_name=args.model,
    )