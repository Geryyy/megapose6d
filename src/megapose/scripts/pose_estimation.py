import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
from ultralytics import YOLO
import pandas as pd

from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType, PandasTensorCollection
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

def detect_objects_yolov8(
    image: np.ndarray,
    model_path: str,
    confidence_threshold: float = 0.3,
) -> DetectionsType:
    """Detect objects using YOLOv8 model."""
    # Convert image to RGB (YOLOv8 expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load YOLOv8 model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(image_rgb, conf=confidence_threshold, verbose=False)
    
    # Extract detections
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] in xyxy format
    scores = results[0].boxes.conf.cpu().numpy()  # [N]
    labels = ["block"] * len(boxes)  # Single class 'block'
    
    # Convert to MegaPose DetectionsType (PandasTensorCollection)
    bboxes = torch.tensor(boxes, dtype=torch.float32).cuda()
    infos = pd.DataFrame({
        "batch_im_id": [0] * len(boxes),  # Single image, so batch_im_id = 0
        "label": labels,
        "score": scores
    })
    
    detections = PandasTensorCollection(infos=infos, bboxes=bboxes)
    return detections

def load_observation(
    image_path: Path,
    camera_data: CameraData = None,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    """Load an RGB image and optional camera data."""
    rgb = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
    
    if camera_data is None:
        camera_data = CameraData(
            resolution=rgb.shape[:2],
            K=np.array([[800, 0, rgb.shape[1]/2],
                       [0, 800, rgb.shape[0]/2],
                       [0, 0, 1]], dtype=np.float32),
        )
    
    depth = None
    if load_depth:
        raise NotImplementedError("Depth loading not implemented")
    
    return rgb, depth, camera_data

def load_observation_tensor(
    image_path: Path,
    camera_data: CameraData = None,
    load_depth: bool = False,
) -> ObservationTensor:
    """Convert image to ObservationTensor for MegaPose."""
    rgb, depth, camera_data = load_observation(image_path, camera_data, load_depth)
    
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_gray = cv2.cvtColor(rgb_gray, cv2.COLOR_GRAY2RGB)
    
    observation = ObservationTensor.from_numpy(rgb_gray, depth, camera_data.K)
    return observation

def make_object_dataset(ply_path: Path) -> RigidObjectDataset:
    """Create a RigidObjectDataset from a PLY file."""
    label = "block"
    mesh_units = "mm"
    rigid_objects = [RigidObject(label=label, mesh_path=ply_path, mesh_units=mesh_units)]
    return RigidObjectDataset(rigid_objects)

def save_predictions(
    output_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    """Save estimated poses to JSON."""
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
    """Generate lightweight visualizations using OpenCV."""
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
    
    # Create visualizations directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # 1. Detections visualization
    img_detections = rgb.copy()
    for bbox, info in zip(detections.tensors["bboxes"].cpu().numpy(), detections.infos.itertuples()):
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{info.label} ({info.score:.2f})"
        cv2.rectangle(img_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_detections, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(str(vis_dir / "detections.png"), cv2.cvtColor(img_detections, cv2.COLOR_RGB2BGR))
    
    # 2. Pose overlay visualization
    img_pose = cv2.addWeighted(rgb, 0.5, renderings.rgb, 0.5, 0.0)
    cv2.imwrite(str(vis_dir / "pose_overlay.png"), cv2.cvtColor(img_pose, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Wrote visualizations to {vis_dir}")


def run_pose_estimation(
    image_path: Path,
    ply_path: Path,
    output_dir: Path,
    model_path: str,
    model_name: str = "megapose-1.0-RGB",
    camera_data: CameraData = None,
) -> None:
    """Run 6D pose estimation using YOLOv8 for detection and MegaPose for pose estimation."""
    # Load image
    rgb, _, camera_data = load_observation(image_path, camera_data)
    
    # Detect objects
    logger.info("Running YOLOv8 detection")
    detections = detect_objects_yolov8(rgb, model_path)
    
    # Load observation tensor
    observation = load_observation_tensor(image_path, camera_data, load_depth=False).cuda()
    
    # Create object dataset
    object_dataset = make_object_dataset(ply_path)
    
    # Load MegaPose model
    logger.info(f"Loading MegaPose model: {model_name}")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()
    
    # Run inference
    logger.info("Running MegaPose inference")
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
    parser = argparse.ArgumentParser(description="6D Pose Estimation with YOLOv8 and MegaPose")
    parser.add_argument("--image", type=Path, required=True, default="/home/geraldebmer/Documents/Megapose/examples/legoblock/image.png", help="Path to input RGB image")
    parser.add_argument("--ply", type=Path, required=True, default="/home/geraldebmer/Documents/Megapose/examples/legoblock/block.ply", help="Path to PLY file")
    parser.add_argument("--model-path", type=str, default="/home/geraldebmer/repos/megapose6d/src/detection/runs/train/exp/fold_4/weights/best.pt", 
                        help="Path to YOLOv8 model")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_DATA_DIR / "outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB", help="MegaPose model name")
    args = parser.parse_args()
    
    run_pose_estimation(
        image_path=args.image,
        ply_path=args.ply,
        output_dir=args.output_dir,
        model_path=args.model_path,
        model_name=args.model,
    )