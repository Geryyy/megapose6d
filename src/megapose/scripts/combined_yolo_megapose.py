import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
from ultralytics import YOLO

# MegaPose imports
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
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

def run_yolo_detection(model_path: str, source: str, conf: float = 0.3) -> Tuple[np.ndarray, List[dict], str]:
    """Run YOLOv8 detection and return image, bounding boxes, and output image path."""
    try:
        # Load YOLOv8 model
        model = YOLO(model_path)
        
        # Run prediction
        results = model.predict(source=source, conf=conf, save=True, project="runs/detect", name="predict", exist_ok=True)
        
        # Load input image
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Failed to load image at {source}")
        
        # Get output image path
        output_dir = Path("runs/detect/predict")
        output_image = output_dir / Path(source).name
        
        if not output_image.exists():
            raise FileNotFoundError(f"Output image not found at {output_image}")
        
        # Extract bounding boxes, labels, and scores
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] in xyxy format
        scores = results[0].boxes.conf.cpu().numpy()  # [N]
        labels = ["legoblock"] * len(boxes)  # Single class 'block'
        
        # Create detection data
        bboxes = [
            {
                "label": label,
                "bbox_modal": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score)
            }
            for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores)
        ]
        
        return img, bboxes, str(output_image)
    except Exception as e:
        logger.error(f"Error during YOLO detection: {str(e)}")
        raise

def create_observation_tensor(rgb: np.ndarray, K: np.ndarray, load_depth: bool = False) -> ObservationTensor:
    """Create an observation tensor from RGB image and camera matrix."""
    depth = None
    observation = ObservationTensor.from_numpy(rgb, depth, K)
    return observation

def make_object_dataset(mesh_dir: Path) -> RigidObjectDataset:
    """Create a rigid object dataset from mesh directory."""
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = mesh_dir.iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"Multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"Couldn't find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    return RigidObjectDataset(rigid_objects)

def convert_bboxes_to_detections(bboxes: List[dict]) -> DetectionsType:
    """Convert YOLO bounding boxes to MegaPose detections."""
    object_data = [
        ObjectData(
            label=bbox["label"],
            bbox_modal=bbox["bbox_modal"]
        )
        for bbox in bboxes
    ]
    detections = make_detections_from_object_data(object_data).cuda()
    return detections

def run_pose_estimation(
    rgb: np.ndarray,
    bboxes: List[dict],
    camera_data: CameraData,
    model_name: str,
    mesh_dir: Path
) -> PoseEstimatesType:
    """Run MegaPose pose estimation."""
    try:
        model_info = NAMED_MODELS[model_name]
        observation = create_observation_tensor(rgb, camera_data.K, model_info["requires_depth"]).cuda()
        detections = convert_bboxes_to_detections(bboxes).cuda()
        object_dataset = make_object_dataset(mesh_dir)

        logger.info(f"Loading model {model_name}.")
        pose_estimator = load_named_model(model_name, object_dataset).cuda()

        logger.info(f"Running inference.")
        output, _ = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"]
        )
        return output
    except Exception as e:
        logger.error(f"Error during pose estimation: {str(e)}")
        raise

def save_predictions(
    output_dir: Path,
    pose_estimates: PoseEstimatesType
) -> None:
    """Save pose estimation predictions."""
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = output_dir / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")

def make_visualizations(
    rgb: np.ndarray,
    camera_data: CameraData,
    pose_estimates: PoseEstimatesType,
    mesh_dir: Path,
    output_dir: Path
) -> None:
    """Create and save visualizations."""
    try:
        object_dataset = make_object_dataset(mesh_dir)
        renderer = Panda3dSceneRenderer(object_dataset)

        # Convert pose estimates to object data
        labels = pose_estimates.infos["label"]
        poses = pose_estimates.poses.cpu().numpy()
        object_datas = [
            ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
        ]

        camera_data.TWC = Transform(np.eye(4))
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()
        fig_rgb = plotter.plot_image(rgb)
        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)
        fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)

        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
        export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
        export_png(fig_all, filename=vis_dir / "all_results.png")
        logger.info(f"Wrote visualizations to {vis_dir}.")
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        raise

def display_image(image_path: str, window_name: str = "Detection and Pose Estimation"):
    """Display an image using OpenCV."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image at {image_path}")
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")
        raise

def main(
    yolo_model_path: str,
    source: str,
    megapose_model: str,
    conf: float,
    mesh_dir: str,
):
    """Main function to run combined YOLO detection and MegaPose estimation."""
    try:
        # Set up output directory
        output_dir = Path("runs/detect/predict")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run YOLO detection
        rgb, bboxes, output_image = run_yolo_detection(yolo_model_path, source, conf)

        # Create camera data (adjust as needed for your setup)
        camera_data = CameraData(
            K=np.array([
                [800, 0, 320],
                [0, 800, 320],
                [0, 0, 1]
            ], dtype=np.float32),
            resolution=(640, 640)
        )

        # Run MegaPose pose estimation
        pose_estimates = run_pose_estimation(
            rgb, bboxes, camera_data, megapose_model, Path(mesh_dir)
        )

        # Save predictions
        save_predictions(output_dir, pose_estimates)

        # Create visualizations
        make_visualizations(rgb, camera_data, pose_estimates, Path(mesh_dir), output_dir)

        # Display result
        display_image(output_image)

        logger.info(f"Output image: {output_image}")
        logger.info(f"Results saved in: {output_dir}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser(description="Run combined YOLO detection and MegaPose pose estimation")
    parser.add_argument("--yolo-model", type=str, default="src/detection/runs/train/exp/weights/best.pt",
                        help="Path to YOLOv8 model (.pt file)")
    parser.add_argument("--source", type=str, default="image.png",
                        help="Path to input image")
    parser.add_argument("--megapose-model", type=str, default="megapose-1.0-RGB-multi-hypothesis",
                        help="MegaPose model name")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold for detection")
    parser.add_argument("--mesh-dir", type=str, default="meshes",
                        help="Path to directory containing object meshes")
    args = parser.parse_args()

    main(
        yolo_model_path=args.yolo_model,
        source=args.source,
        megapose_model=args.megapose_model,
        conf=args.conf,
        mesh_dir=args.mesh_dir,
    )