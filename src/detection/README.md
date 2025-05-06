# Block detection with yolo

## Fine-Tune

fine tune yolov8
```bash
python finetune_yolov8.py --model yolov8s.pt --data /home/geraldebmer/Pictures/dataset/data.yaml --epochs 100 --imgsz 640 --batch 8 --device 0 --n_folds 5 --split_ratio 0.8 --fliplr 0.5 --flipud 0.5 --degrees 45.0 --hsv_v 0.4 --mosaic 0.8 --scale 0.2 --cumulative
```

no cumulative learning
```bash
python finetune_yolov8.py --model yolov8s.pt --data /home/geraldebmer/Pictures/dataset/data.yaml --epochs 100 --imgsz 640 --batch 8 --device 0 --n_folds 5 --split_ratio 0.8 --fliplr 0.5 --flipud 0.5 --degrees 45.0 --hsv_v 0.4 --mosaic 0.8 --scale 0.2
```

## Test Model
```bash
yolo detect predict model=runs/train/exp/weights/best.pt source=/home/geraldebmer/Documents/Megapose/examples/legoblock/image.png save=True conf=0.3
```
### For the specified image:
```bash
cd /home/geraldebmer/repos/megapose6d/src/detection
python yolo_detect_and_display.py --model src/detection/runs/train/exp/weights/best.pt --source /home/geraldebmer/Documents/Megapose/examples/legoblock/image.png --conf 0.3 --save
```
### For a random dataset image:
```bash
python yolo_detect_and_display.py --model src/detection/runs/train/exp/weights/best.pt --dataset-dir /home/geraldebmer/Pictures/dataset --conf 0.3 --save --random
```

# Megapose

```bash
python pose_estimation.py --image /home/geraldebmer/Documents/Megapose/examples/legoblock/image.png --ply /home/geraldebmer/Documents/Megapose/examples/legoblock/block.ply --model-path /home/geraldebmer/repos/megapose6d/src/detection/runs/train/exp/fold_4/weights/best.pt --output-dir outputs
```

# Combined

```bash
python combined_yolo_megapose.py --yolo-model /home/geraldebmer/repos/megapose6d/src/weights/legoblocks.pt --source /home/geraldebmer/Documents/Megapose/examples/legoblock/image_rgb.jpg --mesh-dir /home/geraldebmer/Documents/Megapose/examples/legoblock/meshes --conf 0.3
```