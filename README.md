# Traffic Monitoring System

Production-level video-based traffic monitoring with vehicle detection, tracking, ANPR (Automatic Number Plate Recognition), and helmet detection for motorcycle riders.

## Features

- **Vehicle Detection & Tracking** — YOLOv8 + DeepSORT for real-time multi-object tracking with unique IDs
- **Number Plate Recognition (ANPR)** — Multi-frame aggregation with quality scoring; only the best crop per vehicle is kept
- **Plate Preprocessing** — Resize, sharpen, CLAHE contrast enhancement, perspective correction
- **OCR Validation** — PaddleOCR with regex-based format validation; partial/guessed results marked `UNREADABLE`
- **Helmet Detection** — Motorcycle-only; detects helmet/no-helmet for riders; one final image per rider
- **Deduplication** — Vehicle-level (DeepSORT ID), plate-level (best quality), helmet-level (highest confidence)
- **Annotated Video Output** — Bounding boxes, plate text, helmet status labels
- **CSV Logs** — Plate and helmet logs with vehicle ID, text, confidence, frame number

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic (vehicle detection + tracking only)

```bash
python traffic_monitor.py --input video.mp4 --output output/
```

### Full pipeline (with custom plate and helmet models)

```bash
python traffic_monitor.py \
    --input video.mp4 \
    --output output/ \
    --plate-model plate_detector.pt \
    --helmet-model helmet_detector.pt
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | *required* | Path to input video file |
| `--output`, `-o` | *required* | Path to output directory |
| `--vehicle-model` | `yolov8n.pt` | YOLOv8 vehicle detection model |
| `--plate-model` | `None` | YOLOv8 plate detection model |
| `--helmet-model` | `None` | YOLOv8 helmet detection model |
| `--vehicle-conf` | `0.4` | Vehicle detection confidence threshold |
| `--plate-conf` | `0.3` | Plate detection confidence threshold |
| `--helmet-conf` | `0.4` | Helmet detection confidence threshold |
| `--max-age` | `30` | DeepSORT max age for tracks |

## Output Structure

```
output/
  annotated_output.mp4      # Video with bounding boxes & labels
  plate_log.csv             # Plate OCR results per vehicle
  helmet_log.csv            # Helmet detection results per motorcycle
  plate_crops/              # Best plate crop per vehicle
    vehicle_1_plate.jpg
    vehicle_2_plate.jpg
  helmet_crops/             # One crop per motorcycle rider
    vehicle_3_helmet.jpg
    vehicle_5_no_helmet.jpg
```

## Custom Models

The system uses COCO-pretrained YOLOv8 for vehicle detection out of the box. For plate and helmet detection, you need custom-trained YOLOv8 models:

- **Plate model**: Train on a license plate dataset; the model should detect plate regions within vehicle crops.
- **Helmet model**: Train with two classes — `0: helmet`, `1: no_helmet`. The system crops the upper 60% of motorcycle bounding boxes (rider area) before running this model.

## Technical Stack

| Component | Library |
|-----------|---------|
| Vehicle Detection | YOLOv8 (ultralytics) |
| Vehicle Tracking | DeepSORT (deep-sort-realtime) |
| Plate Detection | YOLOv8 (custom model) |
| OCR | PaddleOCR |
| Preprocessing | OpenCV |
| Video I/O | OpenCV |
