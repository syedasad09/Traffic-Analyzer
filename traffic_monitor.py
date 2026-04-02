#!/usr/bin/env python3
"""
Production-Level Traffic Monitoring System
==========================================
Video-based traffic monitoring with:
- Vehicle detection and tracking (YOLOv8 + DeepSORT)
- Automatic Number Plate Recognition (YOLOv8 + PaddleOCR)
- Helmet detection for motorcycle riders (YOLOv8)
- Multi-frame aggregation and deduplication
- Annotated video output with CSV logs

Usage:
    python traffic_monitor.py --input video.mp4 --output output_dir/
    python traffic_monitor.py --input video.mp4 --output output_dir/ \\
        --plate-model plate_det.pt --helmet-model helmet_det.pt

Requirements:
    pip install -r requirements.txt
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# COCO class IDs for vehicles
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

MOTORCYCLE_CLASS_ID = 3

# Common license-plate regex patterns (extend as needed)
PLATE_PATTERNS: List[str] = [
    r"^[A-Z]{2,3}[\s\-]?\d{1,4}[\s\-]?[A-Z]{0,3}$",
    r"^[A-Z0-9]{5,10}$",
    r"^\d{1,4}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4}$",
    r"^[A-Z]{1,3}[\s\-]?\d{1,4}[\s\-]?\d{1,4}$",
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class PlateDetection:
    """A single plate detection with quality metrics."""

    vehicle_id: int
    frame_number: int
    crop: np.ndarray
    text: str
    confidence: float
    sharpness: float
    resolution: Tuple[int, int]
    quality_score: float
    is_valid: bool


@dataclass
class HelmetDetection:
    """Helmet detection for a motorcycle rider."""

    vehicle_id: int
    frame_number: int
    crop: np.ndarray
    has_helmet: bool
    confidence: float


@dataclass
class TrackedVehicle:
    """Aggregated data for a tracked vehicle across all frames."""

    vehicle_id: int
    vehicle_class: str
    plate_detections: List[PlateDetection] = field(default_factory=list)
    helmet_detections: List[HelmetDetection] = field(default_factory=list)
    best_plate: Optional[PlateDetection] = None
    final_helmet: Optional[HelmetDetection] = None
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    frame_count: int = 0


# ---------------------------------------------------------------------------
# Plate Quality Scoring
# ---------------------------------------------------------------------------
class PlateQualityScorer:
    """Scores plate crops to select the best one across frames."""

    @staticmethod
    def compute_sharpness(image: np.ndarray) -> float:
        """Laplacian variance as a sharpness proxy."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_resolution_score(image: np.ndarray) -> float:
        h, w = image.shape[:2]
        area = h * w
        if area >= 10000:
            return 1.0
        if area >= 5000:
            return 0.8
        if area >= 2000:
            return 0.5
        return 0.2

    @staticmethod
    def validate_plate_text(text: str) -> bool:
        """Return True only if text matches a known plate format."""
        if not text or len(text) < 3:
            return False
        cleaned = re.sub(r"[\s\-]", "", text).upper()
        return any(re.match(p, cleaned) for p in PLATE_PATTERNS)

    @staticmethod
    def compute_text_completeness(text: str) -> float:
        if not text:
            return 0.0
        alnum_count = sum(c.isalnum() for c in text)
        if alnum_count >= 6:
            return 1.0
        if alnum_count >= 4:
            return 0.7
        if alnum_count >= 2:
            return 0.4
        return 0.1

    @classmethod
    def score(cls, image: np.ndarray, ocr_text: str, ocr_confidence: float) -> float:
        """Weighted quality score combining OCR confidence, completeness,
        sharpness, and resolution."""
        sharpness_norm = min(cls.compute_sharpness(image) / 500.0, 1.0)
        resolution = cls.compute_resolution_score(image)
        completeness = cls.compute_text_completeness(ocr_text)
        quality = (
            0.35 * ocr_confidence
            + 0.25 * completeness
            + 0.20 * sharpness_norm
            + 0.20 * resolution
        )
        return round(quality, 4)


# ---------------------------------------------------------------------------
# Plate Pre-processing
# ---------------------------------------------------------------------------
class PlatePreprocessor:
    """Resize, sharpen, enhance contrast, and apply perspective correction."""

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h < 10 or w < 10:
            return image

        # Resize to standard height
        target_h = 64
        scale = target_h / h
        resized = cv2.resize(
            image, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC
        )

        # Grayscale
        gray = (
            cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if len(resized.shape) == 3
            else resized
        )

        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)

        # Perspective correction
        enhanced = PlatePreprocessor._perspective_correct(enhanced)

        # Back to BGR for PaddleOCR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _perspective_correct(gray: np.ndarray) -> np.ndarray:
        """Attempt four-point perspective correction via contour detection."""
        try:
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return gray

            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                s = pts.sum(axis=1)
                d = np.diff(pts, axis=1).ravel()
                rect = np.array(
                    [
                        pts[np.argmin(s)],
                        pts[np.argmin(d)],
                        pts[np.argmax(s)],
                        pts[np.argmax(d)],
                    ],
                    dtype=np.float32,
                )
                h, w = gray.shape[:2]
                dst = np.array(
                    [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
                )
                matrix = cv2.getPerspectiveTransform(rect, dst)
                return cv2.warpPerspective(gray, matrix, (w, h))
        except Exception:
            pass
        return gray


# ---------------------------------------------------------------------------
# Main System
# ---------------------------------------------------------------------------
class TrafficMonitoringSystem:
    """Orchestrates vehicle detection, tracking, ANPR, and helmet detection."""

    def __init__(
        self,
        vehicle_model_path: str = "yolov8n.pt",
        plate_model_path: Optional[str] = None,
        helmet_model_path: Optional[str] = None,
        vehicle_conf: float = 0.4,
        plate_conf: float = 0.3,
        helmet_conf: float = 0.4,
        max_age: int = 30,
    ):
        logger.info("Initializing Traffic Monitoring System ...")

        # --- Models -----------------------------------------------------------
        self.vehicle_model = YOLO(vehicle_model_path)
        logger.info("Vehicle model loaded: %s", vehicle_model_path)

        self.plate_model: Optional[YOLO] = None
        if plate_model_path and os.path.exists(plate_model_path):
            self.plate_model = YOLO(plate_model_path)
            logger.info("Plate model loaded: %s", plate_model_path)
        else:
            logger.warning(
                "No plate detection model provided — plate detection disabled."
            )

        self.helmet_model: Optional[YOLO] = None
        if helmet_model_path and os.path.exists(helmet_model_path):
            self.helmet_model = YOLO(helmet_model_path)
            logger.info("Helmet model loaded: %s", helmet_model_path)
        else:
            logger.warning(
                "No helmet detection model provided — helmet detection disabled."
            )

        # --- OCR --------------------------------------------------------------
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        logger.info("PaddleOCR initialized")

        # --- Tracker ----------------------------------------------------------
        self.tracker = DeepSort(max_age=max_age)
        logger.info("DeepSORT tracker initialized")

        # --- Thresholds -------------------------------------------------------
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        self.helmet_conf = helmet_conf

        # --- State ------------------------------------------------------------
        self.vehicles: Dict[int, TrackedVehicle] = {}
        self.preprocessor = PlatePreprocessor()
        self.scorer = PlateQualityScorer()

    # --------------------------------------------------------------------- #
    # Vehicle Detection & Tracking
    # --------------------------------------------------------------------- #
    def _detect_vehicles(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        """Return list of (bbox, confidence, class_id) for vehicles."""
        results = self.vehicle_model(frame, conf=self.vehicle_conf, verbose=False)[0]
        detections: List[Tuple[Tuple[int, int, int, int], float, int]] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append(((x1, y1, x2, y2), conf, cls_id))
        return detections

    def _track_vehicles(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float, int]],
        frame: np.ndarray,
    ) -> List[Tuple[int, Tuple[int, int, int, int], int]]:
        """Track vehicles with DeepSORT. Returns (track_id, bbox, class_id)."""
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        ds_dets = []
        for bbox, conf, cls_id in detections:
            x1, y1, x2, y2 = bbox
            ds_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        tracks = self.tracker.update_tracks(ds_dets, frame=frame)

        tracked: List[Tuple[int, Tuple[int, int, int, int], int]] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = (int(v) for v in ltrb)
            cls_id = track.det_class if track.det_class is not None else 2
            tracked.append((tid, (x1, y1, x2, y2), cls_id))
        return tracked

    # --------------------------------------------------------------------- #
    # Plate Detection & OCR
    # --------------------------------------------------------------------- #
    def _detect_plates(
        self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect plates inside a vehicle bbox. Returns (crop, abs_bbox)."""
        if self.plate_model is None:
            return []

        x1, y1, x2, y2 = vehicle_bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return []

        vehicle_crop = frame[y1:y2, x1:x2]
        results = self.plate_model(vehicle_crop, conf=self.plate_conf, verbose=False)[0]

        plates: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        for box in results.boxes:
            px1, py1, px2, py2 = (int(v) for v in box.xyxy[0].tolist())
            abs_x1, abs_y1 = x1 + px1, y1 + py1
            abs_x2, abs_y2 = x1 + px2, y1 + py2
            crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
            if crop.size > 0:
                plates.append((crop, (abs_x1, abs_y1, abs_x2, abs_y2)))
        return plates

    def _run_ocr(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """Run PaddleOCR on a preprocessed plate crop."""
        preprocessed = self.preprocessor.preprocess(plate_crop)
        try:
            results = self.ocr.ocr(preprocessed, cls=True)
            if not results or not results[0]:
                return "", 0.0

            texts: List[str] = []
            confidences: List[float] = []
            for line in results[0]:
                texts.append(line[1][0])
                confidences.append(line[1][1])

            full_text = " ".join(texts).strip().upper()
            full_text = re.sub(r"[^A-Z0-9\s\-]", "", full_text)
            avg_conf = sum(confidences) / len(confidences)
            return full_text, avg_conf
        except Exception as exc:
            logger.debug("OCR error: %s", exc)
            return "", 0.0

    def _process_plate(
        self, vehicle_id: int, frame_number: int, plate_crop: np.ndarray
    ) -> PlateDetection:
        """OCR a plate crop, validate, and score quality."""
        text, confidence = self._run_ocr(plate_crop)
        sharpness = self.scorer.compute_sharpness(plate_crop)
        resolution = (plate_crop.shape[0], plate_crop.shape[1])
        quality_score = self.scorer.score(plate_crop, text, confidence)
        is_valid = self.scorer.validate_plate_text(text) and confidence >= 0.6

        if not is_valid:
            text = "UNREADABLE"

        return PlateDetection(
            vehicle_id=vehicle_id,
            frame_number=frame_number,
            crop=plate_crop.copy(),
            text=text,
            confidence=confidence,
            sharpness=sharpness,
            resolution=resolution,
            quality_score=quality_score,
            is_valid=is_valid,
        )

    # --------------------------------------------------------------------- #
    # Helmet Detection
    # --------------------------------------------------------------------- #
    def _detect_helmets(
        self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]
    ) -> List[Tuple[np.ndarray, bool, float]]:
        """Detect helmets in the rider area (upper 60 %) of a motorcycle bbox."""
        if self.helmet_model is None:
            return []

        x1, y1, x2, y2 = vehicle_bbox
        fh, fw = frame.shape[:2]
        rider_y2 = y1 + int((y2 - y1) * 0.6)
        x1, y1 = max(0, x1), max(0, y1)
        x2, rider_y2 = min(fw, x2), min(fh, rider_y2)
        if x2 - x1 < 10 or rider_y2 - y1 < 10:
            return []

        rider_crop = frame[y1:rider_y2, x1:x2]
        results = self.helmet_model(rider_crop, conf=self.helmet_conf, verbose=False)[0]

        helmet_results: List[Tuple[np.ndarray, bool, float]] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            hx1, hy1, hx2, hy2 = (int(v) for v in box.xyxy[0].tolist())
            crop = rider_crop[hy1:hy2, hx1:hx2]
            if crop.size > 0:
                has_helmet = cls_id == 0  # convention: 0=helmet, 1=no_helmet
                helmet_results.append((crop.copy(), has_helmet, conf))
        return helmet_results

    # --------------------------------------------------------------------- #
    # Vehicle State Management & Deduplication
    # --------------------------------------------------------------------- #
    def _get_or_create_vehicle(
        self, vehicle_id: int, cls_id: int, bbox: Tuple[int, int, int, int]
    ) -> TrackedVehicle:
        if vehicle_id not in self.vehicles:
            self.vehicles[vehicle_id] = TrackedVehicle(
                vehicle_id=vehicle_id,
                vehicle_class=VEHICLE_CLASSES.get(cls_id, "unknown"),
            )
        vehicle = self.vehicles[vehicle_id]
        vehicle.last_bbox = bbox
        vehicle.frame_count += 1
        return vehicle

    @staticmethod
    def _update_best_plate(vehicle: TrackedVehicle) -> None:
        """Keep only the highest-quality plate (prefer valid ones)."""
        valid = [p for p in vehicle.plate_detections if p.is_valid]
        candidates = valid if valid else vehicle.plate_detections
        if candidates:
            vehicle.best_plate = max(candidates, key=lambda p: p.quality_score)

    @staticmethod
    def _update_final_helmet(vehicle: TrackedVehicle) -> None:
        """Keep only the highest-confidence helmet detection."""
        if vehicle.helmet_detections:
            vehicle.final_helmet = max(
                vehicle.helmet_detections, key=lambda h: h.confidence
            )

    # --------------------------------------------------------------------- #
    # Annotation Drawing
    # --------------------------------------------------------------------- #
    def _draw_annotations(
        self,
        frame: np.ndarray,
        tracked: List[Tuple[int, Tuple[int, int, int, int], int]],
    ) -> np.ndarray:
        annotated = frame.copy()

        for track_id, bbox, cls_id in tracked:
            x1, y1, x2, y2 = bbox
            vehicle = self.vehicles.get(track_id)

            # Box colour
            color = (0, 165, 255) if cls_id == MOTORCYCLE_CLASS_ID else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            cls_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
            label = f"ID:{track_id} {cls_name}"

            # Plate text
            if vehicle and vehicle.best_plate:
                label += f" | {vehicle.best_plate.text}"

            # Helmet status (motorcycles)
            if (
                vehicle
                and cls_id == MOTORCYCLE_CLASS_ID
                and vehicle.final_helmet is not None
            ):
                status = "Helmet" if vehicle.final_helmet.has_helmet else "No Helmet"
                h_color = (
                    (0, 255, 0) if vehicle.final_helmet.has_helmet else (0, 0, 255)
                )
                label += f" | {status}"
                cv2.putText(
                    annotated,
                    status,
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    h_color,
                    2,
                )

            # Label background
            font_scale, thickness = 0.7, 2
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return annotated

    # --------------------------------------------------------------------- #
    # Result Saving
    # --------------------------------------------------------------------- #
    def _save_results(self, output_dir: Path) -> None:
        plates_dir = output_dir / "plate_crops"
        helmets_dir = output_dir / "helmet_crops"
        plates_dir.mkdir(parents=True, exist_ok=True)
        helmets_dir.mkdir(parents=True, exist_ok=True)

        # --- Plate CSV -------------------------------------------------------
        plate_csv = output_dir / "plate_log.csv"
        with open(plate_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "vehicle_id",
                    "vehicle_class",
                    "plate_text",
                    "confidence",
                    "quality_score",
                    "sharpness",
                    "resolution",
                    "frame_number",
                    "is_valid",
                ]
            )
            for vid, vehicle in sorted(self.vehicles.items()):
                plate = vehicle.best_plate
                if plate is None:
                    continue
                crop_path = plates_dir / f"vehicle_{vid}_plate.jpg"
                cv2.imwrite(str(crop_path), plate.crop)
                writer.writerow(
                    [
                        vid,
                        vehicle.vehicle_class,
                        plate.text,
                        f"{plate.confidence:.4f}",
                        f"{plate.quality_score:.4f}",
                        f"{plate.sharpness:.2f}",
                        f"{plate.resolution[1]}x{plate.resolution[0]}",
                        plate.frame_number,
                        plate.is_valid,
                    ]
                )
        logger.info("Plate log saved: %s", plate_csv)

        # --- Helmet CSV ------------------------------------------------------
        helmet_csv = output_dir / "helmet_log.csv"
        with open(helmet_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["vehicle_id", "has_helmet", "confidence", "frame_number"]
            )
            for vid, vehicle in sorted(self.vehicles.items()):
                if vehicle.vehicle_class != "motorcycle":
                    continue
                helmet = vehicle.final_helmet
                if helmet is None:
                    continue
                status_tag = "helmet" if helmet.has_helmet else "no_helmet"
                crop_path = helmets_dir / f"vehicle_{vid}_{status_tag}.jpg"
                cv2.imwrite(str(crop_path), helmet.crop)
                writer.writerow(
                    [vid, helmet.has_helmet, f"{helmet.confidence:.4f}", helmet.frame_number]
                )
        logger.info("Helmet log saved: %s", helmet_csv)

    # --------------------------------------------------------------------- #
    # Main Processing Loop
    # --------------------------------------------------------------------- #
    def process_video(self, input_path: str, output_dir: str) -> None:
        """Process a video file end-to-end."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", input_path)
            sys.exit(1)

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Input: %s (%dx%d @ %d fps, %d frames)",
            input_path, width, height, fps, total_frames,
        )

        out_video = out / "annotated_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vwriter = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

        # Process plates/helmets at ~10 fps for efficiency
        process_every_n = max(1, fps // 10)
        frame_num = 0
        t0 = time.time()

        logger.info("Processing video ...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            # 1. Detect vehicles
            detections = self._detect_vehicles(frame)

            # 2. Track vehicles
            tracked = self._track_vehicles(detections, frame)

            # 3. Per-vehicle processing
            for tid, bbox, cls_id in tracked:
                vehicle = self._get_or_create_vehicle(tid, cls_id, bbox)

                if frame_num % process_every_n != 0:
                    continue

                # 4. Plate detection + OCR (all vehicles)
                for plate_crop, _plate_bbox in self._detect_plates(frame, bbox):
                    det = self._process_plate(tid, frame_num, plate_crop)
                    vehicle.plate_detections.append(det)
                    self._update_best_plate(vehicle)

                # 5. Helmet detection (motorcycles — mandatory)
                if cls_id == MOTORCYCLE_CLASS_ID:
                    for crop, has_helmet, conf in self._detect_helmets(frame, bbox):
                        hd = HelmetDetection(
                            vehicle_id=tid,
                            frame_number=frame_num,
                            crop=crop,
                            has_helmet=has_helmet,
                            confidence=conf,
                        )
                        vehicle.helmet_detections.append(hd)
                        self._update_final_helmet(vehicle)

            # 6. Draw annotations & write frame
            annotated = self._draw_annotations(frame, tracked)
            vwriter.write(annotated)

            # Progress log every 5 seconds of video
            if frame_num % (fps * 5) == 0 or frame_num == total_frames:
                elapsed = time.time() - t0
                pct = (frame_num / total_frames * 100) if total_frames else 0
                logger.info(
                    "Frame %d/%d (%.1f%%) | Vehicles: %d | Elapsed: %.1fs",
                    frame_num, total_frames, pct, len(self.vehicles), elapsed,
                )

        cap.release()
        vwriter.release()

        elapsed = time.time() - t0
        logger.info("Video processing complete in %.1fs", elapsed)
        logger.info("Annotated video: %s", out_video)

        # 7. Save deduplicated results
        self._save_results(out)

        # 8. Summary
        total_v = len(self.vehicles)
        motos = sum(1 for v in self.vehicles.values() if v.vehicle_class == "motorcycle")
        plates_ok = sum(
            1 for v in self.vehicles.values() if v.best_plate and v.best_plate.is_valid
        )
        helmets_ok = sum(1 for v in self.vehicles.values() if v.final_helmet is not None)

        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("  Total vehicles tracked : %d", total_v)
        logger.info("  Motorcycles            : %d", motos)
        logger.info("  Plates read            : %d / %d", plates_ok, total_v)
        logger.info("  Helmet detections      : %d / %d", helmets_ok, motos)
        logger.info("  Output directory       : %s", out)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-Level Traffic Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--vehicle-model",
        default="yolov8n.pt",
        help="YOLOv8 vehicle detection model (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--plate-model", default=None, help="YOLOv8 plate detection model path"
    )
    parser.add_argument(
        "--helmet-model", default=None, help="YOLOv8 helmet detection model path"
    )
    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=0.4,
        help="Vehicle detection confidence (default: 0.4)",
    )
    parser.add_argument(
        "--plate-conf",
        type=float,
        default=0.3,
        help="Plate detection confidence (default: 0.3)",
    )
    parser.add_argument(
        "--helmet-conf",
        type=float,
        default=0.4,
        help="Helmet detection confidence (default: 0.4)",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="DeepSORT max age for tracks (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        logger.error("Input video not found: %s", args.input)
        sys.exit(1)

    system = TrafficMonitoringSystem(
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        helmet_model_path=args.helmet_model,
        vehicle_conf=args.vehicle_conf,
        plate_conf=args.plate_conf,
        helmet_conf=args.helmet_conf,
        max_age=args.max_age,
    )
    system.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
