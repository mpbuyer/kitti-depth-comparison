#!/usr/bin/env python3
"""
Compute MAE table for KITTI depth estimation methods vs LiDAR.
"""

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

from src.calibration import Calibration
from src.data_loader import KITTIDataLoader
from src.depth_estimation import (
    RAFTStereoEstimator,
    StereoDepthEstimator,
    DepthAnythingV2Estimator,
    UniDepthEstimator,
)
from src.tracklet_processing import TrackletProcessor, points_in_bbox
from src.utils import calc_distances, get_best_distance
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate MAE table for depth estimation methods on KITTI"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2011_09_26",
        help="KITTI date folder to scan (default: 2011_09_26)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Explicit KITTI sequences to process (overrides --date scan)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["stereo", "raftstereo", "depthanything", "unidepth"],
        choices=["stereo", "raftstereo", "depthanything", "depthanything2", "unidepth"],
        help="Depth methods to compare",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--use_distance",
        action="store_true",
        help="Use XY-plane distance instead of forward (x) depth",
    )
    parser.add_argument(
        "--download_missing",
        action="store_true",
        help="Download missing sequences if not found locally",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path (default: ./mae_table.csv)",
    )
    return parser.parse_args()


def discover_sequences(date_dir):
    sync_dirs = sorted(glob.glob(os.path.join(date_dir, "*_sync")))
    return [os.path.basename(d).replace("_sync", "") for d in sync_dirs]


def get_depth_estimator(method, config, calibration, cam):
    if method == "stereo":
        return StereoDepthEstimator(config["stereo"], calibration)
    if method == "raftstereo":
        return RAFTStereoEstimator(config["raftstereo"], calibration)
    if method in {"depthanything", "depthanything2"}:
        return DepthAnythingV2Estimator(config["depthanything2"])
    if method == "unidepth":
        return UniDepthEstimator(config["unidepth"], calibration.cam_projections[cam])
    raise ValueError(f"Unknown method: {method}")


def compute_measurement(
    points,
    corners_3d,
    use_distance,
    technique,
    clip_distance,
):
    if corners_3d is None or len(points) == 0:
        return None

    filtered_points = points[points_in_bbox(points, corners_3d)]
    if len(filtered_points) == 0:
        return None

    # Align with LiDAR clip distance behavior
    filtered_points = filtered_points[filtered_points[:, 0] > clip_distance]
    if len(filtered_points) == 0:
        return None

    ranges = calc_distances(filtered_points) if use_distance else filtered_points[:, 0]
    measurement = get_best_distance(ranges, technique=technique)
    if not np.isfinite(measurement) or measurement <= 0:
        return None
    return measurement


def compute_lidar_measurement(
    lidar_points,
    calibration,
    cam,
    corners_3d,
    use_distance,
    technique,
    clip_distance,
    img_width,
    img_height,
):
    if corners_3d is None or len(lidar_points) == 0:
        return None

    filtered_points = lidar_points[points_in_bbox(lidar_points, corners_3d)]
    if len(filtered_points) == 0:
        return None

    filtered_points, _, _, _ = calibration.get_lidar_in_image_fov(
        filtered_points,
        cam,
        0,
        0,
        img_width - 1,
        img_height - 1,
        return_more=True,
        clip_distance=clip_distance,
    )
    if len(filtered_points) == 0:
        return None

    ranges = calc_distances(filtered_points) if use_distance else filtered_points[:, 0]
    measurement = get_best_distance(ranges, technique=technique)
    if not np.isfinite(measurement) or measurement <= 0:
        return None
    return measurement


def process_sequence(method, sequence, config, use_distance, download_missing):
    cam = config["processing"]["camera"]
    technique = config["processing"]["depth_technique"]
    clip_distance = config["visualization"]["clip_distance"]

    data_loader = KITTIDataLoader(sequence)

    if data_loader.sequence_dir.exists():
        data_loader._load_file_paths()
        data_loader._validate_data()
    elif download_missing:
        data_loader.download_and_extract()
    else:
        print(f"Skipping {sequence}: sequence directory not found")
        return 0.0, 0

    calibration = Calibration(
        data_loader.cam_to_cam_file,
        data_loader.velo_to_cam_file,
    )
    tracklet_processor = TrackletProcessor(data_loader.tracklet_file)
    depth_estimator = get_depth_estimator(method, config, calibration, cam)

    total_abs_error = 0.0
    total_count = 0

    for frame_idx in range(len(data_loader.left_image_files)):
        try:
            if method == "stereo":
                depth_map = depth_estimator.compute_depth(
                    data_loader.left_image_files[frame_idx],
                    data_loader.right_image_files[frame_idx],
                )
            elif method == "raftstereo":
                depth_map = depth_estimator.compute_depth(
                    data_loader.left_image_files[frame_idx],
                    data_loader.right_image_files[frame_idx],
                )
            elif method in {"depthanything", "depthanything2"}:
                img = cv2.imread(data_loader.left_image_files[frame_idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                depth_map = depth_estimator.compute_depth(img)
            elif method == "unidepth":
                depth_map = depth_estimator.compute_depth(
                    data_loader.left_image_files[frame_idx]
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            lidar_points = data_loader.load_velodyne_points(
                data_loader.point_files[frame_idx]
            )[:, :3]

            # Use image dimensions for LiDAR FOV clipping
            img = cv2.imread(data_loader.left_image_files[frame_idx])
            img_height, img_width = img.shape[:2]

            projected_points = calibration.depth_to_lidar_points(depth_map, cam)

            tracklets = tracklet_processor.get_tracklets_for_frame(frame_idx)
            for _, tracklet_data in tracklets:
                corners_3d = tracklet_data["corners_3d"]

                lidar_measure = compute_lidar_measurement(
                    lidar_points,
                    calibration,
                    cam,
                    corners_3d,
                    use_distance,
                    technique,
                    clip_distance,
                    img_width,
                    img_height,
                )
                if lidar_measure is None:
                    continue

                cam_measure = compute_measurement(
                    projected_points,
                    corners_3d,
                    use_distance,
                    technique,
                    clip_distance,
                )
                if cam_measure is None:
                    continue

                total_abs_error += abs(lidar_measure - cam_measure)
                total_count += 1

        except Exception as e:
            print(f"Warning: {sequence} frame {frame_idx} failed: {e}")
            continue

    mae = total_abs_error / total_count if total_count > 0 else 0.0
    return mae, total_count


def main():
    args = parse_arguments()
    config = load_config(args.config)

    if args.sequences:
        sequences = args.sequences
    else:
        sequences = discover_sequences(args.date)
        if not sequences:
            print(f"No sequences found in {args.date}")
            sys.exit(1)

    output_csv = args.output_csv or "mae_table.csv"

    results = {}

    start_time = time.time()
    for sequence in sequences:
        results[sequence] = {}
        for method in args.methods:
            print(f"Processing {sequence} with {method}...")
            mae, count = process_sequence(
                method,
                sequence,
                config,
                args.use_distance,
                args.download_missing,
            )
            results[sequence][method] = {"mae": mae, "count": count}

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sequence"] + args.methods
        writer.writerow(header)
        for sequence in sequences:
            row = [sequence]
            for method in args.methods:
                entry = results.get(sequence, {}).get(method)
                row.append("" if entry is None else f"{entry['mae']:.4f}")
            writer.writerow(row)

    print("\nMAE Table")
    print("=" * 60)
    for sequence in sequences:
        for method in args.methods:
            entry = results.get(sequence, {}).get(method, {"mae": 0.0, "count": 0})
            print(
                f"{sequence} | {method:<14} | MAE {entry['mae']:.4f} | n={entry['count']}"
            )
    print("=" * 60)
    print(f"Saved CSV: {output_csv}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
