#!/usr/bin/env python3
"""
KITTI Depth Comparison - Main Script
Compare LiDAR depth with Stereo Matching or deep learning methods
"""

import argparse
from argparse import Namespace
import os
import sys
import yaml
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from src.data_loader import KITTIDataLoader
from src.calibration import Calibration
from src.depth_estimation import *
from src.tracklet_processing import TrackletProcessor
from src.visualization import Visualizer
from src.utils import setup_output_dir

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compare LiDAR depth with Stereo or Monocular Estimation on KITTI dataset')
    
    parser.add_argument('--sequence',type=str,required=True,
                        help='KITTI sequence to process (e.g., 2011_09_26_drive_0048)')

    parser.add_argument('--method',type=str,choices=['stereo', 'depthanything2', 'unidepth', 'metric3d', 'raftstereo'],required=True,
                        help='Depth estimation method to compare with LiDAR')
    
    parser.add_argument('--config',type=str,default='config.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--fps',type=int, default=10, help='Video fps between 1 and 20')

    parser.add_argument('--use_distance', type=bool, default=False, help=
                        'Put True if you want xy-plane distance or keep False for just forward (x-direction) depth')

    return parser.parse_args()

def create_comparison_frame(img_lidar, img_comparison, lidar_title, method_name, target_size=(1500, 1000)):

    fig, axes = plt.subplots(2, 1, figsize=(15,10))
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, hspace=0.1)

    # Top image: LiDAR
    axes[0].imshow(img_lidar)
    axes[0].set_title(lidar_title, fontsize=20)
    axes[0].axis('off')

    # Bottom image: Comparison method
    axes[1].imshow(img_comparison)
    axes[1].set_title(method_name, fontsize=20)
    axes[1].axis('off')

    # Convert to numpy array
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape((height, width, 4))
    img = img[:, :, :3]  # Remove alpha channel, keep RGB

    if img.shape[:2] != (target_size[1], target_size[0]):
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    plt.close(fig)
    return img

def main():
    args = parse_arguments()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)

    output_dir = setup_output_dir(config['video']['output_dir'])

    print(f"\n{'='*60}")
    print(f"KITTI Depth Comparison")
    print(f"Sequence: {args.sequence}")
    print(f"Method: {args.method.upper()}")
    print(f"{'='*60}\n")

    try:
        data_loader = KITTIDataLoader(args.sequence)
        data_loader.download_and_extract()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    try:
        calibration = Calibration(
            data_loader.cam_to_cam_file,
            data_loader.velo_to_cam_file
        )
    except FileNotFoundError as e:
        print(f"Error: Calibration file not found: {e}")
        sys.exit(1)

    try:
        tracklet_processor = TrackletProcessor(data_loader.tracklet_file)
    except FileNotFoundError:
        print(f"Error: Tracklet file not found for sequence {args.sequence}")
        print("This sequence may not have tracklet annotations.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tracklets: {e}")
        sys.exit(1)

    if len(data_loader.left_image_files) == 0:
        print(f"Error: No images found for sequence {args.sequence}")
        sys.exit(1)

    if len(data_loader.point_files) == 0:
        print(f"Error: No point cloud files found for sequence {args.sequence}")
        sys.exit(1)

    # Initialize depth estimator
    cam = config['processing']['camera']

    method_name = 'Distance' if args.use_distance else 'Depth'
    lidar_title = method_name + " from LiDAR"

    if args.method == 'stereo':
        depth_estimator = StereoDepthEstimator(config['stereo'], calibration)
        method_name += ' from Stereo Matching'
    elif args.method == 'depthanything2':
        depth_estimator = DepthAnythingV2Estimator(config['depthanything2'])
        method_name += ' from DepthAnythingV2'
    elif args.method == 'unidepth':
        depth_estimator = UniDepthEstimator(config['unidepth'], calibration.cam_projections[cam])
        method_name += ' from UniDepth'
    elif args.method == 'metric3d':
        depth_estimator = Metric3DEstimator(config['metric3d'], calibration.cam_projections[cam])
        method_name += ' from Metric3D'
    else:
        depth_estimator = RAFTStereoEstimator(config['raftstereo'], calibration)
        method_name += ' from RAFT-Stereo'

    # Initialize visualizer
    visualizer = Visualizer(config, calibration, cam)

    print(f"\nProcessing {len(data_loader.left_image_files)} frames...")
    result_video = []

    for frame_idx in tqdm(range(len(data_loader.left_image_files))):
        try:
            # Load data for this frame
            img = cv2.imread(data_loader.left_image_files[frame_idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            points = data_loader.load_velodyne_points(data_loader.point_files[frame_idx])[:, :3]
            
            # Compute depth map
            if args.method == 'stereo' or args.method == 'raftstereo':
                depth_map = depth_estimator.compute_depth(
                    data_loader.left_image_files[frame_idx],
                    data_loader.right_image_files[frame_idx]
                )
            elif args.method == 'depthanything2' or args.method == 'metric3d':
                depth_map = depth_estimator.compute_depth(img)

            elif args.method == 'unidepth':
                depth_map = depth_estimator.compute_depth(data_loader.left_image_files[frame_idx])
            
            # Create visualizations
            img_lidar = img.copy()
            img_comparison = img.copy()
            
            # Process tracklets
            tracklets = tracklet_processor.get_tracklets_for_frame(frame_idx)
            
            for tracklet_idx, tracklet_data in tracklets:
                # Visualize on LiDAR image
                img_lidar = visualizer.draw_tracklet(
                    img_lidar, tracklet_data, tracklet_idx, 
                    points, depth_map, args.use_distance, use_lidar=True
                )
                
                # Visualize on comparison image
                img_comparison = visualizer.draw_tracklet(
                    img_comparison, tracklet_data, tracklet_idx,
                    points, depth_map, args.use_distance, use_lidar=False
                )
            
            frame = create_comparison_frame(img_lidar, img_comparison, lidar_title, method_name)
            result_video.append(frame)
            
        except Exception as e:
            print(f"\nWarning: Error processing frame {frame_idx}: {e}")
            continue

    # Save video
    if len(result_video) == 0:
        print("\nError: No frames were successfully processed.")
        sys.exit(1)

    print(f"\nSaving video with {len(result_video)} frames...")
    suffix = 'dist' if args.use_distance else 'depth'
    output_filename = args.sequence + "_" + args.method + "_" + suffix + ".mp4"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Existing {output_filename} has been replaced.")

    fourcc = cv2.VideoWriter_fourcc(*config['video']['codec'])
    fps = max(1, min(args.fps, 20)) if args.fps else config['video']['fps']
    out = cv2.VideoWriter(
        output_path, fourcc, fps,
        (config['video']['width'], config['video']['height'])
    )

    for frame in result_video:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()

    print(f"\n{'='*60}")
    print(f"✓ Video saved to: {output_path}")
    print(f"✓ Total frames: {len(result_video)}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()