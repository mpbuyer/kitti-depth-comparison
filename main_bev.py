#!/usr/bin/env python3
"""
KITTI Bird Eye View - Main Script
Compare original image with bird eye view from depth information
"""

import argparse
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
from src.bev_transform import BirdEyeView
from src.segmentation import *

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Convert KITTI images to Bird Eye View')
    
    parser.add_argument('--sequence',type=str,required=True,
                        help='KITTI sequence to process (e.g., 2011_09_26_drive_0048)')

    parser.add_argument('--method',type=str,choices=['stereo', 'depthanything2', 'unidepth', 'metric3d'],required=True,
                        help='Depth estimation method')
    
    parser.add_argument('--segmented', type=bool, default=False, help='Whether to segment KITTI images for readability')
    
    parser.add_argument('--config',type=str,default='config.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--fps',type=int, default=10, help='Video fps between 1 and 20')

    return parser.parse_args()

def create_comparison_frame(img, bev_img, img_title, bev_title, target_size=(1500, 1000)):

    fig, axes = plt.subplots(1, 2, figsize=(20,8))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1)

    aspect_orig = img.shape[1] / img.shape[0]  # width/height
    aspect_bev = bev_img.shape[1] / bev_img.shape[0]

    # original
    axes[0].imshow(img)
    axes[0].set_title(img_title, fontsize=20)
    axes[0].axis('off')
    # axes[0].set_aspect(1.0)

    # bird eye view
    axes[1].imshow(bev_img)
    axes[1].set_title(bev_title, fontsize=20)
    axes[1].axis('off')
    # axes[1].set_aspect('auto')
    axes[1].set_box_aspect(aspect_orig * 0.32)

    # print(img.shape)
    # print(bev_img.shape)

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
    print(f"KITTI Bird Eye View")
    print(f"Sequence: {args.sequence}")
    print(f"Depth Method: {args.method.upper()}")
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

    method_name = 'Depth'
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
    else:
        depth_estimator = Metric3DEstimator(config['metric3d'], calibration.cam_projections[cam])
        method_name += ' from Metric3D'

    # Initialize visualizer, bird_eye_view
    # visualizer = Visualizer(config, calibration, cam)
    bird_eye_view = BirdEyeView(config['bird_eye_view'], calibration, cam=cam)

    print(f"\nProcessing {len(data_loader.left_image_files)} frames...")
    result_video = []

    for frame_idx in tqdm(range(len(data_loader.left_image_files))):
        # try:
            # Load data for this frame
            img = cv2.imread(data_loader.left_image_files[frame_idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            points = data_loader.load_velodyne_points(data_loader.point_files[frame_idx])[:, :3]
            
            # Compute depth map
            if args.method == 'stereo':
                depth_map = depth_estimator.compute_depth(
                    data_loader.left_image_files[frame_idx],
                    data_loader.right_image_files[frame_idx]
                )
            elif args.method == 'depthanything2':
                depth_map = depth_estimator.compute_depth(img)

            elif args.method == 'unidepth':
                depth_map = depth_estimator.compute_depth(data_loader.left_image_files[frame_idx])

            else:
                depth_map = depth_estimator.compute_depth(img)
            
            # Create visualizations
            # cam_points = calibration.depth_to_camera_coords(depth_map, cam)

            if args.segmented:
                Segmentator = SegFormerB5(config['segment']['categories'])
                segmented_img, road_mask = Segmentator.seg_pipeline(data_loader.left_image_files[frame_idx])
                # bev_img = bird_eye_view.bev_from_depth(segmented_img, depth_map, road_mask=road_mask)
                bev_img = bird_eye_view.bev_from_depth(img, depth_map, road_mask=road_mask)
            else:
                bev_img = bird_eye_view.bev_from_depth(img, depth_map)

            bev_img = bird_eye_view.densify_bev_nearest_neighbor(bev_img, max_fill_distance=5)
            # bev_img = bird_eye_view.bev_from_IVP(img)
            frame = create_comparison_frame(img, bev_img, "Original","BEV")
            result_video.append(frame)
            
        # except Exception as e:
        #     print(f"\nWarning: Error processing frame {frame_idx}: {e}")
        #     continue

    # Save video
    if len(result_video) == 0:
        print("\nError: No frames were successfully processed.")
        sys.exit(1)

    print(f"\nSaving video with {len(result_video)} frames...")
    suffix = 'BEV'
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