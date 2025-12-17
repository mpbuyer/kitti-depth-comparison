"""
Visualization module
Handles drawing bounding boxes, points, and text on images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.tracklet_processing import points_in_bbox
from src.utils import calc_distances


class Visualizer:
    """Visualize depth measurements and bounding boxes on images"""
    
    def __init__(self, config, calibration, cam):
        """
        Initialize visualizer
        
        Args:
            config: Configuration dictionary
            calibration: Calibration object
            cam: Camera index
        """
        self.config = config
        self.calibration = calibration
        self.cam = cam
        
        # Extract visualization settings
        self.vis_config = config['visualization']
        self.text_config = config['text_scaling']
        self.clip_distance = self.vis_config['clip_distance']
        
        # Setup colormap for LiDAR points
        cmap = plt.colormaps.get_cmap("hsv")
        self.cmap = np.array([cmap(i/255) for i in range(256)])[:, :3] * 255
    
    def draw_tracklet(self, image, tracklet_data, tracklet_idx, lidar_points, depth_map, use_distance, use_lidar=True):
        """
        Draw tracklet visualization on image
        
        Args:
            image: Image to draw on
            tracklet_data: Tracklet data dictionary
            tracklet_idx: Tracklet index for labeling
            lidar_points: Point cloud (nx3)
            depth_map: Depth map (HxW)
            use_distance: If True, use xy-plane distance; otherwise use depth
            use_lidar: If True, use LiDAR depth; otherwise use depth_map
            
        Returns:
            Image with visualization
        """
        h, w = image.shape[:2]
        corners_3d = tracklet_data['corners_3d']
        
        # Project corners to image
        corners_2d, corner_depths = self.calibration.project_velo_to_image(
            corners_3d, self.cam, return_depths=True
        )
        
        # Skip if object is behind camera
        if np.any(corner_depths <= 0):
            return image
        
        # Get depth measurement
        if use_lidar:
            # Filter points in bounding box
            filtered_points = lidar_points[points_in_bbox(lidar_points, corners_3d)]

            # Only points in image view
            filtered_points, pts_2d, _, depths = self.calibration.get_lidar_in_image_fov(filtered_points, self.cam, 0, 0, w-1, h-1, 
                                                                           return_more=True, clip_distance=self.clip_distance)
            image = self._draw_lidar_points(image, pts_2d, depths)
        else:
            projected_points = self.calibration.depth_to_lidar_points(depth_map, self.cam)
            filtered_points = projected_points[points_in_bbox(projected_points, corners_3d)]
            
            # Segmentation at best; for debugging otherwise
            # filtered_points, pts_2d, _, depths = self.calibration.get_lidar_in_image_fov(filtered_points, self.cam, 0, 0, w-1, h-1, 
            #                                                               return_more=True, clip_distance=self.clip_distance)
            # image = self._draw_lidar_points(image, pts_2d, depths)

        # occlusions
        if len(filtered_points) < 1:
                return image
        
        # depth or xy-plane distance "radius"
        ranges = calc_distances(filtered_points) if use_distance else filtered_points[:,0]
        
        # only closest and median for now
        measurement = ranges.min() if self.config['processing']['depth_technique'] == 'closest' else np.median(ranges)
        
        # Draw 3D bounding box
        color = (
            tuple(self.vis_config['car_color']) 
            if tracklet_data['type'] == 'Car' 
            else tuple(self.vis_config['other_color'])
        )
        image = self._draw_3d_box(image, corners_2d, color)
        
        # Add depth text and label
        bbox_width = self._get_bbox_width(corners_2d, w)
        image = self._add_text_annotations(
            image, corners_2d, h, w, bbox_width,
            measurement, tracklet_data['type'], tracklet_idx
        )
        
        return image
    
    def _draw_lidar_points(self, image, pts_2d, depths):
        if len(pts_2d) == 0:
            return image
        
        # Draw points with depth coloring
        for i in range(len(pts_2d)):
            depth = depths[i]
            color = self.cmap[int(min(510.0 / max(depth, 0.1), 255)), :]
            
            cv2.circle(
                image,
                (int(np.round(pts_2d[i, 0])), int(np.round(pts_2d[i, 1]))),
                2,
                color=tuple(color),
                thickness=-1
            )
        
        return image
    
    def _draw_3d_box(self, image, corners_2d, color):
        corners_2d = corners_2d.astype(np.int32)
        thickness = self.vis_config['bbox_thickness']
        
        # Draw bottom face (indices 0-3)
        for i in range(4):
            j = (i + 1) % 4
            cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
        
        # Draw top face (indices 4-7)
        for i in range(4, 8):
            j = 4 + ((i - 4 + 1) % 4)
            cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
        
        # Draw vertical lines connecting bottom and top
        for i in range(4):
            cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i + 4]), color, thickness)
        
        # Draw front face in different color to show orientation
        cv2.line(image, tuple(corners_2d[0]), tuple(corners_2d[1]), (0, 0, 255), thickness + 1)
        
        return image
    
    def _add_text_annotations(self, image, corners_2d, h, w, bbox_width, 
                              depth_value, object_type, tracklet_idx):
        # Calculate text position
        x_min = int(max(0, corners_2d[:, 0].min()))
        y_min = int(max(0, corners_2d[:, 1].min()))
        y_max = int(min(h-1, corners_2d[:, 1].max()))
        
        cx = x_min
        cy = int((y_min + y_max) / 2)
        
        # Get text properties based on bbox size
        font_scale, thickness = self._get_text_properties(bbox_width)
        
        # Draw depth text
        depth_text = f"{depth_value:.2f}m"
        if depth_value >= 100:
            depth_text = "N/A" # stereo failed, and DepthAnything should be maxed at 80
        # black border
        cv2.putText(
            image, depth_text, (cx, cy),
            cv2.FONT_HERSHEY_DUPLEX, font_scale,
            (0,0,0), thickness + 4
        )
        # then text color
        cv2.putText(
            image, depth_text, (cx, cy),
            cv2.FONT_HERSHEY_DUPLEX, font_scale,
            tuple(self.vis_config['text_color']), thickness
        )
        
        # Draw object label
        label = f"{object_type} {tracklet_idx}"
        label_y = cy + int(26 * font_scale)
        # black border
        cv2.putText(
            image, label, (cx, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.2,
            (0,0,0), thickness + 4
        )
        # then text color
        cv2.putText(
            image, label, (cx, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.2,
            tuple(self.vis_config['label_color']), thickness
        )
        
        return image
    
    def _get_bbox_width(self, corners_2d, img_width):
        x_min = int(max(0, corners_2d[:, 0].min()))
        x_max = int(min(img_width - 1, corners_2d[:, 0].max()))
        return max(0, x_max - x_min)
    
    def _get_text_properties(self, bbox_width):
        """
        Calculate font scale and thickness based on bounding box width
        """
        cfg = self.text_config
        
        # Linear scaling
        font_scale = cfg['base_font_scale'] * (bbox_width / cfg['reference_width'])
        
        # Thickness scales with square root
        thickness = int(cfg['base_thickness'] * np.sqrt(bbox_width / cfg['reference_width']))
        
        # Clamp values
        font_scale = np.clip(font_scale, cfg['min_font_scale'], cfg['max_font_scale'])
        thickness = np.clip(thickness, cfg['min_thickness'], cfg['max_thickness'])
        
        return font_scale, thickness