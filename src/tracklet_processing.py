"""
Tracklet processing module
Handles loading and processing KITTI tracklet annotations
"""

import numpy as np
import parseTrackletXML as xmlParser


class TrackletProcessor:
    """Process KITTI tracklet annotations"""
    
    def __init__(self, tracklet_file):
        self.tracklets = xmlParser.parseXML(tracklet_file)
    
    def get_tracklets_for_frame(self, frame_idx):
        """Returns: List of (tracklet_idx, tracklet_data) tuples for the frame """
        result = []
        
        for i, tracklet in enumerate(self.tracklets):
            if frame_idx < tracklet.firstFrame:
                continue
            if frame_idx >= tracklet.firstFrame + tracklet.nFrames:
                continue
            
            tracklet_data = self._get_tracklet_data(tracklet, frame_idx)
            result.append((i, tracklet_data))
        
        return result
    
    def _get_tracklet_data(self, tracklet, frame_idx):
        local_frame = frame_idx - tracklet.firstFrame
        
        return {
            'type': tracklet.objectType,
            'size': tracklet.size,  # [h, w, l]
            'translation': tracklet.trans[local_frame],
            'rotation': tracklet.rots[local_frame],
            'corners_3d': self._get_3d_bbox_corners(tracklet, frame_idx)
        }
    
    @staticmethod
    def _get_3d_bbox_corners(tracklet, frame_idx):
        """Returns: 8x3 array of corner coordinates in LiDAR frame"""
        if frame_idx < tracklet.firstFrame or frame_idx >= tracklet.firstFrame + tracklet.nFrames:
            return None
        
        # Get pose at this frame
        local_frame = frame_idx - tracklet.firstFrame
        translation = tracklet.trans[local_frame]
        rotation = tracklet.rots[local_frame]
        h, w, l = tracklet.size
        
        # Create 3D bounding box corners in LiDAR coordinate system
        # Box is at bottom center
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [0, 0, 0, 0, h, h, h, h]
        corners = np.array([x_corners, y_corners, z_corners])
        
        # Rotate around z-axis
        rot_mat = np.array([
            [np.cos(rotation[2]), -np.sin(rotation[2]), 0],
            [np.sin(rotation[2]), np.cos(rotation[2]), 0],
            [0, 0, 1]
        ])
        corners = rot_mat @ corners
        
        # Translate
        corners[0, :] += translation[0]
        corners[1, :] += translation[1]
        corners[2, :] += translation[2]
        
        return corners.T


def points_in_bbox(points, bbox_corners):
    """Returns: Boolean mask of points inside bbox"""
    min_corner = bbox_corners.min(axis=0)
    max_corner = bbox_corners.max(axis=0)
    
    mask = (
        (points[:, 0] >= min_corner[0]) & (points[:, 0] <= max_corner[0]) &
        (points[:, 1] >= min_corner[1]) & (points[:, 1] <= max_corner[1]) &
        (points[:, 2] >= min_corner[2]) & (points[:, 2] <= max_corner[2])
    )
    
    return mask
