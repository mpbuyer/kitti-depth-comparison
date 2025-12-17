"""
Calibration module for KITTI dataset
Handles camera calibration and coordinate transformations
"""

import numpy as np

class Calibration:
    """KITTI calibration data handler"""

    def __init__(self, cam_to_cam, velo_to_cam):
        """
        Args:
            cam_to_cam, velo_to_cam: calibration files
        """
        self.calib = self.read_calib_file(cam_to_cam) | self.read_calib_file(velo_to_cam)
        
        # Camera projection matrices
        # camera 0 (left gray), 1 (right gray), 2 (left color), 3 (right color)
        self.P0 = self.calib['P_rect_00'].reshape(3, 4)
        self.P1 = self.calib['P_rect_01'].reshape(3, 4)
        self.P2 = self.calib['P_rect_02'].reshape(3, 4)
        self.P3 = self.calib['P_rect_03'].reshape(3, 4)

        self.cam_projections = {0: self.P0, 1: self.P1, 2: self.P2, 3: self.P3}
        
        # Rectification matrices
        self.R0_rect = self.calib['R_rect_00'].reshape(3, 3)
        self.R2_rect = self.calib['R_rect_02'].reshape(3, 3)

        #unrectify
        self.R0_rect_inv = np.linalg.inv(self.R0_rect)
        
        # Velodyne to camera transformation
        self.R = self.calib['R'].reshape(3, 3)  # 3x3 rotation
        self.t = self.calib['T'].reshape(3, 1)  # 3x1 translation
        self.Tr_velo_to_cam = np.hstack([self.R, self.t])  # [R | T] -> 3x4

        # Save velo -> image matrix computations
        self.R0_homo = np.vstack((self.R0_rect, np.zeros((1, 3))))
        self.r0_rt = self.R0_homo @ self.Tr_velo_to_cam
        # cam 2 most likely chosen
        self.p2_r0_rt = self.P2 @ self.r0_rt

    @staticmethod
    def read_calib_file(filepath):
        """  Returns: Dictionary of calibration parameters"""
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    @staticmethod
    def cart2hom(pts_3d):
        """Convert Cartesian coordinates to Homogeneous coordinates"""
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_velo_to_image(self, pts_3d_velo, cam, return_depths=False):
        """
        Project 3D points from Velodyne frame to image frame
        
        Args:
            pts_3d_velo: nx3 array of 3D points in Velodyne frame
            cam: Camera index (0, 1, 2, or 3)
            return_depths: If True, also return depth values
            
        Returns:
            pts_2d: nx2 array of 2D pixel coordinates
            depths: (optional) n array of depth values
        """  
        # Choose projection matrix
        proj = self.cam_projections[cam] @ self.r0_rt if cam != 2 else self.p2_r0_rt
        
        pts_3d_homo = np.transpose(self.cart2hom(pts_3d_velo))
        
        # Project to image
        p_r0_rt_x = proj @ pts_3d_homo
        pts_2d = np.transpose(p_r0_rt_x)

        if return_depths:
            depths = pts_2d[:,2].copy()
        
        # Convert from homogeneous to Euclidean coordinates
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        
        if return_depths:
            return pts_2d[:, :2], depths
        
        return pts_2d[:, :2]
    
    def project_camera_to_velo(self, pts_3d_cam):
        """
        Project 3D points from camera frame to Velodyne frame
        """

        pts_cam0 = (self.R0_rect_inv @ pts_3d_cam.T).T
        
        R = self.Tr_velo_to_cam[:,:3]
        t = self.Tr_velo_to_cam[:,3]

        #Inverse R|T
        R_inv = R.T
        return (R_inv @ (pts_cam0 - t).T).T
    

    
    def depth_to_camera_coords(self, depth_map, cam):
        """Convert depth map to 3D camera points"""
        # Choose camera projection matrix
        P = self.cam_projections[cam]
        h, w = depth_map.shape

        # create grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u, v, depth = u.flatten(), v.flatten(), depth_map.flatten()

        fx, fy = P[0, 0], P[1, 1]
        cx, cy = P[0, 2], P[1, 2]

        # inverse of P
        x = (u - cx)*depth/fx
        y = (v - cy)*depth/fy
        z = depth

        return np.stack([x,y,z], axis=1)
    
    def depth_to_lidar_points(self, depth_map, cam):
        """depth map -> LiDAR coordinates pipeline"""
        pts_cam = self.depth_to_camera_coords(depth_map, cam)
        return self.project_camera_to_velo(pts_cam)

    def get_lidar_in_image_fov(self, pc_velo, cam, xmin, ymin, xmax, ymax, 
                                return_more=False, clip_distance=2.0):
        """
        Filter LiDAR points to keep only those in image field of view
        
        Args:
            pc_velo: nx3 array of 3D points in Velodyne frame
            cam: Camera index
            xmin, ymin, xmax, ymax: Image boundaries
            return_more: If True, return additional information
            clip_distance: Minimum distance threshold (meters)
            
        Returns:
            imgfov_pc_velo: Filtered point cloud
            pts_2d: (optional) 2D projections
            fov_inds: (optional) Boolean mask
            depths: (optional)
        """
        pts_2d, depths = self.project_velo_to_image(pc_velo, cam, return_depths=True)
        
        # Filter points within image bounds
        fov_inds = (
            (pts_2d[:, 0] < xmax) &
            (pts_2d[:, 0] >= xmin) &
            (pts_2d[:, 1] < ymax) &
            (pts_2d[:, 1] >= ymin)
        )
        
        # Filter points too close to camera
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
        
        imgfov_pc_velo = pc_velo[fov_inds, :]
        
        if return_more:
            return imgfov_pc_velo, pts_2d, fov_inds, depths
        else:
            return imgfov_pc_velo