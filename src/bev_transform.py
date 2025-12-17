
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import distance_transform_edt, gaussian_filter
matplotlib.use('Agg')  # Use non-interactive backend

class BirdEyeView:
    """Turn one camera image to bird eye view"""
    
    def __init__(self, config, calibration, cam=2):
        self.config = config
        self.calibration = calibration
        self.cam = cam
        # intrinsics
        P = self.calibration.cam_projections[cam]
        self.fx, self.fy = P[0, 0], P[1, 1]
        self.cx, self.cy = P[0, 2], P[1, 2]

        self.x_resolution = config['x_resolution']
        self.z_resolution = config['z_resolution']

        # Calculate BEV image dimensions
        self.bev_width = int((config['xmax'] - config['xmin']) / self.x_resolution)
        self.bev_height = int((config['zmax'] - config['zmin']) / self.z_resolution)

        print("Bird Eye View READY")
        print(f"BEV image width: {self.bev_width}")
        print(f"BEV image height: {self.bev_height}")

        self.bev_template = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)

    def bev_from_depth(self, image, depth_map, road_mask=None):
        """
        Create bird eye view from depth info and possible road mask
        
        Args:
            image: original colored image
            depth_map: from a depth estimation method
            
        Returns:
            bird eye view image
        """  
        h,w = image.shape[:2]
        bev_image = np.copy(self.bev_template)

        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        X = (u_coords - self.cx) * depth_map/self.fx
        Z = depth_map
        
        # Check bounds
        x_valid = (X >= self.config['xmin']) & (X <= self.config['xmax'])
        z_valid = (Z >= self.config['zmin']) & (Z <= self.config['zmax'])
        valid_mask = (depth_map > self.config['zmin']) & x_valid & z_valid

        # print(image.shape)
        # print(depth_map.shape)
        # print(road_mask.shape)

        # check road or whatever regions of interest
        if road_mask is not None:
            valid_mask = valid_mask & (road_mask > 0)

        bev_x = ((X - self.config['xmin'])/self.x_resolution).astype(np.int32)
        # reverse y-order
        bev_y = (self.bev_height - 1 -(Z - self.config['zmin'])/self.z_resolution).astype(np.int32)

        bev_x = np.clip(bev_x, 0, self.bev_width - 1)
        bev_y = np.clip(bev_y, 0, self.bev_height - 1)

        valid_v, valid_u = v_coords[valid_mask], u_coords[valid_mask]
        valid_bev_y, valid_bev_x = bev_y[valid_mask], bev_x[valid_mask]

        bev_image[valid_bev_y, valid_bev_x] = image[valid_v, valid_u]
              
        return bev_image
    
    def densify_bev_nearest_neighbor(self, bev_image, max_fill_distance=5):
        """
        Fill black/empty pixels with nearest valid neighbor
        
        Args:
            bev_image: BEV image (H, W, C) or (H, W)
            max_fill_distance: Maximum distance in pixels to fill
        """
        # Create mask of valid pixels (non-black)
        if len(bev_image.shape) == 3:
            valid_mask = np.any(bev_image > 0, axis=-1)
        else:
            valid_mask = bev_image > 0
        
        # Find nearest valid pixel for each invalid pixel
        indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
        
        # Fill invalid pixels with nearest valid pixel's value
        if len(bev_image.shape) == 3:
            densified = bev_image.copy()
            for c in range(bev_image.shape[2]):
                densified[:, :, c] = bev_image[:, :, c][tuple(indices)]
        else:
            densified = bev_image[tuple(indices)]
        
        # Optional: only fill pixels within max_fill_distance
        if max_fill_distance is not None:
            distances = distance_transform_edt(~valid_mask)
            densified[distances > max_fill_distance] = 0
        
        return densified
    
    def bev_from_IVP(self, image):
        """
        Create bird eye view from original image using inverse perspective mapping
        """
        bev_x, bev_y = np.meshgrid(np.arange(self.bev_width), np.arange(self.bev_height))
        X = bev_x * self.x_resolution + self.config['xmin']
        Z = bev_y * self.z_resolution + self.config['zmin']
        Y = -self.config['cam_pos'] # ground plane

        u = self.fx * X/Z + self.cx
        v = self.fy * Y/Z + self.cy

        bev_image = self.bilinear_sample(image, u, v)
        bev_image = np.flipud(bev_image) # had to switch y-axis at some point
        return bev_image

    def bilinear_sample(self, image, u, v):
        """determine bev_image color with 
        bilinear interpolation using the given image"""
        h,w = image.shape[:2]
        valid_mask = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        output_shape = u.shape + (image.shape[2],) if len(image.shape) == 3 else u.shape
        output = np.zeros(output_shape, dtype=image.dtype)

        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        u0, v0 = np.floor(u_valid).astype(np.int32), np.floor(v_valid).astype(np.int32)
        u1, v1 = u0 + 1, v0 + 1
        
        du = u_valid - u0
        dv = v_valid - v0
        
        # Bilinear weights
        w00 = (1 - du) * (1 - dv)
        w01 = (1 - du) * dv
        w10 = du * (1 - dv)
        w11 = du * dv

        if len(image.shape) == 3:
            w00 = w00[..., np.newaxis]
            w01 = w01[..., np.newaxis]
            w10 = w10[..., np.newaxis]
            w11 = w11[..., np.newaxis]
        
        # Weighted sum
        color = (w00 * image[v0, u0] + 
                w01 * image[v1, u0] + 
                w10 * image[v0, u1] + 
                w11 * image[v1, u1])
        
        output[valid_mask] = color
        # print(output.shape) 
        return output
    
    def valid_cam_coords(self, cam_coords):
        """
        Ensure camera coordinates are within bounds specified in the config
        """
        x_valid = (cam_coords[:,0] >= self.config['xmin']) & (cam_coords[:,0] <= self.config['xmax'])
        z_valid = (cam_coords[:,2] >= self.config['zmin']) & (cam_coords[:,2] <= self.config['zmax'])
        return x_valid & z_valid
