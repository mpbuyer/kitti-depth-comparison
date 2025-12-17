"""
Depth estimation module
Implements stereo matching and monocular methods
"""

import cv2
import numpy as np
import torch
import os
import sys
import subprocess
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class StereoDepthEstimator:
    """Stereo depth estimation using Semi-Global Block Matching"""
    
    def __init__(self, config, calibration):
        self.config = config
        self.calibration = calibration
        
        # Create SGBM matcher
        self.matcher = self._create_matcher()
        
        # Extract calibration parameters
        P2 = calibration.P2
        P3 = calibration.P3
        self.focal_length = P2[0, 0]
        self.baseline = abs((P3[0, 3] - P2[0, 3]) / P2[0, 0])
    
    def _create_matcher(self):
        """Create SGBM stereo matcher with configured parameters"""
        block_size = self.config['block_size']
        
        # Map mode string to OpenCV constant
        mode_map = {
            'SGBM': cv2.STEREO_SGBM_MODE_SGBM,
            'SGBM_3WAY': cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            'HH': cv2.STEREO_SGBM_MODE_HH
        }
        mode = mode_map.get(self.config['mode'], cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
        matcher = cv2.StereoSGBM_create(
            minDisparity=self.config['min_disparity'],
            numDisparities=self.config['num_disparities'],
            blockSize=block_size,
            P1=self.config['p1_multiplier'] * 3 * block_size**2,
            P2=self.config['p2_multiplier'] * 3 * block_size**2,
            disp12MaxDiff=self.config['disp12_max_diff'],
            uniquenessRatio=self.config['uniqueness_ratio'],
            speckleWindowSize=self.config['speckle_window_size'],
            speckleRange=self.config['speckle_range'],
            preFilterCap=self.config['pre_filter_cap'],
            mode=mode
        )
        
        return matcher
    
    def compute_depth(self, left_img_path, right_img_path):
        """ Returns: depth_map: Depth in meters (HxW) """
        left = cv2.imread(left_img_path)
        right = cv2.imread(right_img_path)

        #Grayscale?
        # left = cv2.cvtColor(cv2.imread(left_img_path), cv2.COLOR_BGR2GRAY)
        # right = cv2.cvtColor(cv2.imread(right_img_path), cv2.COLOR_BGR2GRAY)
        
        disparity = self.matcher.compute(left, right).astype(np.float32) / 16.0
        
        # Avoid division by zero
        disparity[disparity <= 0] = 0.01
        
        depth_map = (self.focal_length * self.baseline) / disparity
        
        # Clip unrealistic or irrelevant depths
        depth_map[depth_map > 100] = 100
        depth_map[depth_map < 0] = 0
        
        return depth_map


class DepthAnythingV2Estimator:
    """Monocular depth estimation using DepthAnything V2"""
    
    def __init__(self, config):
        self.config = config
        self.model_sizes = {"vits": "Small", "vitb": "Base", "vitl": "Large"}
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self._setup_depthanything()
        
        self._load_model()
    
    def _setup_depthanything(self):
        """Clone DepthAnything repository and download checkpoint"""
        repo_dir = Path("Depth-Anything-V2")
        
        if not repo_dir.exists():
            print("Cloning DepthAnything V2 repository...")
            subprocess.run(
                ['git', 'clone', 'https://github.com/DepthAnything/Depth-Anything-V2'],
                check=True,
                capture_output=True
            )
        
        # Add to Python path
        metric_depth_path = repo_dir / "metric_depth"
        sys.path.insert(0, str(metric_depth_path.absolute()))
        
        # Download checkpoint if needed
        checkpoint_dir = metric_depth_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        encoder = self.config['encoder']
        model_size = self.model_sizes[encoder]
        dataset = self.config['dataset']
        checkpoint_name = f"depth_anything_v2_metric_{dataset}_{encoder}.pth"
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            print(f"Downloading DepthAnything checkpoint ({encoder})...")
            url = (
                f"https://huggingface.co/depth-anything/"
                f"Depth-Anything-V2-Metric-{dataset.upper()}-"
                f"{model_size}/resolve/main/{checkpoint_name}"
            )
            
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length',0))

            with open(checkpoint_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=checkpoint_name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def _load_model(self):
        """Load DepthAnything model"""
        from depth_anything_v2.dpt import DepthAnythingV2
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        encoder = self.config['encoder']
        max_depth = self.config['max_depth']
        
        self.model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        self.model = self.model.to(self.device)
        
        # Load checkpoint
        checkpoint_path = (
            Path("Depth-Anything-V2/metric_depth/checkpoints") /
            f"depth_anything_v2_metric_{self.config['dataset']}_{encoder}.pth"
        )
        
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.eval()
        
        print(f"✓ DepthAnything model loaded ({encoder} on {self.device})")
    
    def compute_depth(self, img):
        # Infer depth
        return self.model.infer_image(img)
    
class UniDepthEstimator:
    """Monocular depth estimation using UniDepth V2"""
    
    def __init__(self, config, intrinsics):
        self.K = torch.tensor(intrinsics[:,:3],dtype=torch.float32)
        self.config = config
        self.backbone = config['backbone']
        self.version = config['version']
        self.model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=self.version, 
                           backbone=self.backbone, pretrained=True, trust_repo=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        if self.version == "v2":
            self.model.resolution_level = config['resolution']
            self.model.interpolation_mode = config['interpolation']

        self.model = self.model.to(self.device).eval()
        
    
    def compute_depth(self, image_path):
        rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        predictions = self.model.infer(rgb, self.K)
        return predictions['depth'].squeeze().cpu().numpy()
    
    # # Point Cloud in Camera Coordinate
    # xyz = predictions["points"]

    # # Intrinsics Prediction
    # intrinsics = predictions["intrinsics"]

class Metric3DEstimator:
    """Monocular depth estimation using Metric3D"""
    
    def __init__(self, config, intrinsics):
        self.config = config
        self.intrinsic = [intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]]
        self.mean = torch.tensor(self.config['padding']).float()[:, None, None]
        self.std = torch.tensor(self.config['std']).float()[:, None, None]

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = torch.hub.load('yvanyin/metric3d', self.config['model'], pretrain=True,  trust_repo=True).to(self.device).eval()

    def compute_depth(self, rgb):
        # how tedious: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py#L145
        rgb_origin = rgb
        h,w = rgb.shape[:2]
        scale = min(self.config['input_height']/h, self.config['input_width']/w)
        rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
        intrinsic = [scale*i for i in self.intrinsic]

        h,w = rgb.shape[:2]
        pad_h = self.config['input_height'] - h
        pad_w = self.config['input_width'] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        rgb = cv2.copyMakeBorder(rgb, pad_info[0], pad_info[1], pad_info[2], pad_info[3], 
                                 cv2.BORDER_CONSTANT, value=self.config['padding'])
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - self.mean), self.std)
        rgb = rgb[None, :, :, :].to(self.device)
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference({'input': rgb})

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()

        #### de-canonical transform
        canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)
        print(pred_depth.shape)

        return pred_depth.squeeze().cpu().numpy()

        