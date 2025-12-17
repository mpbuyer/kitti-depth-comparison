"""
Data loader module for KITTI dataset
Handles downloading, extracting, and loading KITTI data
"""

import os
import glob
import subprocess
import zipfile
import io
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm


class KITTIDataLoader:
    """KITTI dataset downloader and loader"""
    
    BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
    
    def __init__(self, sequence):
        """
        Args:
            sequence: KITTI sequence name (e.g., '2011_09_26_drive_0048')
        """
        self.sequence = sequence
        self.date = sequence.split('_drive_')[0]
        
        # Paths
        self.data_dir = Path(self.date)
        self.sequence_dir = self.data_dir / f"{sequence}_sync"
        
        # File lists (will be populated after download)
        self.left_image_files = []
        self.right_image_files = []
        self.point_files = []
        self.tracklet_file = None
        self.cam_to_cam_file = None
        self.velo_to_cam_file = None
    
    def download_and_extract(self):
        print(f"Preparing KITTI sequence: {self.sequence}")
        
        # Check if already downloaded
        if self.sequence_dir.exists():
            print(f"✓ Sequence directory already exists: {self.sequence_dir}")
        else:
            # Download sequence data
            self._download_file(f"{self.sequence}/{self.sequence}_sync.zip")
            
            # Download tracklets
            self._download_file(f"{self.sequence}/{self.sequence}_tracklets.zip")
        
        # Download calibration if needed
        if not self.data_dir.exists():
            self._download_file(f"{self.date}_calib.zip")
        
        self._extract_zips()
        self._load_file_paths()
        self._validate_data()
    
    def _download_file(self, relative_path):
        url = f"{self.BASE_URL}/{relative_path}"
        filename = relative_path.split('/')[-1]
        
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length',0))

        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _extract_zips(self):
        """Extract all zip files in current directory"""
        zip_files = glob.glob("*.zip")
        
        if not zip_files:
            return
        
        for zip_file in zip_files:
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall()
            os.remove(zip_file)
    
    def _load_file_paths(self):
        # Images
        self.left_image_files = sorted(
            glob.glob(str(self.sequence_dir / "image_02" / "data" / "*.png"))
        )
        self.right_image_files = sorted(
            glob.glob(str(self.sequence_dir / "image_03" / "data" / "*.png"))
        )
        
        # Point clouds
        self.point_files = sorted(
            glob.glob(str(self.sequence_dir / "velodyne_points" / "data" / "*.bin"))
        )
        
        # Tracklets
        tracklet_files = glob.glob(str(self.sequence_dir / "tracklet_labels.xml"))
        self.tracklet_file = tracklet_files[0] if tracklet_files else None
        
        # Calibration
        self.cam_to_cam_file = str(self.data_dir / "calib_cam_to_cam.txt")
        self.velo_to_cam_file = str(self.data_dir / "calib_velo_to_cam.txt")
    
    def _validate_data(self):
        """Validate that required data exists"""
        errors = []
        
        if not self.left_image_files:
            errors.append("No left camera images found")
        
        if not self.right_image_files:
            errors.append("No right camera images found")
        
        if not self.point_files:
            errors.append("No point cloud files found")
        
        if not self.tracklet_file:
            errors.append(
                f"No tracklet file found. "
                f"Sequence '{self.sequence}' may not have tracklet annotations."
            )
        
        if not os.path.exists(self.cam_to_cam_file):
            errors.append("Camera calibration file not found")
        
        if not os.path.exists(self.velo_to_cam_file):
            errors.append("Velodyne calibration file not found")
        
        if errors:
            raise Exception(
                f"Data validation failed for sequence '{self.sequence}':\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
        
        print(f"✓ Found {len(self.left_image_files)} frames")
        print(f"✓ Found {len(self.point_files)} point clouds")
        print(f"✓ Found tracklet file")
        print(f"✓ Found calibration files")
    
    @staticmethod
    def load_velodyne_points(file_path):
        """
        Load binary point cloud file from KITTI
       
       Returns:
            nx4 array of points [x, y, z, reflectance]
        """
        points = np.fromfile(file_path, dtype=np.float32)
        points = points.reshape(-1, 4)
        return points