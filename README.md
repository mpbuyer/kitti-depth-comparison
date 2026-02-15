# KITTI Depth Comparison

Outputs a mp4 video comparing depth estimation methods on [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php) sequences. "Depth" being the closest point to LiDAR or camera in the forward direction inside 3d bounding boxes/"tracklets". Depth from LiDAR is always included with the comparison being a method using a singular or stereo camera.

https://github.com/user-attachments/assets/fcdf5cb9-db62-4637-beee-a75948f66b22

## Results
Aggregated across sequences 0001, 0005, 0014, 0015, 0048, 0052, and 0091 using LiDAR as ground truth.

| Method          | Abs Rel Error| RMSE        | δ < 1.25    |
|-----------------|--------------|-------------|-------------|
| Stereo Matching | 0.0171       | 0.8852      | 0.9912      |
| RAFT-Stereo     | **0.0102**   | **0.7132**  | **1.0000**  |
| DepthAnything V2| 0.0206       | 0.9791      | 0.9898      |
| UniDepth V2     | 0.0158       | 0.9754      | 0.9893      |

Centimeter-level accuracy on objects 10m or closer. Objects 30m+ away are occasionally a few meters off, keeping RMSE close to 1.

## Features

* Download and process KITTI raw data sequences automatically

* Compare LiDAR depth measurements with a chosen camera method

* Visualize 3D bounding boxes with depth annotations

* Generate comparison videos with top-to-bottom visualization

## Prerequisites

* Python 3.10+

* [uv](https://github.com/astral-sh/uv), git

* \~2GB disk space per sequence and deep learning model

## Installation

1. Clone this repository:

```bash
git clone https://github.com/mpbuyer/kitti-depth-comparison
cd kitti-depth-comparison
```

2. Create a virtual environment and install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

example usage:

```bash
python main.py --sequence 2011_09_26_drive_0048 --method stereo
```
Downloading KITTI zip files and deep learning model weights probably take the majority of the running time.

### Required Arguments

* `--sequence`: KITTI sequence to download and process (e.g., `2011_09_26_drive_0048`)

* `--method`: Depth comparison method - `stereo` or `raftstereo` or `depthanything2` or `unidepth`

### Optional Arguments

* `--config`: Path to config file (default: `config.yaml`)

* `--fps`: Video fps (default: 10)

* `--use_distance:` Put True if xy-plane distance (radius) is desired instead of depth

### Supported Methods

* Stereo semi-global block matching

* [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)

* [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)

* [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth)

#### Making RAFT-Stereo work
Download [the models' weights](https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip) and place the Middlebury weights (`raftstereo-middlebury.pth`) in `src/code/`

**I included [Metric3D](https://github.com/YvanYin/Metric3D) in the code but was NOT able to test it on my mps device.**

### Configuration

`config.yaml` has:

* Stereo matching parameters (SGBM settings)

* Deep learning model settings

* Video output settings

* Visualization parameters


## Output

The script generates:

* `output/{sequence}_{method}_{dist or depth}.mp4`

* Downloaded KITTI data in the working directory

* DepthAnything repository if selected

## KITTI Sequences

Some example sequences to try:

* `2011_09_26_drive_0001`

* `2011_09_26_drive_0005`

* `2011_09_26_drive_0015`

* `2011_09_26_drive_0048`

* `2011_09_26_drive_0091`

Note: Not all sequences have tracklets and point clouds.

## Not Tested

* Every method on GPU

# Bonus Bird-Eye-View
Bird-Eye-View (BEV) based on the depth map from the method used on the left colored front camera. A bumpy ride with occlusions!

https://github.com/user-attachments/assets/fe382bbb-cb2e-4973-811c-f9e3cbb968b0

example:

```bash
python main_bev.py --sequence 2011_09_26_drive_0048 --method unidepth --fps 10 --segmented True
```
* `--segmented`: remove all but roads and sidewalks
