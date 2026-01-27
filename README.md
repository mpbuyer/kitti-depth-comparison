# KITTI Depth Comparison

Outputs a mp4 video comparing depth estimation methods on [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php) sequences. Depth being the closest point to LiDAR or camera in the forward direction inside 3d bounding boxes/"tracklets". Depth from LiDAR is always included with the comparison being a method using a singular or stereo camera.

https://github.com/user-attachments/assets/fcdf5cb9-db62-4637-beee-a75948f66b22

## Results
Mean Absolute Error in meters of each camera method (using LiDAR as ground truth) computed on 7 select sequences (in the city category) with **bold** being the best on the corresponding sequence. *Overall* treats the sequences together.

| Sequence | Stereo Matching | RAFT-Stereo | DepthAnything V2 | UniDepth V2 |
|----------|-----------------|-------------|-----------------|------------|
| 0001     | 0.8103          | 0.6219      | 0.0826          | **0.0753**     |
| 0005     | **0.1737**          | 0.1844      | 0.2245          | 0.1886     |
| 0014     | 0.5555          | **0.2162**      | 0.4882          | 0.3038     |
| 0015     | 0.5963          | **0.3810**      | 0.3876          | 0.3820     |
| 0048     | 0.3898          | 0.2635      | 0.1751          | **0.1047**     |
| 0052     | 0.1300          | **0.1203**      | 0.2665          | 0.1585     |
| 0091     | 0.1314          | 0.1320      | 0.1564          | **0.0877**     |
| overall  | 0.3875          | 0.2612      | 0.2892          | <u>**0.2172**</u>|

Sub-meter accuracy across the board!

## Features

* Download and process KITTI raw data sequences automatically

* Compare LiDAR depth measurements with a chosen camera method

* Visualize 3D bounding boxes with depth annotations

* Generate comparison videos with top-to-bottom visualization

## Prerequisites

* Python 3.10+

* [uv](https://github.com/astral-sh/uv), git

* GPU ideally or mps (for neural network inference)

* \~2GB disk space per sequence (and deep learning models)

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
