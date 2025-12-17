"""
Utility functions
Helper functions for data processing and file management
"""

import os
import statistics
import random
import numpy as np


def setup_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.abspath(output_dir)

def calc_distances(points):
    distances = [np.sqrt(points[i,0]**2 + points[i,1]**2) for i in range(len(points))]
    return np.array(distances)

"""
Outlier Rejection methods
=====================================================================================
"""

def filter_outliers_IQR(distances):
    '''Use +- 1.5*IQR to determine outliers'''
    q25, q75 = np.percentile(distances, [25, 75])
    iqr = q75 - q25
    return distances[(distances > q25 - 1.5*iqr) & (distances < q75 + 1.5*iqr)]

def filter_outliers_SD(distances):
    '''Use standard deviations away from mean to determine outliers'''
    if len(distances) == 0:
        return []
    
    if len(distances) == 1:
        return list(distances)
    
    inliers = []
    mu = statistics.mean(distances)
    std = statistics.stdev(distances)
    
    for x in distances:
        if abs(x - mu) < std:
            inliers.append(x)
    
    # If all values were filtered out, return original
    if len(inliers) == 0:
        return list(distances)
    
    return inliers


def filter_outliers_RANSAC(distances):
    '''Use RANSAC to determine outliers'''
    return None

"""
=====================================================================================
"""


def get_best_distance(distances, technique="closest"):
    if len(distances) == 0:
        return 0.0
    
    if technique == "closest":
        return min(distances)
    elif technique == "average":
        return statistics.mean(distances)
    elif technique == "random":
        return random.choice(distances)
    else:  # median
        return statistics.median(sorted(distances))


def validate_sequence_format(sequence):
    """
    Validate KITTI sequence name format
    """
    # Expected format: YYYY_MM_DD_drive_XXXX
    parts = sequence.split('_')
    
    if len(parts) != 5:
        return False
    
    if parts[3] != 'drive':
        return False
    
    try:
        # Check date parts are numeric
        int(parts[0])  # year
        int(parts[1])  # month
        int(parts[2])  # day
        int(parts[4])  # drive number
        return True
    except ValueError:
        return False


def format_time(seconds):
    """
    Format seconds into human-readable time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"