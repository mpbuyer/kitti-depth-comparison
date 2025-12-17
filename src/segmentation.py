
"""
Segmentation module with at least two use cases:
1. Filtering categories in Bird Eye View results
2. Filtering pixels when estimating depth of objects
"""

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

class SegFormerB5:
    """Semantic Segmentation with SegFormer B5 trained on CityScapes"""
    
    def __init__(self, categories):
        # Cityscapes class IDs
        self.CLASS_MAPPING = {
            'car': [13],           # car
            'truck': [14, 15],     # truck, bus, train
            'train': [16],         # train
            'person': [11, 12],    # person, rider
            'bike': [17, 18],      # motorcycle, bicycle
            'background': list(range(11)) + [19]  # everything else
        }
        self.COLOR_PALETTE = np.array([
            [0, 0, 255],      # 0: car - blue
            [139, 0, 139],    # 1: truck - purple
            [0, 255, 0],      # 2: train - green
            [255, 0, 0],      # 3: person - red
            [255, 165, 0],    # 4: bike - orange
            [200, 200, 200],  # 5: background - gray
        ], dtype=np.uint8)

        self.categories = categories

        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(self.device)

    def seg_pipeline(self, image_path):
        """From image path to recolored image based on semantic segmentation result"""
        seg_map = self.inference(image_path)
        road_mask = self.get_road_mask(seg_map)
        return seg_map, road_mask
        # new_map = self.remap_classes(seg_map)
        # return self.colorize_segmentation(new_map)
    
    def get_road_mask(self, seg_map):
        """road + sidewalks + terrain + person + rider + car"""
        # return np.isin(seg_map, [0,1,11,12,13,14,15,16,17,18])
        return np.isin(seg_map, self.categories)

    def inference(self, image_path):
        """"Segmentation model inference"""
        image = Image.open(image_path)
        h, w = image.size[1], image.size[0]
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        # inputs = {k: v.to(self.device) for k,v in input.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits_upsampled = F.interpolate(logits, size=(h,w), mode='bilinear',align_corners=False)

        return logits_upsampled.argmax(dim=1).squeeze().cpu().numpy()
    
    def remap_classes(self, seg_map):
        """"remap CityScapes classes to KITTI classes"""
        output = np.zeros_like(seg_map)
        output[:] = 5  # background default
        
        for new_id, (category, old_ids) in enumerate(self.CLASS_MAPPING.items()):
            for old_id in old_ids:
                output[seg_map == old_id] = new_id
        
        return output
    
    def colorize_segmentation(self, seg_map):
        """Convert segmentation map to colored image"""
        h, w = seg_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(len(self.COLOR_PALETTE)):
            colored[seg_map == class_id] = self.COLOR_PALETTE[class_id]
        
        return colored
