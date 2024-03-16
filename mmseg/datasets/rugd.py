from .builder import DATASETS
from .custom import CustomDataset
import numpy as np

def rgb2mask(img, color_id):
    # assert len(img) == 3
    h, w, c = img.shape
    # out = np.ones((h, w)) * 255
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if tuple(img[i, j]) in color_id:
                out[i][j] = color_id[tuple(img[i, j])]
                
    return out

def palette2id(palette_label, PALETTE):
    color_id = {tuple(c):i for i, c in enumerate(PALETTE)}
    mask = rgb2mask(palette_label, color_id)
    return mask

@DATASETS.register_module()
class RUGDDataset(CustomDataset):
    
    CLASSES = ('dirt', 'sand', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 
               'object', 'asphalt', 'gravel', 'building', 'mulch', 'rockbed', 'log',
               'bicycle', 'person', 'fence', 'bush', 'sign', 'rock', 'bridge', 'concrete', 'picnic-table')
    
    # RGB values
    PALETTE = [
        [108, 64, 20],
        [255, 229, 204],
        [0, 102, 0],
        [0, 255, 0],
        [0, 153, 153],
        [0, 128, 255],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 127],
        [64, 64, 64],
        [255, 128, 0],
        [255, 0, 0],
        [153, 76, 0],
        [102, 102, 0],
        [103, 0, 0],
        [0, 255, 128],
        [204, 153, 255],
        [102, 0, 204],
        [255, 153, 204],
        [0, 102, 102],
        [153, 204, 255],
        [102, 255, 255],
        [101, 101, 11],
        [114, 85, 47]]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs):
        super(RUGDDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )
        
        self.dataset_name = 'rugd'
    
    