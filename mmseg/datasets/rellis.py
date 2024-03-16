from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class RELLISDataset(CustomDataset):
    CLASSES = ('dirt', 'sand', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 
               'object', 'asphalt', 'gravel', 'building', 'mulch', 'rockbed', 'log',
               'bicycle', 'person', 'fence', 'bush', 'sign', 'rock', 'bridge', 'concrete', 'picnic-table')
    
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
        super(RELLISDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )
        
        self.dataset_name = 'rellis'