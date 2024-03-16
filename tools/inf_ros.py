#!/usr/bin/env python
from argparse import ArgumentParser
import rospy
import numpy as np
import time
import torch
import mmcv
import os
import matplotlib.pyplot as plt
from PIL import Image as im
from tools.ros_utils import imgmsg_to_cv2, cv2_to_imgmsg
import time
import argparse
import math
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from sensor_msgs.msg import Image
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import dataset.functional as Fd
from dataset.common import imagenet_mean, imagenet_std, colors_rugd, colors_city
from dataset.transforms import ToTensor
from modeling.deeplab import DeepLab, Decoder
from utils.visualize import un_normalize
from mmseg.apis.inference import *
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
import cv2
import torch.nn.functional as F
from copy import deepcopy

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg

def visualize_segmap(seg_map, name, dataset):
    # segmap : 1 x 21 x  h x w 
    Qn=np.array([0,1,2,9,10,12,13,21,22])
    Qnn=np.array([3,4,5,6,7,8,11,14,15,16,17,18,19,20,23])
    seg_map = seg_map.detach().cpu()
    
    if dataset == 'rugd':
        seg_map[seg_map == 255] = 24
    elif dataset == 'city':
        seg_map[seg_map == 255] = 19

    target = seg_map.argmax(1).squeeze()

    
    if dataset == 'rugd':
        colors_voc_origin = torch.Tensor(colors_rugd)
        new_im = colors_voc_origin[target.long()].numpy()
    elif dataset == 'city':
        colors_voc_origin = torch.Tensor(colors_city)
        new_im = colors_voc_origin[target.long()].numpy()
    new_im = new_im.astype(np.uint8)
    new_im = new_im[:, :, [2, 1, 0]]
    boundary=new_im.copy()
    
    for val in Qn:
        boundary[name[0] == val] = [255, 255, 255]

# Set pixels to (0, 0, 0) where the values are not in Qn
    for val in Qnn:
        boundary[name[0] == val] = [0,0,0]

    boundary_gray = cv2.cvtColor(boundary, cv2.COLOR_BGR2GRAY)
    last_black_pixels = np.zeros(boundary_gray.shape[1], dtype=int)
    for col in range(boundary_gray.shape[1]):
        column = boundary_gray[:, col]
        last_black_pixels[col] = np.where(column == 0)[0][-1]
    for col, last_black_pixel_row in enumerate(last_black_pixels):
        boundary_gray[0:last_black_pixel_row, col] = 0  # Set pixels above to black
        boundary_gray[last_black_pixel_row:, col] = 255  # Set pixels below to white

    
    boundary_updated = cv2.cvtColor(boundary_gray, cv2.COLOR_GRAY2BGR)
    
    return new_im,boundary_updated

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    #parser.add_argument('pretrained_ckpt', help='checkpoint file for eln')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=['same', 'whole', 'slide'],
        default='same',
        help='Inference mode.')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--dataset',
        type=str,
        default="city",
        help='dataset')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

class IDANAV_Manager(object):
    def __init__(self, model, palette, opacity):
        self.model = model
        self.model.eval()
        self.pal = palette
        self.opacity = opacity
        self.raw_image = None
        self.model_input = None
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        
        self.mean = np.float64(self.mean.reshape(1, -1))
        self.stdinv = 1 / np.float64(self.std.reshape(1, -1))
        
        self.pub_seg = rospy.Publisher('Multi_class_seg', Image, queue_size=10)
        self.pub_nav=rospy.Publisher('nav_image', Image, queue_size=10)
        self.pub_nav2=rospy.Publisher('nav_image2', Image, queue_size=10)
        
        rospy.Subscriber('/d400/color/image_raw', Image, self.callback)

        print('Initialization finished')

    def callback(self, msg):
        t1 = time.time()
        # The returned image is RGB
        self.raw_image = imgmsg_to_cv2(msg).astype(float)
        self.original_raw_image = deepcopy(self.raw_image)
        
        # Normalize the RGB image using the mean and std
        cv2.subtract(self.raw_image, self.mean, self.raw_image)
        cv2.multiply(self.raw_image, self.stdinv, self.raw_image)  # inplace
        
        self.model_input = self.raw_image.transpose((2, 0, 1))
        self.model_input = torch.tensor(self.model_input).unsqueeze(0).to(torch.float32)
                 
        with torch.no_grad():
            result = self.model(return_loss=False, img=[self.model_input], img_metas=[None])
            name=result[0]
            name=np.array(name)
            ema_logits=result[1]
        
        
        self.model_input=self.model_input.detach()
        self.model_input = self.model_input.to('cuda:0') 

       
        pred_img, nav_bound=visualize_segmap(ema_logits,name, args.dataset)
        

        
        # If the self.pred_img is already in RGB format, then use 'rgb8' as the encoding
        self.pub_seg.publish(cv2_to_imgmsg(pred_img, encoding='bgr8')) 
        self.pub_nav.publish(cv2_to_imgmsg(nav_bound, encoding='8UC1')) 

        self.pub_nav2.publish(cv2_to_imgmsg(nav_bound, encoding='8UC3')) 
        t2 = time.time()
        
        t=t2-t1
        print('FPS of inference: ',t)
        

if __name__ == '__main__':
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    cfg.data.test['type'] = 'MESHDataset'
    cfg.data.test['data_root'] =  '/home/vail/Masrur_ws/Datasets/MESH2'
    
    dataset = build_dataset(cfg.data.test)
    
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    
    palette = model.PALETTE
    model = MMDataParallel(model, device_ids=[0])
        
    rospy.init_node('idanav_deploy', anonymous=True)
    
    my_node = IDANAV_Manager(model, palette, args.opacity)
    rospy.spin()
