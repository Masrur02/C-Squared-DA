# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target

def strong_transform_ca(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix_ca(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target




def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        class_list = torch.unique(labels)
        nclasses = class_list.shape[0]
        categories_index = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        categories = class_list[torch.Tensor(categories_index).long()]
        categories_new = categories
        ############################################################
        # Group 1 of meta class: object
        #-----------------------------------------------------------
        # if contains traffic light, cut the pole
        if categories_new.__contains__(6) and class_list.__contains__(5):
            categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([5]).long().cuda()), 0))
            # print('judge 6, append 5:', categories_new)
        #-----------------------------------------------------------
        # if contains traffic sign, cut the pole
        if categories_new.__contains__(7) and class_list.__contains__(5):
            categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([5]).long().cuda()), 0))
            # print('judge 7, append 5:', categories_new)
        #-----------------------------------------------------------
        # if contains pole, cut the traffic sign and traffic light
        if categories_new.__contains__(5):
            if class_list.__contains__(6):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([6]).long().cuda()), 0))
                # print('judge 5, append 6:', categories_new)
            if class_list.__contains__(7):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([7]).long().cuda()), 0))
                # print('judge 5, append 7:', categories_new)
        #-----------------------------------------------------------
        ############################################################
        # Group 2 of meta class: human-vehicle
        # if contains rider, cut the bicycle and motorcycle 
        if categories_new.__contains__(12):
            if class_list.__contains__(18):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([18]).long().cuda()), 0))
                # print('judge 12, append 18:', categories_new)
            if class_list.__contains__(17):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([17]).long().cuda()), 0))
                # print('judge 12, append 17:', categories_new)
        # print('final: categories_new', categories_new)
                
        # # Group 4 of meta class: construction
        # # if contains wall, cut the building 
        '''if categories_new.__contains__(3):
             if class_list.__contains__(2):
                 categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([2]).long().cuda()), 0))
        # # if contains wall, cut the building
        if categories_new.__contains__(4):
            if class_list.__contains__(2):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([2]).long().cuda()), 0))'''
        class_masks.append(generate_class_mask(label, categories_new).unsqueeze(0))
        #class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks

def get_context_class_masks(pseudo_labels):
    class_masks = []
    for pseudo_label in pseudo_labels: # torch.Size([19, 512, 512])
        # print('spatial_matrix.shape', spatial_matrix.shape)
        # print('ema_softmax.shape', ema_softmax.shape)
        #spatial_pred = torch.mul(spatial_matrix, ema_softmax).unsqueeze(0) # torch.Size([1, 19, 512, 512])
        pseudo_label=pseudo_label.unsqueeze(0)
        spatial_pseudo_label = pseudo_label.max(1).indices.squeeze(0) # torch.Size([512, 512])
       
        # import pdb
        # pdb.set_trace()
        class_list = torch.unique(spatial_pseudo_label)
        nclasses = class_list.shape[0]

        categories_index = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)

        categories = class_list[torch.Tensor(categories_index).long()]
        # print('categories', categories)
        # import pdb
        # pdb.set_trace()
        categories_new = categories
        ############################################################
        # Group 1 of meta class: object
        #-----------------------------------------------------------
        # if contains traffic light, cut the pole
        '''if categories_new.__contains__(6) and class_list.__contains__(5):
            categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([5]).long().cuda()), 0))
            # print('judge 6, append 5:', categories_new)
        #-----------------------------------------------------------
        # if contains traffic sign, cut the pole
        if categories_new.__contains__(7) and class_list.__contains__(5):
            categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([5]).long().cuda()), 0))
            # print('judge 7, append 5:', categories_new)
        #-----------------------------------------------------------
        # if contains pole, cut the traffic sign and traffic light
        if categories_new.__contains__(5):
            if class_list.__contains__(6):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([6]).long().cuda()), 0))
                # print('judge 5, append 6:', categories_new)
            if class_list.__contains__(7):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([7]).long().cuda()), 0))'''
                # print('judge 5, append 7:', categories_new)
        #-----------------------------------------------------------
        ############################################################
        # Group 2 of meta class: human-vehicle
        # if contains rider, cut the bicycle and motorcycle 
        if categories_new.__contains__(12):
            if class_list.__contains__(18):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([18]).long().cuda()), 0))
                # print('judge 12, append 18:', categories_new)
            if class_list.__contains__(17):
                categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([17]).long().cuda()), 0))
                # print('judge 12, append 17:', categories_new)
        # print('final: categories_new', categories_new)
        #class_masks.append(generate_class_mask(spatial_pseudo_label, categories_new).unsqueeze(0))
        # ############################################################
        # # Group 3 of meta class: flat
        # # if contains road, cut the sidewalk 
        '''if categories_new.__contains__(0):
             if class_list.__contains__(1):
                 categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([1]).long().cuda()), 0))
        # # if contains sidewalk, cut the road
        if categories_new.__contains__(1):
             if class_list.__contains__(0):
                 categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([0]).long().cuda()), 0))'''
        # ############################################################
        # # Group 4 of meta class: construction
        ##if contains wall, cut the building 
        if categories_new.__contains__(3):
             if class_list.__contains__(2):
                 categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([2]).long().cuda()), 0))
        # # if contains wall, cut the building
        if categories_new.__contains__(4):
             if class_list.__contains__(2):
                 categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([2]).long().cuda()), 0))
        # ############################################################
        # # Group 5 of meta class:nature
        # # if contains vegetation, cut the terrain 
        '''if categories_new.__contains__(8):
             if class_list.__contains__(9):
                 categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([9]).long().cuda()), 0))
        # # if contains terrain, cut the vegetation
        if categories_new.__contains__(9):
             if class_list.__contains__(8):
                 categories_new = torch.unique(torch.cat((categories_new, torch.Tensor([8]).long().cuda()), 0)) ''' 
        class_masks.append(generate_class_mask(spatial_pseudo_label, categories_new).unsqueeze(0))            
        # import pdb
        # pdb.set_trace()

    return class_masks



def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target

def one_mix_ca(mask, data = None, target = None):
    
    if not (data is None):
        
        
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[1]) 
        data = (stackedMask0*data[1]+(1-stackedMask0)*data[0]).unsqueeze(0)
       
    if not (target is None):
        
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[1])
        target = (stackedMask0*target[1]+(1-stackedMask0)*target[0]).unsqueeze(0)
    return data, target