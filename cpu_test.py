#timesformer model
import argparse
import os
import os.path as osp
import warnings
import numpy as np

import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

#from mmaction.apis import multi_gpu_test, single_gpu_test
from mmaction.apis import multi_gpu_test 
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks
import time
from tqdm import tqdm
from itertools import combinations
from mmaction.models.recognizers import GumbelSampler2DRecognizer2D, Sampler2DRecognizer2D, Sampler2DRecognizer3D
from mmaction.models.backbones import MobileNetV2TSM, ResNet, MobileNetV2
from mmaction.models.backbones.mobilenet_v2 import InvertedResidual
from mmaction.models.heads import TSMHead
from mmaction.models import build_model
from mmaction.datasets import build_dataset, build_dataloader
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import torch
from tqdm import tqdm
from mmaction.core.evaluation.accuracy import top_k_accuracy, mean_average_precision
from einops import rearrange




# config = "configs/single_anet_timesformer_6x100x1.py"
# checkpoint = "modelzoo/timesformer_6x100x1_anet.pth"
config = "kdconfigs/mini_kinetics/minik_kd_alog_mbnv2_timesformer.py"
checkpoint = "work_dirs/mini_kinetics_kd_bpr_mbnv2_timesformer/best_top1_acc_epoch_2.pth"


cfg = Config.fromfile(config)

dataset = build_dataset(cfg.data.test, dict(test_mode=True))
dataloader_setting = dict(
    videos_per_gpu=cfg.data.get('videos_per_gpu', 20),
    workers_per_gpu=cfg.data.get('workers_per_gpu', 20),
    dist=False,
    shuffle=False)
# dataloader_setting = dict(dataloader_setting,
#                             **cfg.data.get('test_dataloader', {}))
data_loader = build_dataloader(dataset, **dataloader_setting)

model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location='cpu')

def single_gpu_test(model, data_loader, original=False, two=False, permutation=True):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    print("############################################################")
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))  
    total_seg = 6
    perm_num = 2
    perms = list(combinations([i for i in range(total_seg)],perm_num))
    start = time.time()
    end = time.time()
    start = time.time()
    for data in data_loader:
        end = time.time()
        print(f"Get data {end - start:.5f} sec")
        model.eval()
        with torch.no_grad():
            imgs, labels = data['imgs'], data['label']
            ori_imgs, ori_labels = imgs.clone(), labels.clone()

            if two:
                B, T, C, H, W = imgs.shape
                imgs = imgs.reshape(B*T, C, H, W)
                feats = model.extract_feat(imgs)
                logits = model.cls_head(feats, 1)
                logits = logits.reshape(B, T, -1)
                # print(f"labels = {labels}")
                label_logits = logits[range(B), :, labels.squeeze()]
                max_idx = label_logits.max(-1)[1]
                logits = logits[range(B), max_idx]
                result = logits.detach().cpu().numpy()

                # for i in range(num_segs):
                #     logits = model.cls_head(feats[i:i+1], 1)
                #     if logits.softmax(-1)[:,labels.bool()] > result:
                #         result = logits

                    # print(result)
            elif original:
                B, N, C, T, H, W = imgs.shape
                imgs = imgs.reshape(B, C, T, H, W)
                x = model.extract_feat(imgs)
                logits = model.cls_head(x)
                result = logits.detach().cpu().numpy()
            else:
                B, N, C, T, H, W = imgs.shape
                if permutation:
                    perm_imgs = []
                    score = torch.zeros([B, total_seg])
                    for perm in perms:
                        indice = torch.tensor([1 if i in perm else 0 for i in range(total_seg)]).bool()
                        perm_imgs.append(imgs[:,:,:,indice])
                    perm_imgs = torch.stack(perm_imgs, 0)
                    perm_imgs = rearrange(perm_imgs, 'p b n c t h w -> (p b n) c t h w')
                    x = model.extract_feat(perm_imgs)
                    logits = model.cls_head(x) # B*len(perms) num_classes
                    logits = logits.reshape(B, len(perms), -1) # B len(perms) num_classes
                    label_logits = logits[range(B), :, labels.squeeze()]
                    for i in range(len(perms)):
                        for j in range(len(perms[i])):
                            score[:, perms[i][j]] += label_logits[:, i]
                    max_idx = score.topk(6, dim=1)[1].sort(dim=1,descending=False)[0]
                    batch_inds = torch.arange(B).unsqueeze(-1).expand_as(max_idx)
                    imgs = rearrange(imgs, 'b n c t h w -> (b n) t c h w')
                    sampled_imgs = imgs[batch_inds, max_idx]
                    sampled_imgs = rearrange(sampled_imgs, 'b t c h w -> b c t h w')
                    fx = model.extract_feat(sampled_imgs)
                    flogits = model.cls_head(fx)
                    result = flogits.detach().cpu().numpy()                    
                    # result = logits[batch_inds, max_idx].squeeze()
                    # result = result.detach().cpu().numpy()
                    # print(f"result.shape={result.shape}")
                    
                else:
                    imgs = imgs.reshape(B * T, C, H, W) # B*T C H W
                    x = model.extract_feat(imgs.unsqueeze(2))
                    logits = model.cls_head(x) # B*T num_classes
                    logits = logits.reshape(B, T, -1)
                    label_logits = logits[range(B), :, labels.squeeze()]
                    
                    # max_idx = label_logits.topk(1, dim=1)[1].sort(dim=1,descending=False)[0]
                    # batch_inds = torch.arange(B).unsqueeze(-1).expand_as(max_idx)
                    # result = logits[batch_inds, max_idx].squeeze()
                    
                    max_idx = label_logits.max(-1)[1]
                    result = logits[range(B), max_idx, :]

                    result = result.detach().cpu().numpy()
        results.extend(result)
        # print(np.asarray(results).shape)
        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
        start = time.time()
        # results = collect_results_gpu(results, len(dataset))
    return results

# single_gpu_test(model, data_loader)
idata = iter(data_loader)
data = next(idata)
imgs, labels = data['imgs'], data['label']
model(imgs, labels)