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
from einops import rearrange

# Oracle
def single_gpu_test(model, data_loader, original=False, two=False, oracle=False, permutation=False):
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
    sampled_indice = []
    dataset = data_loader.dataset
    start = time.time()
    total_seg = 10
    perm_num = 3
    topk = 6
    softmax = True 
    perms = torch.tensor(list(combinations([i for i in range(total_seg)],perm_num)))
    print(f"two = {two}, original = {original}, oracle = {oracle}, permutation = {permutation}, softmax = {softmax}")
    print(f"perm_num = {perm_num}, topk = {topk}")
    prog_bar = mmcv.ProgressBar(len(dataset))  
    for data in data_loader:
        end = time.time()
        # print(f"Get data {end - start:.5f} sec")
        model.eval()
        with torch.no_grad():
            imgs, labels = data['imgs'], data['label'].squeeze()
            imgs, labels = imgs.cuda(), labels.cuda()
            if two:
                B, T, C, H, W = imgs.shape
                imgs = imgs.reshape(B*T, C, H, W)
                feats = model.module.extract_feat(imgs)
                logits = model.module.cls_head(feats, 1)
                logits = logits.reshape(B, T, -1)
                if softmax:
                    logits = logits.softmax(-1)
                # print(f"labels = {labels}")
                label_logits = logits[range(B), :, labels.squeeze()]
                max_idx = label_logits.max(-1)[1]
                logits = logits[range(B), max_idx]
                result = logits.detach().cpu().numpy()

                # for i in range(num_segs):
                #     logits = model.module.cls_head(feats[i:i+1], 1)
                #     if logits.softmax(-1)[:,labels.bool()] > result:
                #         result = logits

                    # print(result)
            elif original:
                B, N, C, T, H, W = imgs.shape
                imgs = imgs.reshape(B, C, T, H, W)
                x = model.module.extract_feat(imgs)
                logits = model.module.cls_head(x)
                if softmax:
                    logits = logits.softmax(-1)
                result = logits.detach().cpu().numpy()
            elif oracle:
                B, N, C, T, H, W = imgs.shape
                imgs = rearrange(imgs, 'b n c t h w -> (b n) c t h w')
                perm_logits = []
                score = torch.zeros([B, total_seg]).cuda()
                # best = torch.ones([B]).cuda() * -100
                for perm in perms:
                    indice = torch.tensor([1 if i in perm else 0 for i in range(total_seg)]).bool()
                    x = model.module.cls_head(model.module.extract_feat(imgs[:,:,indice]))
                    if softmax:
                        x = x.softmax(-1)
                    perm_logits.append(x) # B num_classes
                perm_logits = torch.stack(perm_logits, 1) # B len(perms) C
                label_logits = perm_logits[range(B), :, labels.squeeze()]
                max_idx = label_logits.max(-1)[1]
                sampled_indice.extend(perms[max_idx].detach().cpu().numpy())
                result = perm_logits[range(B), max_idx, :]
                result = result.detach().cpu().numpy()
            else:
                B, N, C, T, H, W = imgs.shape
                if permutation:
                    perm_imgs = []
                    score = torch.zeros([B, total_seg]).cuda()
                    for perm in perms:
                        indice = torch.tensor([1 if i in perm else 0 for i in range(total_seg)]).bool()
                        perm_imgs.append(imgs[:,:,:,indice])
                    perm_imgs = torch.stack(perm_imgs, 0)
                    perm_imgs = rearrange(perm_imgs, 'p b n c t h w -> (p b n) c t h w')
                    x = model.module.extract_feat(perm_imgs)
                    logits = model.module.cls_head(x) # B*len(perms) num_classes
                    if softmax:
                        logits = logits.softmax(-1)
                    logits = rearrange(logits, '(b p) n -> b p n', b = B, p = len(perms)) # B len(perms) num_classes
                    label_logits = logits[range(B), :, labels.squeeze()]
                    #for i in range(len(perms)):
                    #    for j in range(len(perms[i])):
                    #       score[:, perms[i][j]] += label_logits[:, i]
                    total_score = label_logits.sum(-1)
                    tmp = torch.zeros_like(total_score)
                    for i in range(total_seg):
                        for j in range(len(perms)):
                            if i in perms[j]:
                                tmp += label_logits[:, j]
                        score[:, i] = total_score - tmp
                    max_idx = score.topk(topk, dim=1)[1].sort(dim=1,descending=False)[0]
                    batch_inds = torch.arange(B).unsqueeze(-1).expand_as(max_idx)
                    imgs = rearrange(imgs, 'b n c t h w -> (b n) t c h w')
                    sampled_imgs = imgs[batch_inds, max_idx]
                    sampled_imgs = rearrange(sampled_imgs, 'b t c h w -> b c t h w')
                    fx = model.module.extract_feat(sampled_imgs)
                    flogits = model.module.cls_head(fx)
                    result = flogits.detach().cpu().numpy()
                else:
                    imgs = rearrange(imgs, 'b n c t h w -> (b n t) c h w')
                    # imgs = imgs.reshape(B * T, C, H, W) # B*T C H W
                    x = model.module.extract_feat(imgs.unsqueeze(2))
                    logits = model.module.cls_head(x) # B*T num_classes
                    logits = logits.reshape(B, T, -1)
                    if softmax:
                        logits = logits.softmax(-1)
                    label_logits = logits[range(B), :, labels.squeeze()]

                    max_idx = label_logits.topk(topk, dim=1)[1].sort(dim=1,descending=False)[0]
                    batch_inds = torch.arange(B).unsqueeze(-1).expand_as(max_idx)
                    imgs = rearrange(imgs, '(b t) c h w -> b t c h w', b=B, t=T)
                    sampled_imgs = imgs[batch_inds, max_idx]
                    sampled_imgs = rearrange(sampled_imgs, 'b t c h w -> b c t h w')
                    fx = model.module.extract_feat(sampled_imgs)
                    flogits = model.module.cls_head(fx)
                    result = flogits.detach().cpu().numpy()

        results.extend(result)
        # print(np.asarray(results).shape)
        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
        start = time.time()
        # results = collect_results_gpu(results, len(dataset))
    return results, sampled_indice

# def single_gpu_test(model, data_loader):
#     """Test model with a single gpu.

#     This method tests model with a single gpu and displays test progress bar.

#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (nn.Dataloader): Pytorch data loader.

#     Returns:
#         list: The prediction results.
#     """
#     model.eval()
#     results = torch.zeros(181).to(model.device)
#     count = torch.zeros(181).to(model.device)
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     for data in data_loader:
#         with torch.no_grad():
#             imgs, labels = data['imgs'], data['label']
#             label = labels.argmax().item()
#             batches = imgs.shape[0]
#             num_segs = imgs.shape[0] // batches
#             feats = model.extract_feat(imgs)
#             # logits = model.cls_head(feats, num_segs)
#             feature = torch.flatten(model.cls_head.avg_pool(feats), 1)
#             print(feature)
#         results[label] += feature
#         count[label] += feature
#         # use the first key as main key to calculate the batch size
#         batch_size = len(next(iter(data.values())))
#         for _ in range(batch_size):
#             prog_bar.update()
#     results /= count
#     return results



def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results in cpu mode.

    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # synchronizes all processes to make sure tmpdir exist
    dist.barrier()
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    # synchronizes all processes for loding pickle file
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results in gpu mode.

    It encodes results to gpu tensors and use gpu communication for results
    collection.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    return None

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def inference_pytorch(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    print("with pytorch")
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    # turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    print("Building model Start")
    start = time.time()
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    end = time.time()
    print(f"Done {end - start:.5f} sec")
    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    print("Loading weight Start")
    start = time.time()
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    end = time.time()
    print(f"Done {end - start:.5f} sec")
    again_load = cfg.get('again_load', None)
    if again_load is not None:
        load_checkpoint(model, again_load, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    print("Parallelizing Start")
    start = time.time()
    model = MMDataParallel(model, device_ids=[0])
    end = time.time()
    print(f"Done {end - start:.5f} sec")
    outputs = single_gpu_test(model, data_loader)

    return outputs

def main():
    print("Loading args")
    start = time.time()
    args = parse_args()
    end = time.time()
    print(f"Done {end - start:.5f} sec")

    if args.tensorrt and args.onnx:
        raise ValueError(
            'Cannot set onnx mode and tensorrt mode at the same time.')

    # args.config = "configs/single_anet_timesformer_6x100x1.py"
    # args.checkpoint = "modelzoo/timesformer_6x100x1_anet.pth"

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # dataset = build_dataset(cfg.data.train, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    print("Building dataloader")
    start = time.time()
    data_loader = build_dataloader(dataset, **dataloader_setting)
    end = time.time()
    print(f"Done {end - start:.5f} sec")

    print("pytorch")
    outputs, sampled_indice = inference_pytorch(args, cfg, distributed, data_loader)


    import numpy as np
    a = np.asarray(outputs)
    b = np.asarray(sampled_indice)
    #torch.save(a, './tensors/sthv2_u10to2_oracle_s_result.pt')
    #torch.save(b, './tensors/sthv2_u10to2_oracle_indice.pt')


    time.sleep(2)
    rank, _ = get_dist_info()

    if rank == 0:
        if output_config.get('out', None):
            out = output_config['out']
            print(f'\nwriting results to {out}')
            dataset.dump_results(outputs, **output_config)
        if eval_config:
            eval_res = dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
