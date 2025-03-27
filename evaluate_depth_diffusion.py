from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import time
from thop import clever_format
from thop import profile

import datasets
import networks
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def profile_once(encoder, decoder, x, pre_disp):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, pre_disp), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
    #     "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        pre_encoder_dict=torch.load("path/teacher/encoder.pth")
        pre_decoder_dict=torch.load("path/teacher/depth.pth")
        
        pre_encoder = networks.LiteMono(model="lite-mono-8m",
                                        width=opt.width, height=opt.height)
        pre_depth_decoder = networks.DepthDecoder(pre_encoder.num_ch_enc,
                                                     opt.scales)
        
        # pre_encoder = torch.nn.DataParallel(pre_encoder)
        # pre_depth_decoder = torch.nn.DataParallel(pre_depth_decoder)

        pre_encoder.cuda()
        pre_encoder.eval()
        pre_depth_decoder.cuda()
        pre_depth_decoder.eval()

        pre_encoder.load_state_dict({k: v for k, v in pre_encoder_dict.items() if k in pre_encoder.state_dict()})
        pre_depth_decoder.load_state_dict({k: v for k, v in pre_decoder_dict.items() if k in pre_depth_decoder.state_dict()})
        
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.LiteMono(model=opt.model,
                                    height=encoder_dict['height'],
                                    width=encoder_dict['width'])
        
        depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc,
                                                     opt.scales)
        
        # encoder = torch.nn.DataParallel(encoder)
        # depth_decoder = torch.nn.DataParallel(depth_decoder)

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})

        pred_disps = []
        rgbs = []
        pre_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
       
        with torch.no_grad():
            index = 1
            t_total = 0
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
            
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                # output = depth_decoder(encoder(input_color),input_color[:,1,:,:].unsqueeze(1))
                
                pre_disp = pre_depth_decoder(pre_encoder(input_color))
                # pre_disp_np = pre_disp[("disp", 0)].cpu().numpy()
                
                flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color, pre_disp)
                pred_dis = 0
                 
                for i in range(1):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    features = encoder(input_color)
                    output = depth_decoder(features, pre_disp)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    t_total += (t1 - t0)
                    index = index + 1

                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_dis += pred_disp
                
                pred_disp = pred_dis / 1
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                
                pred_disps.append(pred_disp)
                # rgbs.append(input_color_np)
                # pre_disps.append(pre_disp_np)
        print(t_total / index)
        pred_disps = np.concatenate(pred_disps)
        # rgbs = np.concatenate(rgbs)
        # pre_disps = np.concatenate(pre_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval, allow_pickle=True)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()


    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path,allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    # if opt.eval_stereo:
    #     print("   Stereo evaluation - "
    #           "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
    #     opt.disable_median_scaling = True
    #     opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    # else:
        # print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]

        disp_height, disp_width = pred_disp.shape[:2]

        # -----------------------------visualize color map-----------------------------
        # save_vis = False
        save_vis = False
        save_dir = "path"

        if save_vis:
        # rgb =.permute(1, 2, 0)
            rgb = np.transpose(rgbs[i], (1, 2, 0))
            # print(pred_disp.shape)
            pre_disp =pre_disps[i].reshape([192, 640])
            # print(pre_disp.shape)
            pic_name = filenames[i].split()[0].split('/')[-1]+'_'+filenames[i].split()[1]+'.png'
            pic_name_pre = filenames[i].split()[0].split('/')[-1]+'_'+filenames[i].split()[1]+'_pre.png'
            pic_name_gt = filenames[i].split()[0].split('/')[-1]+'_'+filenames[i].split()[1]+'_gt.png'
            pic_name_rgb = filenames[i].split()[0].split('/')[-1]+'_'+filenames[i].split()[1]+'_rgb.png'
            gt_depth_resize=cv2.resize(gt_depth, (disp_width, disp_height))
            # rgb_resize=cv2.resize(rgb, (disp_width, disp_height))
            # data[("color", 0, 0)]
            # plt.imsave(os.path.join(save_dir,pic_name), pred_disp, cmap='magma')
            # plt.imsave(os.path.join(save_dir,pic_name), pred_disp, cmap='magma')
            # plt.imsave(os.path.join(save_dir,pic_name_pre), pre_disp, cmap='magma')
            # plt.imsave(os.path.join(save_dir,pic_name_gt), gt_depth_resize, cmap='magma')
            # plt.imsave(os.path.join(save_dir,pic_name_rgb), rgb)
            # depth_name = filenames[i].split()[0].split('/')[-1]+'_'+filenames[i].split()[1]+'.npy'
            # np.save(os.path.join(save_dir,'npz',depth_name), 1/pred_disp)
        # -----------------------------------------------------------------------------
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        # if opt.eval_split == "eigen":
        if opt.eval_split == "eigen" or opt.eval_split == "fortest":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    results_edit=open('results.txt',mode='a')
    results_edit.write("\n " + 'model_name: %s '%(opt.load_weights_folder))
    results_edit.write("\n " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    results_edit.write("\n " + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    results_edit.close()
    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d))
    print("\n-> Done!")


if __name__ == "__main__":
    options = LiteMonoOptions()
    evaluate(options.parse())
