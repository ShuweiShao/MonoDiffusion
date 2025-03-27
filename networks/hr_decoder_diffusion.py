from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .hr_layers import *

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from typing import Union, Dict, Tuple, Optional
def downsample(x):
    return F.interpolate(x, scale_factor=0.5, mode="bilinear")

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class HRDFDepthDecoder(nn.Module):
    def __init__(self, ch_enc = [64,128,216,288,288], scales=range(4),num_ch_enc = [ 64, 64, 128, 256, 512 ], num_output_channels=1):
        super(HRDFDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        
        # decoder
        self.convs = nn.ModuleDict()
        
        # feature fusion
        self.convs["f4"] = Attention_Module(self.ch_enc[4]  , num_ch_enc[4])
        self.convs["f3"] = Attention_Module(self.ch_enc[3]  , num_ch_enc[3])
        self.convs["f2"] = Attention_Module(self.ch_enc[2]  , num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.ch_enc[1]  , num_ch_enc[1])
        
        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                        self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        for i in range(4):
            if i<3:
                self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i]+1, self.num_output_channels)
            else:
                self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
            self.convs["uncerconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(16+32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24), nn.ELU(), Conv3x3(24,2,bias=False))
        self.delta_gen1[3].conv.weight.data.zero_()

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(32+64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48), nn.ELU(), Conv3x3(48,2,bias=False))
        self.delta_gen2[3].conv.weight.data.zero_()

        self.delta_gen3 = nn.Sequential(
            nn.Conv2d(64+128, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96), nn.ELU(), Conv3x3(96,2,bias=False))
        self.delta_gen3[3].conv.weight.data.zero_()

        self.delta = [self.delta_gen1, self.delta_gen2, self.delta_gen3]
        ##############################Diffusion#######################################
        self.model = ScheduledCNNRefine(channels_in=16, channels_noise=1)
        # 推断步数
        self.diffusion_inference_steps = 20
        # self.diffusion_inference_steps = 30
        # DDIMScheduler: 这个调度程序用于生成一系列的时间步骤，使得ScheduledCNNRefine可以在每个时间步骤上运行。
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False)
        #CNNDDIMPipiline: 这个管道类用于运行ScheduledCNNRefine，并将噪声图像降噪和扩散，直到达到预定义的时间步骤。
        self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)
        #########################################################
        self.conv1=nn.Conv2d(in_channels=17, out_channels=16, kernel_size=3, stride=1, padding=1)
        #########################################################
        self.relu=nn.ReLU(inplace=True)
    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    # def forward(self, input_features):
    def forward(self, input_features, gt):

        outputs = {}
        feat={}
        # print(input_features[4].shape)
        # print(input_features[3].shape)
        # print(input_features[2].shape)
        # print(input_features[1].shape)
        # print(input_features[0].shape)
        feat[4] = self.convs["f4"](input_features[4])
        feat[3] = self.convs["f3"](input_features[3])
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = input_features[0]

        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))

        feat_list = [x,features["X_04"],features["X_13"],features["X_22"]]
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))

        for i in range(3,0,-1):
            h, w = feat_list[i-1].shape[2:]
            high_stage = F.interpolate(input=feat_list[i],
                                   size=(h, w),
                                   mode='bilinear',
                                   align_corners=True)
            concat = torch.cat((high_stage,feat_list[i-1]),1)   #128+64 64+32 32+16
            delta = self.delta[i-1](concat) # 2 1 0

            _disp = self.bilinear_interpolate_torch_gridsample(outputs[("disp", i)], (h, w), delta)
            concat_feat = torch.cat((_disp,feat_list[i-1]),1)
            # print(concat_feat.shape)
            if i != 1:
                outputs[("disp", i-1)] = self.sigmoid(self.convs["dispconv{}".format(i-1)](concat_feat))
                outputs[("delta", i-1)] = delta
        # print(concat_feat.shape)
        # torch.Size([2, 17, 192, 640])

        ##########为了不改变16通道的卷积，在此加上一个卷积层使通道数由17->16#################
        x=self.conv1(concat_feat)
        # print(x.shape)
        ####################################################################################
        refined_depth,pred = self.pipeline(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            shape=gt.shape[-3:],
            input_args=(
                x,
                None,
                None,
                None
            ),
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False,
        )
        # ####################################################################################
        # refined_depth=self.relu(refined_depth)
        # ####################################################################################

        refined_depths=[]

        with torch.no_grad():
            for re in pred:
                # refined = self.conv_inv_transform(re)
                refined_depths.append(re)

        outputs[("disp", 0)] = refined_depth
        outputs["re-diffusion"]=refined_depths

        # print(refined_depth.shape)
        # for i in range(3):
        #     outputs[("disp", i+1)]=downsample(outputs[("disp", i)])
            
        ddim_loss = self.ddim_loss(
                pred_depth=refined_depth,
                gt_depth=gt,
                refine_module_inputs=(
                    x,
                    None,
                    None,
                    None
                ),
                ############################
                blur_depth_t=refined_depth,
                # blur_depth_t=gt,
                weight=1.0)
        # outputs[("disp", 0)] = self.sigmoid(self.convs["dispconv0"](x))                 #[12,1,192,640]
        # outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](features["X_04"]))  #[12,1,96,320]
        # outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv2"](features["X_13"]))  #[12,1,48,160]
        # outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))  #[12,1,24,80]

        # for i in self.scales:
        
        #     outputs[("uncer", 0)] = self.sigmoid(self.convs["uncerconv0"](x))
        #     outputs[("uncer", 1)] = self.sigmoid(self.convs["uncerconv1"](features["X_04"]))
        #     outputs[("uncer", 2)] = self.sigmoid(self.convs["uncerconv2"](features["X_13"]))
        #     outputs[("uncer", 3)] = self.sigmoid(self.convs["uncerconv3"](features["X_22"]))            
        outputs["ddim_loss"]=ddim_loss

        return outputs
    
    def ddim_loss(self, gt_depth, refine_module_inputs, blur_depth_t, weight, **kwargs):

        noise = torch.randn(blur_depth_t.shape).to(blur_depth_t.device)
        bs = blur_depth_t.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_depth.device).long()

        noisy_images = self.scheduler.add_noise(blur_depth_t, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss
    
    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
                             ]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class Conv3x3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=1,
                 padding=1,
                 bias=True,
                 use_refl=True,
                 name=None):
        super().__init__()
        self.name = name
        if use_refl:
            # self.pad = nn.ReplicationPad2d(padding)
            self.pad = nn.ReflectionPad2d(padding)
        else:
            self.pad = nn.ZeroPad2d(padding)
        conv = nn.Conv2d(int(in_channels),
                         int(out_channels),
                         3,
                         dilation=dilation,
                         bias=bias)
        if self.name:
            setattr(self, self.name, conv)
        else:
            self.conv = conv

    def forward(self, x):
        out = self.pad(x)
        if self.name:
            use_conv = getattr(self, self.name)
        else:
            use_conv = self.conv
        out = use_conv(out)
        return out
    

class CNNDDIMPipiline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            input_args,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            return_dict: bool = True,
            **kwargs,
    ) -> Union[Dict, Tuple]:
        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            raise RuntimeError(
                "generator.device == 'cpu'",
                "0.11.0",
                message,
            )
            generator = None

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)
        # Xt为纯高斯噪声
        self.scheduler.set_timesteps(num_inference_steps)
        #设置时间步

        pred=[]
        for t in self.scheduler.timesteps:
            # [950, 900, ..., 50, 0]

            # 1. predict noise model_output
            # 根据Fp、Xt、t预测t-1时刻的噪声
            model_output = self.model(image, t.to(device), *input_args)

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # 这里的eat=0,generator=None
            # 根据Xt、t、预测的t-1时刻噪声获得x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample']


            # print(image.shape)
            pred.append(image)
        # if not return_dict:
        #     return (image,)
        return image,pred
        # return {'images': image}


class ScheduledCNNRefine(nn.Module):
    def __init__(self, channels_in, channels_noise, **kwargs):
        super().__init__(**kwargs)
        #噪声嵌入
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(channels_noise, 16, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(4, 16),
            # # 不能用batch norm，会统计输入方差，方差会不停的变
            # nn.ReLU(True),
            # nn.Conv2d(16, channels_in, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(4, channels_in),
            # nn.ReLU(True),
        )
        #时序嵌入
        self.time_embedding = nn.Embedding(1280, channels_in)
        #预测
        self.pred = nn.Sequential(
            nn.Conv2d(channels_in, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(True),
            nn.Conv2d(16, channels_noise, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(4, channels_noise),
            # nn.ReLU(True),
            # nn.Tanh(True),
        )

    def forward(self, noisy_image, t, *args):
        #这里只用到了特征图与模糊深度
        feat, blur_depth, sparse_depth, sparse_mask = args
        # print('debug: feat shape {}'.format(feat.shape))
        # diff = (noisy_image - blur_depth).abs()
        # print('debug: noisy_image shape {}'.format(noisy_image.shape))
        
        #将特征和时序嵌入合在一起
        if t.numel() == 1:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None]
            # feat = feat + self.time_embedding(t)[None, :, None, None]
            # t 如果本身是一个值，需要扩充第一个bs维度 (这个暂时不适用)
        else:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None]
        # layer(feat) - noise_image
        # blur_depth = self.layer(feat); 
        # ret =  a* noisy_image - b * blur_depth
        # print('debug: noisy_image shape {}'.format(noisy_image.shape))

        #特征+时序+噪声
        # print(feat.shape)
        # print(self.noise_embedding(noisy_image).shape)
        feat = feat + self.noise_embedding(noisy_image)
        # feat = feat + noisy_image

        #预测噪声
        ret = self.pred(feat)

        return ret