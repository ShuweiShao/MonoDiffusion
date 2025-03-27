from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from typing import Union, Dict, Tuple, Optional

class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # print(i, num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("conconv", s)] = ConvBlock(self.num_ch_dec[s], 16)
            if s != 2:
                self.convs[("disptocon", s)] = ConvBlock(1, 16)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)
        ##############################Diffusion#######################################
        self.model_2 = ScheduledCNNRefine(channels_in=16, channels_noise=1)
        self.model_1 = ScheduledCNNRefine(channels_in=16, channels_noise=1)
        self.model_0 = ScheduledCNNRefine(channels_in=16, channels_noise=1)
        # 推断步数
        self.diffusion_inference_steps_2 = 5
        self.diffusion_inference_steps_1 = 4
        self.diffusion_inference_steps_0 = 3
        # self.diffusion_inference_steps = 30
        # DDIMScheduler: 这个调度程序用于生成一系列的时间步骤，使得ScheduledCNNRefine可以在每个时间步骤上运行。
        self.scheduler_2 = DDIMScheduler(num_train_timesteps=250, clip_sample=False)
        self.scheduler_1 = DDIMScheduler(num_train_timesteps=200, clip_sample=False)
        self.scheduler_0 = DDIMScheduler(num_train_timesteps=150, clip_sample=False)
        #CNNDDIMPipiline: 这个管道类用于运行ScheduledCNNRefine，并将噪声图像降噪和扩散，直到达到预定义的时间步骤。
        self.pipeline_2 = CNNDDIMPipiline_2(self.model_2, self.scheduler_2)
        self.pipeline_1 = CNNDDIMPipiline_1(self.model_1, self.scheduler_1)
        self.pipeline_0 = CNNDDIMPipiline_0(self.model_0, self.scheduler_0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features, gt, mask=None):
        self.outputs = {}

        if mask == None:
            input_features = input_features
        else:
            b, c, h, w = input_features[0].shape
            mask_initial =  (torch.rand(b, 1, h, w).to(input_features[0].device) > 0.2).float()
            input_features[0] = input_features[0] * mask_initial
            self.outputs[("mask", 0)] = mask_initial

            for i in range(len(input_features)):
                if i > 0:
                    b, c, h, w = input_features[i].shape
                    mask = F.interpolate(mask_initial, [h, w], mode="nearest")
                    input_features[i] = input_features[i] * mask
                    self.outputs[("mask", i)] = mask
    
        x = input_features[-1]
        conditions = []
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                f = upsample(self.convs[("conconv", i)](x), mode='bilinear')
                conditions.append(f)
        
        refined_depth_2, pred_2, initial_noise = self.pipeline_2(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            shape=gt[("disp_diffusion", 2)].shape[-3:],
            input_args=(
                conditions[0],
                None,
                None,
                None
            ),
            num_inference_steps=self.diffusion_inference_steps_2,
            return_dict=False,
        )

        refined_depth_1, pred_1 = self.pipeline_1(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            shape=gt[("disp_diffusion", 1)].shape[-3:],
            ini_noise = upsample(initial_noise),
            input_args=(
                conditions[1] + upsample(self.convs[("disptocon", 1)](refined_depth_2)),
                None,
                None,
                None
            ),
            num_inference_steps=self.diffusion_inference_steps_1,
            return_dict=False,
        )

        refined_depth_0, pred_0 = self.pipeline_0(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            shape=gt[("disp_diffusion", 0)].shape[-3:],
            ini_noise = upsample(initial_noise, scale_factor=4),
            input_args=(
                conditions[2] + upsample(self.convs[("disptocon", 0)](refined_depth_1)),
                None,
                None,
                None
            ),
            num_inference_steps=self.diffusion_inference_steps_0,
            return_dict=False,
        )

        refined_depths=[]

        with torch.no_grad():
            for re in pred_0:
                refined_depths.append(self.sigmoid(re))

        self.outputs[("disp", 0)] = self.sigmoid(refined_depth_0)
        self.outputs[("disp", 1)] = self.sigmoid(refined_depth_1)
        self.outputs[("disp", 2)] = self.sigmoid(refined_depth_2)
        self.outputs["re-diffusion"]=refined_depths
            
        ddim_loss_2 = self.ddim_loss_2(
                pred_depth=refined_depth_2,
                gt_depth=gt[("disp_diffusion", 2)],
                refine_module_inputs=(
                    conditions[0],
                    None,
                    None,
                    None
                ),
                # blur_depth_t=refined_depth,
                blur_depth_t=gt[("disp_diffusion", 2)],
                weight=1.0)

        ddim_loss_1 = self.ddim_loss_1(
                pred_depth=refined_depth_1,
                gt_depth=gt[("disp_diffusion", 1)],
                refine_module_inputs=(
                    conditions[1] + upsample(self.convs[("disptocon", 1)](refined_depth_2)),
                    None,
                    None,
                    None
                ),
                # blur_depth_t=refined_depth,
                blur_depth_t=gt[("disp_diffusion", 1)],
                weight=1.0)
        
        ddim_loss_0 = self.ddim_loss_0(
                pred_depth=refined_depth_0,
                gt_depth=gt[("disp_diffusion", 0)],
                refine_module_inputs=(
                    conditions[2] + upsample(self.convs[("disptocon", 0)](refined_depth_1)),
                    None,
                    None,
                    None
                ),
                # blur_depth_t=refined_depth,
                blur_depth_t=gt[("disp_diffusion", 0)],
                weight=1.0)
         
        self.outputs["ddim_loss", 2] = ddim_loss_2
        self.outputs["ddim_loss", 1] = ddim_loss_1
        self.outputs["ddim_loss", 0] = ddim_loss_0

        return self.outputs
    
    def ddim_loss_2(self, gt_depth, refine_module_inputs, blur_depth_t, weight, **kwargs):

        noise = torch.randn(blur_depth_t.shape).to(blur_depth_t.device)
        bs = blur_depth_t.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler_2.num_train_timesteps, (bs,), device=gt_depth.device).long()

        noisy_images = self.scheduler_2.add_noise(blur_depth_t, noise, timesteps)

        noise_pred = self.model_2(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss
    
    def ddim_loss_1(self, gt_depth, refine_module_inputs, blur_depth_t, weight, **kwargs):

        noise = torch.randn(blur_depth_t.shape).to(blur_depth_t.device)
        bs = blur_depth_t.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler_1.num_train_timesteps, (bs,), device=gt_depth.device).long()

        noisy_images = self.scheduler_1.add_noise(blur_depth_t, noise, timesteps)

        noise_pred = self.model_1(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss
    
    def ddim_loss_0(self, gt_depth, refine_module_inputs, blur_depth_t, weight, **kwargs):

        noise = torch.randn(blur_depth_t.shape).to(blur_depth_t.device)
        bs = blur_depth_t.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler_0.num_train_timesteps, (bs,), device=gt_depth.device).long()

        noisy_images = self.scheduler_0.add_noise(blur_depth_t, noise, timesteps)

        noise_pred = self.model_0(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss
    

class CNNDDIMPipiline_2:
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
        initial_noise = image
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
        return image, pred, initial_noise
        # return {'images': image}

class CNNDDIMPipiline_1:
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
            ini_noise,
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

        # image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)
        image = ini_noise
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
        return image, pred
        # return {'images': image}

class CNNDDIMPipiline_0:
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
            ini_noise,
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

        # image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)
        image = ini_noise
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
        return image, pred
        # return {'images': image}


class ScheduledCNNRefine(nn.Module):
    def __init__(self, channels_in, channels_noise, **kwargs):
        super().__init__(**kwargs)
        #噪声嵌入
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(channels_noise, 16, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(4, 16),
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


