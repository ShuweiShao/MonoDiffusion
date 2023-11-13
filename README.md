<div align="center">

<h1>MonoDiffusion: Self-Supervised Monocular Depth Estimation Using Diffusion Model</h1>

<div>
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=ecZHSVQAAAAJ' target='_blank'>Shuwei Shao</a><sup>1</sup>&emsp;
    <a target='_blank'>Zhongcai Pei</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=5PoZrcYAAAAJ' target='_blank'>Weihai Chen</a><sup>1</sup>&emsp;
    <a target='_blank'>Dingchi Sun</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=7E0QgKUAAAAJ' target='_blank'>Peter C. Y. Chen</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=LiUX7WQAAAAJ' target='_blank'>Zhengguo Li</a><sup>3</sup>&emsp;
</div>
<div>
    <sup>1</sup>Beihang University, <sup>2</sup>National University of Singapore, <sup>3</sup>A*STAR
</div>


<div>
    <h4 align="center">
        • <a href="https://arxiv.org/abs/2309.14137" target='_blank'>Arxiv 2023</a> •
    </h4>
</div>

<strong>In this work, we introduce a novel self-supervised depth estimation framework, dubbed MonoDiffusion, by formulating it as an iterative denoising process. Because the depth ground-truth is unavailable in the training phase, we develop a pseudo ground-truth diffusion process to assist the diffusion in MonoDiffusion. The pseudo ground-truth diffusion gradually adds noise to the depth map generated by a pre-trained teacher model. Moreover, the teacher model allows applying a distillation loss to guide the denoised depth. Further, we develop a masked visual condition mechanism to enhance the denoising ability of model. Extensive experiments are conducted on the KITTI and Make3D datasets and the proposed MonoDiffusion outperforms prior state-of-the-art competitors. </strong>

<div style="text-align:center">
<img src="assets/teaser.jpg"  width="80%" height="80%">
</div>

---

</div>
The source code is comming!
