追加のREADMEの[日本語版は下にあります。](#このリポジトリについて)

## About

This code is for video generation on GPUs with less than 24GB of VRAM using Block Swap.

## Environment Setup

Create a new venv and install the latest PyTorch and TorchVision (verified to work with 2.5.1).

Install the required packages using `requirements_opt.txt` (torchvision, pandas, gradio, etc. are commented out).

Install SageAttention according to [this](https://www.reddit.com/r/StableDiffusion/comments/1h7hunp/how_to_run_hunyuanvideo_on_a_single_24gb_vram_card/?rdt=36679). (You may need to update the Microsoft Visual C++ redistributable package.)

## Download the Model

Download the model according to the official README and place it in any directory as follows:

```shell
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

## Inference

The following is an example of generating a 960x544, 129-frame video on a 24GB VRAM GPU.

```shell
python generate_video_optimized.py --model-base /path/to/ckpts --fp8 
--video-size 544 960 --video-length 129 --infer-steps 30  --prompt "A cat walks on the grass, realistic style." 
--flow-reverse --save-path path/to/results --attn-mode sageattn --output-type video 
--blocks-to-swap 20 --img-in-txt-in-offloading
```

Specify the directory of the downloaded model with `--model-base` (by keeping the directory structure above, you don't need to specify `--dit-weight`).

Specify `--fp8` to reduce the memory usage by converting the DiT weights to float8_e4m3fn.

Specify the attention implementation to use with `--attn-mode`. You can specify `sageattn` or `flash` (Flash Attention 2 is required).

Specify the output type with `--output-type`. You can specify `video`, `latent`, or `both`.

`--blocks-to-swap` specifies the number of blocks to offload to the CPU (Block Swap). The maximum is 38. Do not specify `--use-cpu-offload`.

Specify `--img-in-txt-in-offloading` to offload `img_in` and `txt_in` to the CPU.

If your VRAM is less than 24GB, you can use the `--vae-chunk-size` option to reduce the memory usage of the VAE, like `--vae-chunk-size 16`. Consider using the existing `--vae-tiling` option.

`--latent-path` option is also available to decode saved latents only.

Other options are the same as `sample_video.py`.

For 24GB VRAM, with `--block-to-swap 38` specified, 1280x720 seems to be the limit at 109 frames.

The original README is below.

----

## このリポジトリについて

Block Swapを使用して、24GB以下のVRAMのGPUで動画生成を行うためのコードです。

## 環境整備

新しくvenvを作成します。最新のPyTorchとTorchVisionをインストールします（2.5.1で動作確認済み）。

`requirements_opt.txt`を使用して、必要なパッケージをインストールします（torchvisionとpandas、gradio等をコメントアウトしています）。

[こちら](https://www.reddit.com/r/StableDiffusion/comments/1h7hunp/how_to_run_hunyuanvideo_on_a_single_24gb_vram_card/?rdt=36679)を参考にSageAttentionをインストールします。（Microsoft Visual C++ 再頒布可能パッケージを最新にする必要があるかもしれません。）

※環境整備に関する質問にはお答えできません。

## モデルのダウンロード

公式のREADMEを参考にダウンロードし、任意のディレクトリに以下のように配置します。

```shell
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

## 推論の実施

以下は24GB VRAMのGPUで960x544、129フレームの動画を生成する場合の例です。

```shell
python generate_video_optimized.py --model-base /path/to/ckpts --fp8 
--video-size 544 960 --video-length 129 --infer-steps 30  --prompt "A cat walks on the grass, realistic style." 
--flow-reverse --save-path path/to/results --attn-mode sageattn --output-type video 
--blocks-to-swap 20 --img-in-txt-in-offloading
```

`--model-base`にはダウンロードしたモデルのディレクトリを指定します（上のディレクトリ構成にしておくことで`--dit-weight`の指定は不要）。

`--fp8`を指定するとDiTの重みをfloat8_e4m3fnにして省メモリ化します。

`--attn-mode`には使用するattentionの実装を指定します。`sageattn`、`flash`（Flash Attention 2が必要）が指定できます。

`--output-type`には`video`、`latent`、`both`が指定できます。

`--blocks-to-swap`はCPUへoffloading（Block Swap）するブロック数を指定します。最大38です。`--use-cpu-offload`は指定しないでください。

`--img-in-txt-in-offloading`を指定すると`img_in`と`txt_in`をCPUにオフロードします。

VRAMが24GBより少ない場合、`--vae-chunk-size 16`のように`--vae-chunk-size`オプションを指定してVAEの省メモリ化を行うことができます。元からある`--vae-tiling`オプションの利用も検討してください。

保存したlatentのデコードのみを行う`--latent-path`オプションもあります。

他のオプションは`sample_video.py`と同じです。

24GB VRAMの場合、`--block-to-swap 38`指定時、1280x720では109フレームが限界のようです。

以下はオリジナルのREADMEです。

---- 

<!-- ## **HunyuanVideo** -->

[中文阅读](./README_zh.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/logo.png"  height=100>
</p>

# HunyuanVideo: A Systematic Framework For Large Video Generation Model

<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo Code&message=Github&color=blue"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green"></a>
</div>
<div align="center">
  <a href="https://arxiv.org/abs/2412.03603"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com/hunyuanvideo.pdf"><img src="https://img.shields.io/static/v1?label=Tech Report&message=High Quality Version (~350M)&color=red"></a>
</div>
<div align="center">
  <a href="https://huggingface.co/tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-PromptRewrite"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-PromptRewrite&message=HuggingFace&color=yellow"></a>

 [![Replicate](https://replicate.com/zsxkib/hunyuan-video/badge)](https://replicate.com/zsxkib/hunyuan-video)
</div>

<p align="center">
    👋 Join our <a href="assets/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/GpARqvrh" target="_blank">Discord</a> 
</p>
<p align="center">

-----

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring HunyuanVideo. You can find more visualizations on our [project page](https://aivideo.hunyuan.tencent.com).

> [**HunyuanVideo: A Systematic Framework For Large Video Generation Model**](https://arxiv.org/abs/2412.03603) <be>


## 🎥 Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/f37925a3-7d42-40c9-8a9b-5a010c7198e2" width="50%">
</div>

The video is heavily compressed due to compliance of GitHub policy. The high quality version can be downloaded from [here](https://aivideo.hunyuan.tencent.com/download/HunyuanVideo/material/demo.mov).

## 🔥🔥🔥 News!!
* Dec 7, 2024: 🚀 We release the parallel inference code for HunyuanVideo powered by [xDiT](https://github.com/xdit-project/xDiT).
* Dec 3, 2024: 🤗 We release the inference code and model weights of HunyuanVideo.

## 📑 Open-source Plan

- HunyuanVideo (Text-to-Video Model)
  - [x] Inference 
  - [x] Checkpoints
  - [x] Multi-gpus Sequence Parallel inference (Faster inference speed on more gpus)
  - [x] Web Demo (Gradio) 
  - [ ] Penguin Video Benchmark
  - [ ] ComfyUI
  - [ ] Diffusers 
  - [ ] Multi-gpus PipeFusion inference (Low memory requirmenets)
- HunyuanVideo (Image-to-Video Model)
  - [ ] Inference 
  - [ ] Checkpoints 

## Contents
- [HunyuanVideo: A Systematic Framework For Large Video Generation Model](#hunyuanvideo-a-systematic-framework-for-large-video-generation-model)
  - [🎥 Demo](#-demo)
  - [🔥🔥🔥 News!!](#-news)
  - [📑 Open-source Plan](#-open-source-plan)
  - [Contents](#contents)
  - [**Abstract**](#abstract)
  - [**HunyuanVideo Overall Architecture**](#hunyuanvideo-overall-architecture)
  - [🎉 **HunyuanVideo Key Features**](#-hunyuanvideo-key-features)
    - [**Unified Image and Video Generative Architecture**](#unified-image-and-video-generative-architecture)
    - [**MLLM Text Encoder**](#mllm-text-encoder)
    - [**3D VAE**](#3d-vae)
    - [**Prompt Rewrite**](#prompt-rewrite)
  - [📈 Comparisons](#-comparisons)
  - [📜 Requirements](#-requirements)
  - [🛠️ Dependencies and Installation](#️-dependencies-and-installation)
    - [Installation Guide for Linux](#installation-guide-for-linux)
  - [🧱 Download Pretrained Models](#-download-pretrained-models)
  - [🔑 Single-gpu Inference](#-single-gpu-inference)
    - [Using Command Line](#using-command-line)
    - [Run a Gradio Server](#run-a-gradio-server)
    - [More Configurations](#more-configurations)
  - [🚀 Parallel Inference on Multiple GPUs by xDiT](#-parallel-inference-on-multiple-gpus-by-xdit)
    - [Install Dependencies Compatible with xDiT](#install-dependencies-compatible-with-xdit)
    - [Using Command Line](#using-command-line-1)
  - [🔗 BibTeX](#-bibtex)
  - [🧩 Projects that use HunyuanVideo](#-projects-that-use-hunyuanvideo)
  - [Acknowledgements](#acknowledgements)
  - [Star History](#star-history)
---

## **Abstract**
We present HunyuanVideo, a novel open-source video foundation model that exhibits performance in video generation that is comparable to, if not superior to, leading closed-source models. In order to train HunyuanVideo model, we adopt several key technologies for model learning, including data curation, image-video joint model training, and an efficient infrastructure designed to facilitate large-scale model training and inference. Additionally, through an effective strategy for scaling model architecture and dataset, we successfully trained a video generative model with over 13 billion parameters, making it the largest among all open-source models. 

We conducted extensive experiments and implemented a series of targeted designs to ensure high visual quality, motion diversity, text-video alignment, and generation stability. According to professional human evaluation results, HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3, Luma 1.6, and 3 top-performing Chinese video generative models. By releasing the code and weights of the foundation model and its applications, we aim to bridge the gap between closed-source and open-source video foundation models. This initiative will empower everyone in the community to experiment with their ideas, fostering a more dynamic and vibrant video generation ecosystem. 

## **HunyuanVideo Overall Architecture**

HunyuanVideo is trained on a spatial-temporally
compressed latent space, which is compressed through a Causal 3D VAE. Text prompts are encoded
using a large language model, and used as the conditions. Taking Gaussian noise and the conditions as
input, our generative model produces an output latent, which is then decoded to images or videos through
the 3D VAE decoder.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/overall.png"  height=300>
</p>

## 🎉 **HunyuanVideo Key Features**
### **Unified Image and Video Generative Architecture**
HunyuanVideo introduces the Transformer design and employs a Full Attention mechanism for unified image and video generation. 
Specifically, we use a "Dual-stream to Single-stream" hybrid model design for video generation. In the dual-stream phase, video and text
tokens are processed independently through multiple Transformer blocks, enabling each modality to learn its own appropriate modulation mechanisms without interference. In the single-stream phase, we concatenate the video and text
tokens and feed them into subsequent Transformer blocks for effective multimodal information fusion.
This design captures complex interactions between visual and semantic information, enhancing
overall model performance.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/backbone.png"  height=350>
</p>

### **MLLM Text Encoder**
Some previous text-to-video models typically use pretrained CLIP and T5-XXL as text encoders where CLIP uses Transformer Encoder and T5 uses an Encoder-Decoder structure. In contrast, we utilize a pretrained Multimodal Large Language Model (MLLM) with a Decoder-Only structure as our text encoder, which has the following advantages: (i) Compared with T5, MLLM after visual instruction finetuning has better image-text alignment in the feature space, which alleviates the difficulty of instruction following in diffusion models; (ii)
Compared with CLIP, MLLM has been demonstrated superior ability in image detail description
and complex reasoning; (iii) MLLM can play as a zero-shot learner by following system instructions prepended to user prompts, helping text features pay more attention to key information. In addition, MLLM is based on causal attention while T5-XXL utilizes bidirectional attention that produces better text guidance for diffusion models. Therefore, we introduce an extra bidirectional token refiner to enhance text features.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/text_encoder.png"  height=275>
</p>

### **3D VAE**
HunyuanVideo trains a 3D VAE with CausalConv3D to compress pixel-space videos and images into a compact latent space. We set the compression ratios of video length, space and channel to 4, 8 and 16 respectively. This can significantly reduce the number of tokens for the subsequent diffusion transformer model, allowing us to train videos at the original resolution and frame rate.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/assets/3dvae.png"  height=150>
</p>

### **Prompt Rewrite**
To address the variability in linguistic style and length of user-provided prompts, we fine-tune the [Hunyuan-Large model](https://github.com/Tencent/Tencent-Hunyuan-Large) as our prompt rewrite model to adapt the original user prompt to model-preferred prompt.

We provide two rewrite modes: Normal mode and Master mode, which can be called using different prompts. The prompts are shown [here](hyvideo/prompt_rewrite.py). The Normal mode is designed to enhance the video generation model's comprehension of user intent, facilitating a more accurate interpretation of the instructions provided. The Master mode enhances the description of aspects such as composition, lighting, and camera movement, which leans towards generating videos with a higher visual quality. However, this emphasis may occasionally result in the loss of some semantic details. 

The Prompt Rewrite Model can be directly deployed and inferred using the [Hunyuan-Large original code](https://github.com/Tencent/Tencent-Hunyuan-Large). We release the weights of the Prompt Rewrite Model [here](https://huggingface.co/Tencent/HunyuanVideo-PromptRewrite).

## 📈 Comparisons

To evaluate the performance of HunyuanVideo, we selected five strong baselines from closed-source video generation models. In total, we utilized 1,533 text prompts, generating an equal number of video samples with HunyuanVideo in a single run. For a fair comparison, we conducted inference only once, avoiding any cherry-picking of results. When comparing with the baseline methods, we maintained the default settings for all selected models, ensuring consistent video resolution. Videos were assessed based on three criteria: Text Alignment, Motion Quality, and Visual Quality. More than 60 professional evaluators performed the evaluation. Notably, HunyuanVideo demonstrated the best overall performance, particularly excelling in motion quality. Please note that the evaluation is based on Hunyuan Video's high-quality version. This is different from the currently released fast version.

<p align="center">
<table> 
<thead> 
<tr> 
    <th rowspan="2">Model</th> <th rowspan="2">Open Source</th> <th>Duration</th> <th>Text Alignment</th> <th>Motion Quality</th> <th rowspan="2">Visual Quality</th> <th rowspan="2">Overall</th>  <th rowspan="2">Ranking</th>
</tr> 
</thead> 
<tbody> 
<tr> 
    <td>HunyuanVideo (Ours)</td> <td> ✔ </td> <td>5s</td> <td>61.8%</td> <td>66.5%</td> <td>95.7%</td> <td>41.3%</td> <td>1</td>
</tr> 
<tr> 
    <td>CNTopA (API)</td> <td> &#10008 </td> <td>5s</td> <td>62.6%</td> <td>61.7%</td> <td>95.6%</td> <td>37.7%</td> <td>2</td>
</tr> 
<tr> 
    <td>CNTopB (Web)</td> <td> &#10008</td> <td>5s</td> <td>60.1%</td> <td>62.9%</td> <td>97.7%</td> <td>37.5%</td> <td>3</td>
</tr> 
<tr> 
    <td>GEN-3 alpha (Web)</td> <td>&#10008</td> <td>6s</td> <td>47.7%</td> <td>54.7%</td> <td>97.5%</td> <td>27.4%</td> <td>4</td> 
</tr> 
<tr> 
    <td>Luma1.6 (API)</td><td>&#10008</td> <td>5s</td> <td>57.6%</td> <td>44.2%</td> <td>94.1%</td> <td>24.8%</td> <td>6</td>
</tr>
<tr> 
    <td>CNTopC (Web)</td> <td>&#10008</td> <td>5s</td> <td>48.4%</td> <td>47.2%</td> <td>96.3%</td> <td>24.6%</td> <td>5</td>
</tr> 
</tbody>
</table>
</p>

## 📜 Requirements

The following table shows the requirements for running HunyuanVideo model (batch size = 1) to generate videos:

|     Model    |  Setting<br/>(height/width/frame) | GPU Peak Memory  |
|:------------:|:--------------------------------:|:----------------:|
| HunyuanVideo   |        720px1280px129f          |       60GB        |
| HunyuanVideo   |        544px960px129f           |       45GB        |

* An NVIDIA GPU with CUDA support is required. 
  * The model is tested on a single 80G GPU.
  * **Minimum**: The minimum GPU memory required is 60GB for 720px1280px129f and 45G for 544px960px129f.
  * **Recommended**: We recommend using a GPU with 80GB of memory for better generation quality.
* Tested operating system: Linux

## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/tencent/HunyuanVideo
cd HunyuanVideo
```

### Installation Guide for Linux

We provide an `environment.yml` file for setting up a Conda environment.
Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

We recommend CUDA versions 12.4 or 11.8 for the manual installation.

```shell
# 1. Prepare conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate HunyuanVideo

# 3. Install pip dependencies
python -m pip install -r requirements.txt

# 4. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

In case of running into float point exception(core dump) on the specific GPU type, you may try the following solutions:

```shell
# Option 1: Making sure you have installed CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 (or simply using our CUDA 12 docker image).
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/

# Option 2: Forcing to explictly use the CUDA 11.8 compiled version of Pytorch and all the other packages
pip uninstall -r requirements.txt  # uninstall all packages
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

Additionally, HunyuanVideo also provides a pre-built Docker image. Use the following command to pull and run the docker image.

```shell
# For CUDA 12.4 (updated to avoid float point exception)
docker pull hunyuanvideo/hunyuanvideo:cuda_12
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_12

# For CUDA 11.8
docker pull hunyuanvideo/hunyuanvideo:cuda_11
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_11
```


## 🧱 Download Pretrained Models

The details of download pretrained models are shown [here](ckpts/README.md).

## 🔑 Single-gpu Inference
We list the height/width/frame settings we support in the following table.

|      Resolution       |           h/w=9:16           |    h/w=16:9     |     h/w=4:3     |     h/w=3:4     |     h/w=1:1     |
|:---------------------:|:----------------------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|         540p          |        544px960px129f        |  960px544px129f | 624px832px129f  |  832px624px129f |  720px720px129f |
| 720p (recommended)    |       720px1280px129f        | 1280px720px129f | 1104px832px129f | 832px1104px129f | 960px960px129f  |

### Using Command Line

```bash
cd HunyuanVideo

python3 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results
```

### Run a Gradio Server
```bash
python3 gradio_server.py --flow-reverse

# set SERVER_NAME and SERVER_PORT manually
# SERVER_NAME=0.0.0.0 SERVER_PORT=8081 python3 gradio_server.py --flow-reverse
```

### More Configurations

We list some more useful configurations for easy usage:

|        Argument        |  Default  |                Description                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--prompt`       |   None    |   The text prompt for video generation    |
|     `--video-size`     | 720 1280  |      The size of the generated video      |
|    `--video-length`    |    129    |     The length of the generated video     |
|    `--infer-steps`     |    50     |     The number of steps for sampling      |
| `--embedded-cfg-scale` |    6.0    |    Embeded  Classifier free guidance scale       |
|     `--flow-shift`     |    7.0    | Shift factor for flow matching schedulers |
|     `--flow-reverse`   |    False  | If reverse, learning/sampling from t=1 -> t=0 |
|        `--seed`        |     None  |   The random seed for generating video, if None, we init a random seed    |
|  `--use-cpu-offload`   |   False   |    Use CPU offload for the model load to save more memory, necessary for high-res video generation    |
|     `--save-path`      | ./results |     Path to save the generated video      |


## 🚀 Parallel Inference on Multiple GPUs by xDiT

[xDiT](https://github.com/xdit-project/xDiT) is a Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters.
It has successfully provided low-latency parallel inference solutions for a variety of DiTs models, including mochi-1, CogVideoX, Flux.1, SD3, etc. This repo adopted the [Unified Sequence Parallelism (USP)](https://arxiv.org/abs/2405.07719) APIs for parallel inference of the HunyuanVideo model.

### Install Dependencies Compatible with xDiT

```
# 1. Create a black conda environment
conda create -n hunyuanxdit python==3.10.9
conda activate hunyuanxdit

# 3. Install PyTorch component with CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install pip dependencies
python -m pip install -r requirements_xdit.txt
```

You can skip the above steps and pull the pre-built docker image directly, which is built from [docker/Dockerfile_xDiT](./docker/Dockerfile_xDiT)

```
docker pull thufeifeibear/hunyuanvideo:latest
```

### Using Command Line

For example, to generate a video with 8 GPUs, you can use the following command:

```bash
cd HunyuanVideo

torchrun --nproc_per_node=8 sample_video.py \
    --video-size 1280 720 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --save-path ./results
```

You can change the `--ulysses-degree` and `--ring-degree` to control the parallel configurations for the best performance. The valid parallel configurations are shown in the following table.

<details>
<summary>Supported Parallel Configurations (Click to expand)</summary>

|     --video-size     | --video-length | --ulysses-degree x --ring-degree | --nproc_per_node |
|----------------------|----------------|----------------------------------|------------------|
| 1280 720 or 720 1280 | 129            | 8x1,4x2,2x4,1x8                  | 8                |
| 1280 720 or 720 1280 | 129            | 1x5                              | 5                |
| 1280 720 or 720 1280 | 129            | 4x1,2x2,1x4                      | 4                |
| 1280 720 or 720 1280 | 129            | 3x1,1x3                          | 3                |
| 1280 720 or 720 1280 | 129            | 2x1,1x2                          | 2                |
| 1104 832 or 832 1104 | 129            | 4x1,2x2,1x4                      | 4                |
| 1104 832 or 832 1104 | 129            | 3x1,1x3                          | 3                |
| 1104 832 or 832 1104 | 129            | 2x1,1x2                          | 2                |
| 960 960              | 129            | 6x1,3x2,2x3,1x6                  | 6                |
| 960 960              | 129            | 4x1,2x2,1x4                      | 4                |
| 960 960              | 129            | 3x1,1x3                          | 3                |
| 960 960              | 129            | 1x2,2x1                          | 2                |
| 960 544 or 544 960   | 129            | 6x1,3x2,2x3,1x6                  | 6                |
| 960 544 or 544 960   | 129            | 4x1,2x2,1x4                      | 4                |
| 960 544 or 544 960   | 129            | 3x1,1x3                          | 3                |
| 960 544 or 544 960   | 129            | 1x2,2x1                          | 2                |
| 832 624 or 624 832   | 129            | 4x1,2x2,1x4                      | 4                |
| 624 832 or 624 832   | 129            | 3x1,1x3                          | 3                |
| 832 624 or 624 832   | 129            | 2x1,1x2                          | 2                |
| 720 720              | 129            | 1x5                              | 5                |
| 720 720              | 129            | 3x1,1x3                          | 3                |

</details>


<p align="center">
<table align="center">
<thead>
<tr>
    <th colspan="4">Latency (Sec) for 1280x720 (129 frames 50 steps) on 8xGPU</th>
</tr>
<tr>
    <th>1</th>
    <th>2</th>
    <th>4</th>
    <th>8</th>
</tr>
</thead>
<tbody>
<tr>
    <th>1904.08</th>
    <th>934.09 (2.04x)</th>
    <th>514.08 (3.70x)</th>
    <th>337.58 (5.64x)</th>
</tr>

</tbody>
</table>
</p>

## 🔗 BibTeX
If you find [HunyuanVideo](https://arxiv.org/abs/2412.03603) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv preprint arXiv:2412.03603},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03603}, 
}
```



## 🧩 Projects that use HunyuanVideo

If you develop/use HunyuanVideo in your projects, welcome to let us know.

- ComfyUI (with support for F8 Inference and Video2Video Generation): [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper) by [Kijai](https://github.com/kijai)



## Acknowledgements

We would like to thank the contributors to the [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
Additionally, we also thank the Tencent Hunyuan Multimodal team for their help with the text encoder. 

## Star History
<a href="https://star-history.com/#Tencent/HunyuanVideo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/HunyuanVideo&type=Date" />
 </picture>
</a>
