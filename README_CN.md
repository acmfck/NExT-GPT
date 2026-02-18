# <img src="NExT-GPT-Lagacy/code/nextgpt.png" style="width: 5%"> NExT-GPT：任意到任意多模态大模型
[Shengqiong Wu](https://chocowu.github.io/), [Hao Fei](http://haofei.vip/)*, [Leigang Qu](#), [Wei Ji](https://jiwei0523.github.io/), and [Tat-Seng Chua](https://www.chuatatseng.com/).
(*通讯作者)

**ICML 2024，Oral 论文**

**[NExT++ Research Center](https://www.nextcenter.org/)，新加坡国立大学计算机学院**

-----

<a href='https://next-gpt.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='#'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<a href='https://arxiv.org/pdf/2309.05519'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
![License](https://img.shields.io/badge/License-BSD-blue.svg)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=aqw2SCWeWD0)


本仓库提供 **NExT-GPT** 的代码、数据与模型权重。NExT-GPT 是首个端到端的多模态大模型（MM-LLM），能够以任意组合（任意到任意）的文本、图像、视频与音频等作为输入并生成输出。


**注意**：我们已将旧版本代码封装在 [NExT-GPT-Lagacy](NExT-GPT-Lagacy) 中。所有训练与调优流程请以新的代码库为准。

-----------

## 🎉 新闻 

- [x] [2023.09.15] 🚀🚀 发布 NExT-GPT 版本 `7b_tiva_v0` 的代码。
- [x] [2023.09.27] 🔨🧩 增加了模态混合的 batch sampler。
- [x] [2023.10.01] 📢📢 发布 T2M 指令数据集。
- [x] [2023.10.04] 👏👏 发布 NExT-GPT 版本 [7b_tiva_v0](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0) 的 checkpoint。
- [x] [2023.10.15] 🔨🚀 更新 NExT-GPT 版本 [7b_tiva_v0](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0)。
- [x] [2024.10.07] 👏👏 发布数据与对应构建方法，详见 [DATA_README.md](data/DATA_README.md)。


## 👉 TODO 
- [ ] 更新更多类型与规模的 LLM。
- [ ] 扩展更多输入与输出模态。
- [ ] ...



-----------

## 示例演示
这里展示 NExT-GPT 生成的示例。
更多示例请访问 [项目主页](https://next-gpt.github.io/) 或在线 [Demo](https://acc414b22d6839d28f.gradio.live)。 


https://github.com/NExT-GPT/NExT-GPT/assets/18722770/0c2b3d88-a533-4899-ab44-65580fe54538


https://github.com/NExT-GPT/NExT-GPT/assets/18722770/eb1319a6-38aa-4546-a96e-163207e7de93


https://github.com/NExT-GPT/NExT-GPT/assets/18722770/36bec0ad-9bad-4bcf-bc37-92b028f1bc6a



<span id='introduction'/>

## 简要介绍 


NExt-GPT 基于已有的预训练 LLM、多模态编码器与先进扩散模型，并进行了充分的端到端指令微调。

<p align="center" width="100%">
<a target="_blank"><img src="figures/framework.png" alt="Video-LLaMA" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

- **多模态编码阶段**：利用成熟的编码器对不同模态的输入进行编码，并通过投影层将其映射为 LLM 可理解的语言式表示。
- **LLM 理解与推理阶段**：以开源 LLM 作为核心处理输入信息并进行语义理解与推理。LLM 不仅生成文本 token，还会输出独特的“模态信号”token，用以指示解码层是否以及生成何种模态内容。
- **多模态生成阶段**：根据 LLM 发出的模态信号，基于 Transformer 的输出投影层将信号 token 表示映射到下游多模态解码器可理解的表示。


更多技术细节请参考 [论文](https://arxiv.org/pdf/2309.05519.pdf)。 


-----------


<span id='Usage'/>

## 快速开始



<span id='all_catelogue'/>

### 目录：
* <a href='#Code Structure'>1. 代码结构</a>
* <a href='#Environment Preparation'>2. 环境准备</a>
* <a href='#Training on Your Own'>3. 使用自有数据训练/适配 NExt-GPT</a>
  * <a href='#Prepare Pre-trained Checkpoint'>3.1 准备预训练检查点</a>
  * <a href='#Prepare Dataset'>3.2 准备数据集</a>
  * <a href='#Precompute Embeddings'>3.3 预计算 Embedding</a>
* <a href='#Train NExT-GPT'>3.4 训练 NExT-GPT</a>
* <a href='#Run NExT-GPT System'>4. 运行 NExT-GPT 系统</a>
  * <a href='#Prepare checkpoints'>4.1 准备检查点</a>
  * <a href='#Deploy Demo System'>4.2 部署 Demo 系统</a>
* <a href='#Tuning your own system'>5. 微调你的系统</a>
  * <a href='#Tuning your own dataset'>5.1 数据集</a>
  * <a href='#Tuning your own framework'>5.2 模型框架</a>
  * <a href='#Tuning script'>5.3 微调脚本</a>
 
****




<span id='Code Structure'/>

### 1. 代码结构 

```
.
|-- NExT-GPT-Lagacy       # 模型的旧版本
|-- assets
|-- checkpoints           # 保存预训练与微调检查点
|-- data  
|   |-- IT_data
|   |   |-- MosIT_data
|   |   |-- T+X-T_data    # 文本+[图像/音频/视频] -> 文本 指令数据
|   |   `-- T-T+X_data    # 合成的 文本 -> 文本+[图像/音频/视频] 指令数据
|   |-- T_X_pair_data     # 文本-音频配对数据
|   |   |-- audiocap
|   |   |-- cc3m
|   |   `-- webvid
|   |-- embed 
|   `-- prepare_data.py
|-- figures
|-- merge_lora_weights.py
|-- nextgpt
|   |-- __init__.py
|   |-- constants.py
|   |-- conversation.py
|   |-- dataset
|   |   |-- __init__.py
|   |   |-- audio_processor.py
|   |   |-- base_dataset.py
|   |   |-- catalog.py
|   |   |-- concat_dataset.py
|   |   |-- dataset_utils.py
|   |   `-- sampler.py
|   |-- mm_utils.py
|   |-- model
|   |   |-- __init__.py
|   |   |-- apply_delta.py
|   |   |-- builder.py
|   |   |-- consolidate.py
|   |   |-- language_model
|   |   |-- make_delta.py
|   |   |-- multimodal_decoder
|   |   |-- multimodal_encoder
|   |   |-- multimodal_projector
|   |   |-- nextgpt_arch.py
|   |   `-- utils.py
|   `-- utils.py
|-- scripts
|   |-- finetune.sh
|   |-- pretrain_dec.sh
|   |-- pretrain_enc.sh
|   |-- zero2.json
|   |-- zero3.json
|   `-- zero3_offload.json
|-- LICENSE.md
|-- README.md
|-- nextgpt_trainer.py
|-- predict.py
|-- preprocess_embeddings.py
|-- requirements.txt
|-- train.py
|-- train_mem.py
`-- training_utils.py
```


<span id='Environment Preparation'/>


### 2. 环境准备  <a href='#all_catelogue'>[返回目录]</a>
请先克隆仓库并安装所需环境，可运行如下命令：
```
conda env create -n nextgpt python=3.8

conda activate nextgpt

# CUDA 12.1
conda install pytorch==2.1.2 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

git clone https://github.com/NExT-GPT/NExT-GPT.git
cd NExT-GPT

pip install -r requirements.txt
```

<span id='Training on Your Own'/>

### 3. 使用自有数据训练/适配 NExt-GPT 



<span id='Prepare Pre-trained Checkpoint'/>

#### 3.1. 准备预训练检查点  <a href='#all_catelogue'>[返回目录]</a>
NExT-GPT 基于以下优秀模型进行训练。
请按说明准备检查点。

- `ImageBind`
是统一的图像/视频/音频编码器。预训练权重可从 [这里](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) 下载，版本为 `huge`。之后将 `imagebind_huge.pth` 放到 [[.pretrain_ckpt/imagebind]](./pretrain_ckpt/imagebind)。 
- `Vicuna`：
请从 [[这里]](https://huggingface.co/lmsys/vicuna-7b-v1.5) 获取预训练 Vicuna，并将其放到 [[./pretrain_ckpt/vicuna-7b-v1.5/]](./pretrain_ckpt/vicuna-7b-v1.5)。 
- `Image Diffusion`
用于生成图像。NExT-GPT 使用 [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2) ，版本为 `
v2`。（_会自动下载_）
- `Audio Diffusion`
用于生成音频。NExT-GPT 使用 [AudioLDM](https://github.com/haoheliu/AudioLDM) ，版本为 `l-full`。（_会自动下载_）
- `Video Diffusion`
用于视频生成。我们使用 [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) ，版本为 `v2_576w`。（_会自动下载_）



<span id='Prepare Dataset'/>

#### 3.2. 准备数据集  <a href='#all_catelogue'>[返回目录]</a>
请下载用于训练的数据集：

A) T-X 配对数据
  - ***文本-图像*** 的 `CC3M`，请按 [[这里]](./data/T-X_pair_data/cc3m/prepare.md) 的说明操作，然后将数据放在 [[./data/T-X_pair_data/cc3m]](./data/T-X_pair_data/cc3m)。
  - ***文本-视频*** 的 `WebVid`，请按 [[说明]](./data/T-X_pair_data/webvid/prepare.md) 操作，文件放在 [[./data/T-X_pair_data/webvid]](./data/T-X_pair_data/webvid)。
  - ***文本-音频*** 的 `AudioCap`，请按 [[说明]](./data/T-X_pair_data/audiocap/prepare.md) 操作，数据放在 [[./data/T-X_pair_data/audiocap]](./data/T-X_pair_data/audiocap)。

B) 指令数据
  - T+X-T
    - ***视觉指令数据*** `LLaVA`，从 [这里](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) 下载，放在 [[./data/IT_data/T+X-T_data/llava]](./data/IT_data/T+X-T_data/llava/)。
    - ***文本指令数据*** `Alpaca`，从 [这里](https://github.com/tatsu-lab/stanford_alpaca) 下载，放在 [[./data/IT_data/T+X-T_data/alpaca/]](data/IT_data/T+X-T_data/alpaca/)。
    - ***视频指令数据*** `VideoChat`，从 [这里](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) 下载，放在 [[./data/IT_data/T+X-T_data/videochat/]](data/IT_data/T+X-T_data/videochat/)。
    
    备注：下载完成后，请运行 `prepare_data.py` 进行预处理。
  - T-X+T (T2M)
    - `T-X+T` 指令数据集（T2M）存放在 [[./data/IT_data/T-T+X_data]](./data/IT_data/T-T+X_data)。
   
  - MosIT
    - 从 [这里]() 下载文件，放在 [[./data/IT_data/MosIT_data/]](./data/IT_data/MosIT_data/)。(_我们正在完善数据并处理版权问题。_) 


<span id='Precompute Embeddings'/>

#### 3.3. 预计算 Embedding <a href='#all_catelogue'>[返回目录]</a>
在解码侧对齐训练中，我们最小化信号 token 与 caption 表示之间的距离。
为节省时间与内存，我们使用各扩散模型内部的文本编码器，为图像、音频和视频 caption 预计算文本 embedding。  

请在后续训练前运行如下命令，生成的 `embedding` 文件会保存到 [[./data/embed]](./data/embed)。
```angular2html
cd ./code/
python preprocess_embeddings.py ../data/T-X_pair_data/cc3m/cc3m_generation.json image ../data/embed/ stabilityai/stable-diffusion-2
```

参数说明：
- args[1]：caption 文件路径；
- args[2]：模态，可为 `image`、`video`、`audio`；
- args[3]：embedding 保存路径；
- args[4]：对应的预训练扩散模型名称。



<span id='Train NExT-GPT'/>

#### 3.4. 训练 NExT-GPT  <a href='#all_catelogue'>[返回目录]</a>

首先请参考 [[training_utils.py]](training_utils.py) 了解整体模块与数据配置，数据集配置详见 [nextgpt/dataset/catalog.py](nextgpt/dataset/catalog.py)。
NExT-GPT 的训练包含 3 个步骤：

- **Step-1**：编码侧以 LLM 为中心的多模态对齐。该阶段训练 ***输入投影层***，冻结 ImageBind、LLM 与输出投影层。
  ```angular2html
  # 编码侧 LLM 中心的多模态对齐
  bash scripts/pretrain_enc.sh
  ```



- **Step-2**：解码侧指令对齐。该阶段训练 ***输出投影层***，冻结 ImageBind、LLM 与输入投影层。
  ```angular2html
  # 编码侧 LLM 中心的多模态对齐
  bash scripts/pretrain_enc.sh
  ```




- **Step-3**：指令微调。该阶段对 1) ***LLM***（使用 LoRA）、2) ***输入投影层*** 与 3) ***输出投影层*** 在指令数据集上进行微调。
  ```angular2html
  # 编码侧 LLM 中心的多模态对齐
  bash scripts/pretrain_enc.sh
  ```




<span id='Run NExT-GPT System'/>

## 4. 运行 NExT-GPT 系统 <a href='#all_catelogue'>[返回目录]</a>


<span id='Prepare checkpoints'/>


#### 4.1. 准备检查点

首先加载预训练的 NExT-GPT 系统。
- **Step-1**：加载 `冻结参数`。参见 <a href='#Prepare Pre-trained Checkpoint'>3.1 准备预训练检查点</a>。

- **Step-2**：加载 `可训练参数`。请将 NExT-GPT 系统放在 [./checkpoints/nextgpt-v1.5-7b](./checkpoints/nextgpt-v1.5-7b)。你可以 1) 使用自行训练的参数，或 2) 从 [Huggingface]() 下载我们的检查点。


#### 4.2. 运行预测
完成检查点加载后，可运行预测：
```angular2html
python predict.py
```

---------


<span id='Tuning your own system'/>

## 5. 微调你的系统 <a href='#all_catelogue'>[返回目录]</a>


<span id='Tuning your own dataset'>

#### 5.1. 数据集
你可以自定义数据集，请参考 [base_dataset.py](nextgpt/dataset/base_dataset.py)，并在 [catalog.py]([text](nextgpt/dataset/catalog.py)) 中添加数据集 `catalog`，包括 `target` 与 `parameters`。


<span id='Tuning your own framework'>

#### 5.2. 模型框架
- *多模态编码器*：你可以在 [multimodal encoder 目录](nextgpt/model/multimodal_encoder) 中接入自定义编码器，并在 [builder.py](nextgpt/model/multimodal_encoder/builder.py) 中添加相应代码。
- *多模态解码器*：你可以在 [multimodal decoder 目录](nextgpt/model/multimodal_decoder) 中接入自定义解码器，并在 [builder.py](nextgpt/model/multimodal_decoder/builder.py) 中修改相应代码。
- *投影器*：你可以在 [multimodal projector](nextgpt/model/multimodal_projector/builder.py) 中设计自定义输入/输出投影器。  


<span id='Tuning script'>

#### 5.3. 微调

你可以在 [training_utils.py](training_utils.py) 中预定义模型、数据与训练参数。
微调脚本参考 [finetune.sh](scripts/finetune.sh)。



---------



## 联系方式

如有问题或反馈，欢迎联系 [Shengqiong Wu](mailto:swu@u.nus.edu) 与 [Hao Fei](mailto:haofei37@nus.edu.sg)。


## 引用

如果 NExT-GPT 对你的研究或应用有帮助，欢迎引用：
```
@inproceedings{wu24next,
  title={{NE}x{T}-{GPT}: Any-to-Any Multimodal {LLM}},
  author={Wu, Shengqiong and Fei, Hao and Qu, Leigang and Ji, Wei and Chua, Tat-Seng},
  booktitle={Proceedings of the International Conference on Machine Learning},
  pages = {53366--53397},
  year={2024}
}
```




## 致谢
你可以参考作为本框架与代码仓库基础的相关工作，
[Vicuna](https://github.com/lm-sys/FastChat)，
[ImageBind](https://github.com/facebookresearch/ImageBind)，
[Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)，
[AudioLDM](https://github.com/haoheliu/AudioLDM)，以及
[Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w)。
我们也部分借鉴了以下工作：
[PandaGPT](https://github.com/yxuansu/PandaGPT)，  
[GILL](https://github.com/kohjingyu/gill/)， 
[CoDi](https://codi-gen.github.io/)，
[Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)，
[LLaVA](https://github.com/haotian-liu/LLaVA)，
以及 [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)。
感谢这些优秀工作的贡献。




## 许可说明
本仓库采用 [BSD 3-Clause License](LICENSE.txt)。
NExT-GPT 为研究项目，仅限非商业用途。
严禁将 NExT-GPT 代码用于任何非法、有害、暴力、种族主义或色情用途。
严禁从事任何可能违反上述规范的活动。
任何潜在的商业用途需获得作者批准。
