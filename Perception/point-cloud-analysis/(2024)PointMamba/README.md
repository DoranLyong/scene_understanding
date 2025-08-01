<div  align="center">    
 <img src="./figure/logo.png" width = "100"  align=center />
</div>



<div align="center">
<h1>PointMamba</h1>
<h3>A Simple State Space Model for Point Cloud Analysis</h3>


[Dingkang Liang](https://dk-liang.github.io/)<sup>1</sup> \*, [Xin Zhou](https://lmd0311.github.io/)<sup>1</sup> \*, [Wei Xu](https://scholar.google.com/citations?user=oMvFn0wAAAAJ&hl=en)<sup>1</sup>, [Xingkui Zhu](https://scholar.google.com/citations?user=wKKiNQkAAAAJ&hl=en)<sup>1</sup>, [Zhikang Zou](https://bigteacher-777.github.io/)<sup>2</sup>, [Xiaoqing Ye](https://shuluoshu.github.io/)<sup>2</sup>, [Xiao Tan](https://tanxchong.github.io/)<sup>2</sup> and [Xiang Bai](https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=en)<sup>1†</sup>

<sup>1</sup>  Huazhong University of Science & Technology, <sup>2</sup>  Baidu Inc.

(\*) Equal contribution. ($\dagger$) Corresponding author.

[![arXiv](https://img.shields.io/badge/Arxiv-2402.10739-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.10739)
[![Project](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://lmd0311.github.io/projects/PointMamba/)
[![Zhihu](https://img.shields.io/badge/Blog-zhihu(中文)-blue.svg)](https://zhuanlan.zhihu.com/p/687191399)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLMD0311%2FPointMamba&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/LMD0311/PointMamba?color=success&label=Issues)](https://github.com/LMD0311/PointMamba/issues?q=is%3Aissue+is%3Aclosed) 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

</div>

## 📣 News

- **[12/Mar/2025]** Update the code. The earlier version is available [here](https://github.com/LMD0311/PointMamba/tree/old).
- **[11/Oct/2024]** 🚀 Check out our latest work **[PointGST](https://github.com/jerryfeng2003/PointGST)** which achieves **99.48%**, **97.76%**, and **96.18%** overall accuracy on the ScanObjNN OBJ_BG, OBJ_OBLY, and PB_T50_RS datasets, respectively.
- **[26/Sep/2024]** PointMamba is accepted to **NeurIPS 2024**! 🥳🥳🥳
- **[30/May/2024]** **Update!** We update the architecture and performance. Please check our **latest paper** and compare it with the **new results**. **Code and weight will be updated soon**.
- **[01/Apr/2024]** ScanObjectNN with further data augmentation is now available, check it out!
- **[16/Mar/2024]** The configurations and checkpoints for ModelNet40 are now accessible, check it out!
- **[05/Mar/2024]** Our paper **DAPT ([github](https://github.com/LMD0311/DAPT))** has been accepted by **CVPR 2024**! 🥳🥳🥳 Check it out and give it a star 🌟!
- **[16/Feb/2024]** Release the [paper](https://arxiv.org/abs/2402.10739).

## Abstract

Transformers have become one of the foundational architectures in point cloud analysis tasks due to their excellent global modeling ability. However, the attention mechanism has quadratic complexity, making the design of a linear complexity method with global modeling appealing. In this paper, we propose **PointMamba**, transferring the success of Mamba, a recent representative state space model (SSM), from NLP to point cloud analysis tasks. Unlike traditional Transformers, **PointMamba** employs a linear complexity algorithm, presenting global modeling capacity while significantly reducing computational costs. Specifically, our method leverages space-filling curves for effective point tokenization and adopts an extremely simple, non-hierarchical Mamba encoder as the backbone. Comprehensive evaluations demonstrate that **PointMamba** achieves superior performance across multiple datasets while significantly reducing GPU memory usage and FLOPs. This work underscores the potential of SSMs in 3D vision-related tasks and presents a simple yet effective Mamba-based baseline for future research.

## Overview

<div  align="center">    
 <img src="./figure/pipeline.png" width = "888"  align=center />
</div>


## Main Results

<div  align="center">    
 <img src="./figure/scanobj.png" width = "888"  align=center />
</div>

| Task | Dataset | Config | Acc.   | Download (ckpt/log)                                                                                                                                                                              |
| :---- | :---- | :---- |:-------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Pre-training | ShapeNet | [pretrain.yaml](./cfgs/pretrain.yaml) | N.A.   | [ckpt](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/pretrain.pth)      |
| Classification | ModelNet40 | [finetune_modelnet.yaml](./cfgs/finetune_modelnet.yaml) | 93.6%  | [ckpt](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/finetune_modelnet.pth)                                                                                                    |
| Classification | ScanObjectNN | [finetune_scan_objbg.yaml](./cfgs/finetune_scan_objbg.yaml) | 94.32% | [ckpt](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/finetune_scan_objbg.pth) / [log](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/finetune_scan_objbg.log) |
| Classification | ScanObjectNN | [finetune_scan_objonly.yaml](./cfgs/finetune_scan_objonly.yaml) | 92.60% | [ckpt](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/finetune_scan_only.pth) / [log](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/finetune_scan_only.log)   |
| Classification | ScanObjectNN | [finetune_scan_hardest.yaml](./cfgs/finetune_scan_hardest.yaml) | 89.31% | [ckpt](https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/finetune_scan_hardest.pth)                                                                                                   |

## Getting Started

### Datasets

See [DATASET.md](./docs/DATASET.md) for details.

### Usage

See [USAGE.md](./docs/USAGE.md) for details.
* (CUSTOMIZE) See [INSTALL.md](./docs/INSTALL.md)


## To Do

- [x] Release code.
- [x] Release checkpoints.
- [x] ModelNet40.
- [x] Update the code.

## Acknowledgement

This project is based on Point-BERT ([paper](https://arxiv.org/abs/2111.14819), [code](https://github.com/lulutang0608/Point-BERT?tab=readme-ov-file)), Point-MAE ([paper](https://arxiv.org/abs/2203.06604), [code](https://github.com/Pang-Yatian/Point-MAE)), Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)). Thanks for their wonderful works.

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```bibtex
@inproceedings{liang2024pointmamba,
      title={PointMamba: A Simple State Space Model for Point Cloud Analysis}, 
      author={Liang, Dingkang and Zhou, Xin and Xu, Wei and Zhu, Xingkui and Zou, Zhikang and Ye, Xiaoqing and Tan, Xiao and Bai, Xiang},
      booktitle={Advances in Neural Information Processing Systems},
      year={2024}
}
```
