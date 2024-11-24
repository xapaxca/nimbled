# NimbleD
**NimbleD: Enhancing Self-supervised Monocular Depth Estimation with Pseudo-labels and Large-scale Video Pre-training**

European Conference on Computer Vision (ECCV) 2024 CV4Metaverse Workshop - Oral

[Paper](https://arxiv.org/abs/2408.14177)

## Results on [KITTI](https://www.cvlibs.net/datasets/kitti/) Eigen Split with Median Alignment
| Method            | Par  | AbsRel | SqRel | RMSE  | RMSElog | δ < 1.25<sup>1</sup> | δ < 1.25<sup>2</sup> | δ < 1.25<sup>3</sup> |
| ----------------- | ---- | ------ | ----- | ----- | ------- | -------------------- | -------------------- | -------------------- |
| Monodepth2-R18    | 14.8 | 0.115  | 0.903 | 4.863 | 0.193   | 0.877                | 0.959                | 0.981                |
|  + NimbleD (ours) | 14.8 | 0.100  | 0.739 | 4.440 | 0.175   | 0.898                | 0.967                | 0.985                |
| Monodepth2-R50    | 34.6 | 0.110  | 0.831 | 4.642 | 0.187   | 0.883                | 0.962                | 0.982                |
|  + NimbleD (ours) | 34.6 | 0.097  | 0.721 | 4.377 | 0.172   | 0.904                | 0.968                | 0.985                |
| SwiftDepth-S      | 3.6  | 0.110  | 0.830 | 4.700 | 0.187   | 0.882                | 0.962                | 0.982                |
|  + NimbleD (ours) | 3.6  | 0.098  | 0.733 | 4.401 | 0.174   | 0.901                | 0.968                | 0.985                |
| SwiftDepth        | 6.4  | 0.107  | 0.790 | 4.643 | 0.182   | 0.888                | 0.963                | 0.983                |
|  + NimbleD (ours) | 6.4  | 0.096  | 0.697 | 4.333 | 0.171   | 0.905                | 0.969                | 0.986                |
| LiteMono-small    | 2.5  | 0.110  | 0.802 | 4.671 | 0.186   | 0.879                | 0.961                | 0.982                |
|  + NimbleD (ours) | 2.5  | 0.099  | 0.709 | 4.370 | 0.172   | 0.898                | 0.967                | 0.986                |
| LiteMono          | 3.1  | 0.107  | 0.765 | 4.561 | 0.183   | 0.886                | 0.963                | 0.983                |
|  + NimbleD (ours) | 3.1  | 0.096  | 0.684 | 4.304 | 0.171   | 0.903                | 0.969                | 0.986                |
| LiteMono-8M       | 8.8  | 0.101  | 0.729 | 4.454 | 0.178   | 0.897                | 0.965                | 0.983                |
|  + NimbleD (ours) | 8.8  | 0.092  | 0.646 | 4.194 | 0.165   | 0.910                | 0.970                | 0.986                |


## Setup

Experiments were conducted on Windows 11, Python 3.9.19, CUDA 12.5, PyTorch 1.13.1.

Main dependencies are listed in the [requirements.txt](https://github.com/xapaxca/nimbled/blob/main/requirements.txt).

## Datasets
### KITTI
Refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) for KITTI dataset preparation.
- Image Format: PNG  
- Dataset Size: ~161 GB  

**Generate Pseudo-labels**
```shell
python generate_kitti_pseudo_labels.py --data_dir KITTI_DATA_PATH
```
- Pseudo-labels Size: ~13 GB  

### YouTube

The selected YouTube [videos](https://github.com/xapaxca/nimbled/blob/main/datasets/youtube/urls) were accessed and obtained under a CC-BY license at a resolution of 854x480. 

Proof of access under the CC-BY license can be found here: [CC-BY](https://github.com/xapaxca/nimbled/blob/main/datasets/youtube/cc_by).

Please verify the current license and comply with YouTube’s terms of service before using or downloading videos.

Each video must be saved in the following structured format matching the order of their respective [URL files](https://github.com/xapaxca/nimbled/blob/main/datasets/youtube/urls):

```plaintext
datasets/
└── youtube/
    └── videos/
        ├── driving/
        │   ├── D_0001.mp4
        │   ├── D_0002.mp4
        │   ├── ...
        │   └── D_0035.mp4
        ├── hiking/
        │   ├── H_0001.mp4
        │   ├── H_0002.mp4
        │   ├── ...
        │   └── H_0035.mp4
        └── city_walking/
            ├── CW_0001.mp4
            ├── CW_0002.mp4
            ├── ...
            └── CW_0035.mp4
```

- Videos Size: ~31 GB  

**Extract Frames**
```shell
python ./datasets/youtube/extract_frames.py
```
- Frames Size: ~556 GB  

**Generate Pseudo-labels**
```shell
python generate_youtube_pseudo_labels.py --data_dir YOUTUBE_DATA_PATH
```
- Pseudo-labels Size: ~364 GB  

## Training

MODEL_NAME:
- md2_r18  
- md2_r50  
- swiftdepth_s  
- swiftdepth  
- litemono_s  
- litemono  
- litemono_8m

**Large-scale Video Pre-Training**
```shell
python pretrain_youtube.py --project_name PROJECT_NAME --model_name MODEL_NAME --data_dir YOUTUBE_DATA_PATH --learn_k
```

**Fine-tune on KITTI**
```shell
python finetune_kitti.py --project_name PROJECT_NAME --model_name MODEL_NAME --data_dir KITTI_DATA_PATH --pretrained_weights PRETRAIN_WEIGHTS_PATH --learn_k
```

## Weights
- [**ImageNet Pre-trained Backbone Weights**](https://drive.google.com/file/d/1rFqeLLk0BjFom1kIbr8AHWG_1pyD8_8y)

- [**YouTube Pre-trained Weights**](https://drive.google.com/file/d/1V05Bs21LkeN8px_3ObTo-c1IGnhizDMv)

- [**KITTI Fine-tuned Weights**](https://drive.google.com/file/d/1nHbILffYFjPu92tEUaZkp9b4c5CE_F6j)

## Evaluation

MODEL_NAME:
- md2_r18  
- md2_r50  
- swiftdepth_s  
- swiftdepth  
- litemono_s  
- litemono  
- litemono_8m

**Evaluate on KITTI Eigen split with median alignment**
```shell
python eval_kitti.py --data_dir KITTI_DATA_PATH --weights_dir WEIGHTS_PATH --model_name MODEL_NAME --eval_split eigen --align median
```

**Evaluate on KITTI Eigen-Benchmark split with lsqr alignment**
```shell
python eval_kitti.py --data_dir KITTI_DATA_PATH --weights_dir WEIGHTS_PATH --model_name MODEL_NAME --eval_split eigen_benchmark --align lsqr
```

**Evaluate on [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)**

**Note:** I realized that the evaluation set I used contains a slightly different set of images compared to those used in other papers. Unfortunately, I no longer remember how I generated or where I downloaded these images.

To address this, I have uploaded the evaluation set I used in the paper [here](https://drive.google.com/file/d/1GgzEs-CPa4WX73yrU_x1GIel5KGIz1i0). Since this dataset is intended solely for a head-to-head comparison of improvements over baselines tested on the same data, the comparison remains fair. If you perform evaluations on the standard test set, you will observe only minor differences.

```shell
python eval_nyuv2.py --data_dir NYUv2_DATA_PATH --weights_dir WEIGHTS_PATH --model_name MODEL_NAME --align median
```

**Evaluate on [Make3D](http://make3d.cs.cornell.edu/data.html)**
```shell
python eval_make3d.py --data_dir MAKE3D_DATA_PATH --weights_dir WEIGHTS_PATH --model_name MODEL_NAME
```

## Acknowledgement

The code is inspired by and builds upon the following works: [Monodepth2](https://github.com/nianticlabs/monodepth2), [Lite-Mono](https://github.com/noahzn/Lite-Mono), [SwiftDepth](https://github.com/xapaxca/swiftdepth), [KBR](https://github.com/jspenmar/slowtv_monodepth), [DepthAnything](https://github.com/LiheYoung/Depth-Anything). 

Thank you to the authors for their valuable contributions.

## Attribution

The YouTube videos were accessed under a CC-BY license at the time of collection. The list of video URLs is available [here](https://github.com/xapaxca/nimbled/blob/main/datasets/youtube/urls).

The following creators are acknowledged for their content:

- **Kizzume**
  - [YouTube Channel](https://www.youtube.com/@kizzume)
  - Channel ID: `UCPJJsmyvEFizmsVKznk_pjw`

- **Evan Explores**
  - [YouTube Channel](https://www.youtube.com/@Evan-Explores)
  - Channel ID: `UCqsCOd3o-7vDXFmYwP00fjg`

- **Travel | Relax | Listen**
  - [YouTube Channel](https://www.youtube.com/@TravelRelaxListen)
  - Channel ID: `UCwR7sfacuPghWn-KvVwxOeg`

- **POPtravel**
  - Daniel Sczepansky
  - [YouTube Channel](https://www.youtube.com/@poptravelorg)
  - Channel ID: `UClODDXeUIz1-FaKyN8dsNrA`
  - Website: [www.poptravel.org](https://www.poptravel.org)

Their works are highly appreciated.
