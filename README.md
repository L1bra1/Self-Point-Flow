# Self-Point-Flow
This is the PyTorch code for [Self-Point-Flow: Self-Supervised Scene Flow Estimation from Point Clouds with
Optimal Transport and Random Walk](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Self-Point-Flow_Self-Supervised_Scene_Flow_Estimation_From_Point_Clouds_With_Optimal_CVPR_2021_paper.pdf).
The code is created by Ruibo Li (ruibo001@e.ntu.edu.sg).

## Prerequisities
* Python 3.6.13
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 1.4.0)
* tqdm
* sklearn
* pptk
* yaml

Create a conda environment for Self-Point-Flow: 
```bash
conda create -n Self-Flow python=3.6.13
conda activate Self-Flow
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install tqdm pptk PyYAML sklearn
```

Compile the furthest point sampling, grouping and gathering operation for PyTorch. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).
```bash
cd lib
python setup.py install
cd ../
```

## Data preprocess
By default, the datasets are stored in `SAVE_PATH`. 
### FlyingThings3D
Download and unzip the "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" for DispNet/FlowNet2.0 dataset subsets from the [FlyingThings3D website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we used the paths from [this file](https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_all_download_paths.txt), now they added torrent downloads)
. They will be upzipped into the same directory, `RAW_DATA_PATH`. Then run the following script for 3D reconstruction:

```bash
python data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

Generate surface normals for the training set of FlyingThings3D: 
```bash
python data_preprocess/process_FT3D_s_train_data.py --data_root SAVE_PATH/FlyingThings3D_subset_processed_35m/train --save_root  SAVE_PATH/FlyingThings3D_subset_processed_35m/train_s_norm
```
This dataset is denoted FT3D<sub>s</sub> in our paper. 

### KITTI
* KITTI scene flow data provided by [HPLFlowNet](https://github.com/laoreja/HPLFlowNet)

Download and unzip [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) to directory `RAW_DATA_PATH`.
Run the following script for 3D reconstruction:
```bash
python data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_occ_final
```
This dataset is denoted KITTI<sub>s</sub> in our paper. 

* KITTI scene flow data provided by [FlowNet3D](https://github.com/xingyul/flownet3d)

Download and unzip [data](https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi) processed by FlowNet3D to directory `SAVE_PATH`. This dataset is denoted KITTI<sub>o</sub> in our paper. 

* Unlabeled KITTI raw data

In our paper, we use [raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php) from KITTI for self-supervised scene flow learning. 
We release the unlabeled training data [here](https://drive.google.com/file/d/12S69dpuz3PDujVZIcrDP_8H5QmbWZP9m/view?usp=sharing) for download. This dataset is denoted KITTI<sub>r</sub> in our paper. 



## Evaluation
Set `data_root` in each configuration file to `SAVE_PATH` in the data preprocess section.

### Trained models
Our trained models can be downloaded from [model trained on FT3D<sub>s</sub>](https://drive.google.com/file/d/1PcZFsO4XVJZR_jFY-ZN9s8gGRc3Eqf2c/view?usp=sharing) and [model trained on KITTI<sub>r</sub>](https://drive.google.com/file/d/1mFnMvcyvECSnEad1L_7snMls_IYLVedQ/view?usp=sharing).


### Testing

* Model trained on FT3D<sub>s</sub> 

When evaluating this pre-trained model on FT3D<sub>s</sub> testing data, set `dataset` to `FT3D_s_test`.  And when evaluating this pre-trained model on KITTI<sub>s</sub> data, set `dataset` to `KITTI_s_test`. 
Then run:
```bash
python evaluate.py config_evaluate_FT3D_s.yaml
```

* Model trained on KITTI<sub>r</sub>  

Evaluate this pre-trained model on KITTI<sub>o</sub>: 
```bash
python evaluate.py config_evaluate_KITTI_o.yaml
```

## Training
Set `data_root` in each configuration file to `SAVE_PATH` in the data preprocess section.

* Train model on FT3D<sub>s</sub> with 8192 points as input:
```bash
python train_FT3D_s.py config_train_FT3D_s.yaml
```
* Train model on KITTI<sub>r</sub> with 2048 points as input:
```bash
python train_KITTI_r.py train_KITTI_r.yaml
```

## Citation

If you find this code useful, please cite our paper:
```
@inproceedings{li2021self,
  title={Self-point-flow: Self-supervised scene flow estimation from point clouds with optimal transport and random walk},
  author={Li, Ruibo and Lin, Guosheng and Xie, Lihua},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={15577--15586},
  year={2021}
}
```

## Acknowledgement

Our code is based on [HPLFlowNet](https://github.com/laoreja/HPLFlowNet), [PointPWC](https://github.com/DylanWusee/PointPWC) and [FLOT](https://github.com/valeoai/FLOT).
And the flownet3d model is based on [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch), [FlowNet3D](https://github.com/xingyul/flownet3d), and [flownet3d_pytorch](https://github.com/hyangwinter/flownet3d_pytorch).