# LCCNet

**Fork of** the official PyTorch implementation of the paper “LCCNet: Lidar and Camera Self-Calibration Using Cost Volume Network”. A video of the demonstration of the method can be found on
 https://www.youtube.com/watch?v=UAAGjYT708A



## Table of Contents

- [Main changes to the original repo](#main-changes)
- [Requirements](#Requirements)
- [Pre-trained model](#Pre-trained_model)
- [Evaluation](#Evaluation)
- [Train](#Train)
- [Acknowledgments](#Acknowledgments)
<!-- - [Citation](#Citation)  currently no content! !-->



## Main changes
The main changes between the initial fork and the current version of the repo are:
- Updated the codebase regarding library and function updates for compatibiliy to e.g. new CUDA or PyTorch releases (PyTorch version 1.0.1.post2 is hard/ not possible to get anymore)
- Removed unused code and added new comments for better maintainability
- Added a Dockerfile for containerized execution if needed
- Added a .gitignore to exclude unnecessary files from version control
- Added a script to automatically train the set of models successively



## Requirements

### Hardware:
  - CPU only
    - If no gpu is available, the scripts must be executed with cpu only. Here is a big enough ram for a meaningful batch size needed
    - The cuda installation steps can then be skipped
  - GPU:
    - If a gpu is available you can run the scripts with the gpu - this is probably faster compared to the cpu usage

### Installation: Docker

  - Docker settings which were used:
  ```commandline
  docker build -t lccnet:latest .

  docker run --rm -it --gpus all --shm-size=32g --ipc=host \
      -v ~/Desktop/LCCNet:/workspace:rw \
      lccnet:latest
  
  ```

### Installation: Local

- **Python**: 3.10
- **CUDA & GPU**: CUDA 12.8 toolkit with a compatible NVIDIA driver (≥ 520.xx) if you plan to run on GPU
- **PyTorch**: 2.7.0 with CUDA 12.8
  ```commandline
  pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

### Installation: Note

- In both cases you have to build the correlation_cuda-library e.g. by using
```commandline
cd models/correlation_package/

python3 setup.py build_ext --inplace
```

- Further requirements besides the _main requirements_ are written down in the requirements.txt file



## Pre-trained models

Pre-trained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1VbQV3ERDeT3QbdJviNCN71yoWIItZQnl?usp=sharing) provided by the initial authors of the LCCNet

- Note: The scripts are changed; they do now expect the models in .pth format instead of the .tar format (tar includes pth alongside other not needed information). A conversion script is also provided [here](./convert_tar_to_pth.py).



## Evaluation

<ul>
  <li>1. Download KITTI odometry dataset (http://www.cvlibs.net/datasets/kitti/eval_odometry.php).</li>
  <li>2. Either:
    <ul>
    <li>2.1. Change the path to the dataset in `evaluate_calib.py` (e.g. data_folder = '/path/to/the/KITTI/odometry_color/').</li>
    <li>2.2. Provide directly the configuration values by command line (e.g. "python3 evaluate_calib.py with data_folder = '/path/to/the/KITTI/odometry_color/'")</li>
    </ul>
  </li>
  <li>3. Create a folder named `pretrained` to store the pre-trained models in the root path.</li>
  <li>4. Download pre-trained models and modify the weights path in `evaluate_calib.py`: </li>
</ul>

```python
weights = [
   './pretrained/kitti_iter1.pth',
   './pretrained/kitti_iter2.pth',
   './pretrained/kitti_iter3.pth',
   './pretrained/kitti_iter4.pth',
   './pretrained/kitti_iter5.pth',
]
```
<ul>
  <li>5. Run evaluation:</li>
</ul>

```commandline
python3 evaluate_calib.py
```



## Train
Either:
```commandline
python3 train_with_sacred.py
```
Or, if all iterative models shall be trained in one turn:\
Configurate the script and run it:
```commandline
python3 train_automation.py
```

- As in the [initial paper](https://arxiv.org/pdf/2012.13901) discussed, it makes sense to train the first model with the biggest rotation and translation ranges with e.g. 120 epochs
- All other models can be transfer-learning trained by using the first trained model as a pretrained model; the epochs can be lowered to e.g. 50 epochs



<!--
## Citation
 
Thank you for citing our paper if you use any of this code or datasets.
```
_Insert here the citation_
```
!-->



## Acknowledgments
Original github: [https://github.com/IIPCVLAB/LCCNet](https://github.com/IIPCVLAB/LCCNet)

Original paper: [LCCNet: LiDAR and Camera Self-Calibration using Cost Volume Network](https://arxiv.org/pdf/2012.13901)
