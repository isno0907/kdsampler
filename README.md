
# Installation:
a. Create a conda virtual environment and activate it.

```shell
conda create -n kdsampler python=3.9 -y
conda activate kdsampler
```

b. Install PyTorch and TorchVision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

c. Install Dependencies.

```shell
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install einops
pip install tqdm
pip install pillow
pip install pandas
pip install tensorboard
pip install opencv-python==4.6.0
```

d. Clone the KDSampler repository.

```shell
git clone https://github.com/isno0907/kdsampler
```

e. Install build requirements and then install MMAction2.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```
