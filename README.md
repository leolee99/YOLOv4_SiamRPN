## How to use

## Requierment

- CUDA 10.2
- OpenCV 4.1.1 with contrib(KCF need)
- TensorRT 7.1.3.0
- cuDNN 8.0.0
- Python3.6
- pip3 (make sure it's the new version:20.3.3)
- pycuda (run pip3 install pycuda)
- Numpy (run pip3 install numpy)
- pytorch 1.6.0 (without Anaconda)
- opencv-python 4.4.0.46
- tqdm
- pyyaml
- matplotlib
- colorama
- cython
- tensorboardX

## How to install OpenCV4.1.1 with CUDA in Jetson platform
```bash
sudo apt-get purge libopencv* # delete old version

cd ~

bash install_opencv4.1.1_jetson.sh # download the source and install
```
### Problem

```bash
In file included from /home/esraa/opencv_contrib/modules/cudaarithm/src/cuda/normalize.cu:53:0:
/home/esraa/opencv/modules/core/include/opencv2/core/private.cuda.hpp:75:0: warning: "NPP_VERSION" redefined
 #  define NPP_VERSION (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD)
```

Just use ifndef and define solve it

[solution](https://github.com/opencv/opencv/issues/15398)


```bash
~/opencv_contrib/modules/xfeatures2d/src/boostdesc.cpp:673:20: fatal error: boostdesc_bgm.i: No such file or directory
```

because of the network problem,i can't download those files

[solution](https://www.cnblogs.com/arxive/p/11778731.html)


## How to install PySOT without Anaconda

### Step-by-step instructions

#### Install numpy/pytorch/opencv

pip3 install numpy
pip3 install pytorch
pip3 install opencv-python

#### Install other requirements

pip3 install pyyaml yacs tqdm colorama matplotlib cython tensorboardX

#### Build extensions

python setup.py build_ext --inplace

### Problem

```bash
There are always some errors when we install pytorch, opencv-python and some other libs on aarch64 architecture by pip. Then you can download source from pytorch.org or pypi.org, excute the following instructions on terminal to install them.

python3 setup.py build
python3 setup.py install
```

## How to Excute Demo

### Add PySOT to your PYTHONPATH

export PYTHONPATH=/home/nvidia/yolo_siamrpn++:$PYTHONPATH #Your project path

### Download models

Download models in PySOT Model Zoo and put the model.pth in the correct directory in experiments.

### Webcam demo

#### backbone_ResNet-50

python3 trt_demo.py --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth --video_name video/2.flv

#### backbone_MobileNetv2

python3 trt_demo.py --config experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml --snapshot experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth --video_name video/2.flv

## Resources
[Jetson Packages install list](https://github.com/yqlbu/jetson-packages-family#vs-code-for-aarch64)

[miniforge](https://github.com/conda-forge/miniforge)(better than archiconda)
