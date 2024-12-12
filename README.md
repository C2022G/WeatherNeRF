# WeatherNeRF
**Multi-Image Decoupling of Haze and Scenes using Neural Radiance Fields**
![Overview of our method](https://github.com/C2022G/WeatherNeRF/blob/main/readme/2.png)

The implementation of our code is referenced in [kwea123-npg_pl](https://github.com/kwea123/ngp_pl)。The hardware and software basis on which our model operates is described next
 - Ubuntu 18.04
 -  NVIDIA GeForce RTX 3090 ,CUDA 11.3


## Setup
Let's complete the basic setup before we run the model。

 
+ Clone this repo by
```python
git clone https://github.com/C2022G/WeatherNeRF.git
```
+  Create an anaconda environment
```python
conda create -n dcpnerf python=3.7
``` 
+ cuda code compilation dependency.
	- Install pytorch by
	```python
	conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
	```
	- Install torch-scatter following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation) like
	```python
	pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
	```
	- Install tinycudann following their [instrucion](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)(pytorch extension) like
	```python
	pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
	```
	- Install apex following their [instruction](https://github.com/NVIDIA/apex#linux) like
	```python
	git clone https://github.com/NVIDIA/apex 
	cd apex 
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
	```
	- Install core requirements by
	```python
	pip install -r requirements.tx
	```
  
+ Cuda extension:please run this each time you pull the code.``.
 	```python
	pip install models/csrc/
	# (Upgrade pip to >= 22.1)
	```
  
## Datasets
**The dataset can be obtained from Baidu LLFF(https://pan.baidu.com/s/1s5QFNZ3XoxQQMBQyW0B0oA?pwd=rexh 提取码: rexh).**


## Training
```python
python run.py  \
	--root_dir /data/data/nerf_llff_data/fern
	--recovery_dir /data/program/WeatherDiffusion/result/fern
	--resume /data/data/resume/WeatherDiff64.pth.tar
	--dir_name rain
	--exp_name fern_rain
	--split train
	--downsample 4
	--scale 5
	--num_epochs 2
	--train_epoch 10
	--opacity_weight_1 1e-2
	--bdc_weight_1 1e-3
	--opacity_weight_2 1e-6
	--bdc_weight_2 1e-6
```

The optimal hyperparameters of each scene are obtained by experiments.

When both bdc_weight_2 and opacity_weight_2 are zero, this indicates that only Stage One, specifically five epochs, should be executed.

![](https://github.com/C2022G/WeatherNeRF/blob/main/readme/table1.png)
![](https://github.com/C2022G/WeatherNeRF/blob/main/readme/table2.png)
![](https://github.com/C2022G/WeatherNeRF/blob/main/readme/table3.png)
![](https://github.com/C2022G/WeatherNeRF/blob/main/readme/table4.png)
