<div align="center">
<h2>BaroPoser: Real-time Human Motion Tracking from IMUs and Barometers in Everyday Devices</h2>

[**Libo Zhang**](https://zhanglbthu.github.io/) · [**Xinyu Yi**](https://xinyu-yi.github.io/) · [**Feng Xu**](http://xufeng.site/)

Tsinghua University

</div>

### [Paper](https://www.arxiv.org/abs/2508.03313)
### Abstract
In recent years, tracking human motion using IMUs from everyday devices such as smartphones and smartwatches has gained increasing popularity. However, due to the sparsity of sensor measurements and the lack of datasets capturing human motion over uneven terrain, existing methods often struggle with pose estimation accuracy and are typically limited to recovering movements on flat terrain only. To this end, we present BaroPoser, the first method that combines IMU and barometric data recorded by a smartphone and a smartwatch to estimate human pose and global translation in real time. By leveraging barometric readings, we estimate sensor height changes, which provide valuable cues for both improving the accuracy of human pose estimation and predicting global translation on non-flat terrain. Furthermore, we propose a local thigh coordinate frame to disentangle local and global motion input for better pose representation learning. We evaluate our method on both public benchmark datasets and real-world recordings. Quantitative and qualitative results demonstrate that our approach outperforms the state-of-the-art (SOTA) methods that use IMUs only with the same hardware configuration.

### Pipeline
![teaser](./assets/pipeline.png)

### Results

<div align="center">
    <img src="./assets/BaroPoser.jpg" width="100%">
</div>


## Usage

### Setup

```bash
git clone git@github.com:zhanglbthu/Light-Transport-Gaussian.git
cd Light-Transport-Gaussian
conda create -n LTG python=3.7
conda activate LTG
pip install -r requirements.txt
```

### Dataset
We train the model using a dataset in the form of **LIGHT STAGE**, where the information about the camera and light source is known, specifically, the object is NeRF synthetic data and the light source is directional light.
You can download the generated dataset form [here](https://drive.google.com/drive/folders/1j4YlmIpuZZjyrXb4QxSI86ZgrIfP6mCr?usp=drive_link).
### Preprocess your own dataset
You can also preprocess your own dataset by following the steps below:
1. Download the data preprocessing code from [here](https://drive.google.com/drive/folders/1AiOE_F0imYrxqABN2BVy4On0BxnjbDgV?usp=sharing).
2. Modify the data configuration file `config/data.ini` to fit your own dataset.
3. Modify and run the bash script `bash/get_data.sh` 
### Train
You can download our trained model from [here](https://drive.google.com/drive/folders/1g4r1g_39yXL071Co9uQ7fgqEPaHgfO8B?usp=drive_link).
You can also train your own model by following the steps below:

First, modify the training configuration file `config/optimize.ini` to fit your own dataset.

**Arguments**:
- `root_path`: the root path of the dataset.
- `obj_name`: the name of the object.
- `out_name`: the name of the output folder.
- `data_type`: the type of the dataset, `NeRF` or `OpenIllumination`.

Then, run the training script:
```bash
bash bash/run_single.bash
```
### Test
We provide scripts for testing the relighting results of the trained model on the test set.
Similarly, modify the testing configuration file `config/evaluate.ini` and run the testing script:
```bash
bash bash/eval.bash
```