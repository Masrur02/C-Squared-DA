# C²DA
This code is the official repository for our **C²DA: Contrastive and Context-Aware Learning for Domain Adaptive
Semantic Segmentation**.
# Overview
Unsupervised domain adaptive semantic segmentation (UDA-SS) aims to train a model on the source domain data (e.g., synthetic) and adapt the model to predict target domain data(e.g. real-world) without accessing target annotation data. Most existing UDA-SS methods only focus on the inter-domain knowledge to mitigate the data-shift problem. However, learning the inherent structure of the images and exploring the intrinsic pixel distribution of both domains are ignored; which prevents the UDA-SS methods from producing satisfactory performance like supervised learning. Moreover, incorporating contextual knowledge is also often overlooked. Considering the issues, in this work, we propose a UDA-SS framework that learns both intra-domain and context-aware knowledge. To learn the intra-domain knowledge, we incorporate contrastive loss in both domains, which pulls pixels of similar classes together and pushes the rest away, facilitating intra-image-pixel-wise correlations. To learn context-aware knowledge, we modify the mixing technique by leveraging contextual dependency among the classes to learn context-aware knowledge. Moreover, we adapt the Mask Image Modeling (MIM) technique to properly use context clues for robust visual recognition, using limited information about the masked images. 
![image](https://github.com/Masrur02/IDA/assets/33350185/c0c50896-b52c-4118-bba9-0eaf63f9f900)
# Setup Environment
We recommend setting up a new virtual environment. In that environment, the requirements can be installed with:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```
Further, please download the MiT weights from SegFormer using the following script. If problems occur with the automatic download, please follow the instructions for a manual download within the script.
```bash
sh tools/download_checkpoints.sh
```
# Setup Datasets
**Cityscapes**: Please, download leftImg8bit_trainvaltest.zip and gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
 and extract them to
```bash
data/cityscapes
```
**GTA**: Please, download all image and label packages from [here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to 
```bash 
data/gta
 ```
**RUGD**: Please, download all image and label packages from [here](http://rugd.vision/) and extract them to 
```bash 
data/rugd
 ```
**MESH**: Please, download all image and label packages from [here](http://rugd.vision/) and extract them to 
```bash 
data/MESH
```
The final folder structure should look like this:
```bash 
DEDA
├── ...
├── data
│   
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│  
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── rugd
│   │   ├── images
│   │   ├── labels
│   ├── MESH
│   │   ├── images
│   │   ├── labels
│   │ 
├── 
```

**Data Preprocessing**: Finally, please run the following scripts to convert the label IDs to the train IDs and to generate the class index for RCS:
```bash
#Testing & Predictions

python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```
# Training
A training job for gta2cs can be launched using:
```bash
python run_experiments.py --config configs/C²DA/gtaHR2csHR_hrda.py
```
and a training job for rugd2mesh can be launched using:
```bash
python run_experiments.py --config configs/C²DA/rugd2mesh_hrda.py
```
The logs and checkpoints are stored in 
```bash 
work_dirs/
```
# Testing & Predictions
The provided IDA checkpoint trained on GTA→Cityscapes can be tested on the Cityscapes validation set using:
```bash
sh test.sh work_dirs/gtaHR2csHR_hrda_246ef
```
And the provided IDA checkpoint trained on RUGD→MESH can be tested on the MESH validation set using:
```bash
sh test.sh work_dirs/rugdHR2meshHR_hrda_246ef
```
# Running in ROS
The trained segmentation model can be used for visual navigation by using:
```bash
sh in_ros.sh work_dirs/rugdHR2meshHR_hrda_246ef
```
# Framework Structure
This project is based on [mmsegmentation version 0.16.0.](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0) For more information about the framework structure and the config system, please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html) and the [mmcv documentation](https://mmcv.readthedocs.ihttps//arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for IDA are:

[configs/C²DA/gtaHR2csHR_hrda.py](https://github.com/Masrur02/DEDA_P/blob/main/configs/hrda/gtaHR2csHR_hrda.py): Annotated config file for the final IDA.

[configs/C²DA/rugd2mesh_hrda.py](https://github.com/Masrur02/DEDA_P/blob/main/configs/hrda/rugd2mesh_hrda.py): Annotated config file for the final IDA.

[mmseg/models/segmentors/hrda_encoder_decoder.py](https://github.com/Masrur02/DEDA_P/blob/main/mmseg/models/segmentors/hrda_encoder_decoder.py): Implementation of the HRDA multi-resolution encoding with context and detail crop.

[mmseg/models/decode_heads/hrda_head.py](https://github.com/Masrur02/DEDA_P/blob/main/mmseg/models/decode_heads/hrda_head.py): Implementation of the HRDA decoding with multi-resolution fusion and scale attention.

[mmseg/models/uda/dacs.py](https://github.com/Masrur02/DEDA_P/blob/main/mmseg/models/uda/dacs.py): Implementation of the DAFormer self-training.
# Acknowledgements
IDA is based on the following open-source projects. We thank their authors for making the source code publicly available.
- [HRDA](https://github.com/lhoyer/HRDA#setup-environment)
- [MIC](https://github.com/lhoyer/MIC)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [SegFormer](https://github.com/NVlabs/SegFormer)
- [PiPa](https://github.com/chen742/PiPa)
- [CAMix](https://github.com/qianyuzqy/CAMix)





