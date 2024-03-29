# Codes and dataset (*DFDM*) for Model Attribution of Face-swap Deepfake Videos [<a href="https://arxiv.org/abs/2202.12951">Paper</a>]

We created a new dataset, named *DFDM*, with 6,450 Deepfake videos generated by different Autoencoder models. Specifically, five Autoencoder models with variations in encoder, decoder, intermediate layer, and input resolution, respectively, have been selected to generate Deepfakes based on the same input. We have first observed the visible but subtle visual differences among different Deepfakes, demonstrating the evidence of model attribution artifacts. Then we take Deepfakes model attribution as a multiclass classification task and propose a spatial and temporal attention based method to explore the differences among Deepfakes in *DFDM* dataset.

<img src="Fig/Fig1.jpg" alt="demo" width="600"/>

## Dataset Overview 
The DFDM dataset includes face-swap Deepfakes videos generated from five Autoencoder models based on [Faceswap](https://github.com/deepfakes/faceswap) software, including the *Faceswap*, *Lightweight*, *IAE*, *Dfaker*, and *DFL-H128*. Three H.264 compression rates are considered to get videos with different qualities, including lossless with the constant rate factor (crf) as 0, high quality with crf as 10, and low quality with crf as 23. A total of 6,450 Deepfakes have been created.


| Model | Input | Output | Encoder | Decoder | Variation| 
| :-------------------: | :-----: | :-----: | :---------: | :---------: | :---------: |
|  *Faceswap* (baseline)  |   64    |   64    |  4Conv+1Ups |  3Ups+1Conv |  /
|  *Lightweight*          |   64    |   64    |  3Conv+1Ups |  3Ups+1Conv |  Encoder|  
|  *IAE*                  |   64    |   64    |  4Conv      |  4Ups+1Conv |  Intermediate layers; Shared Encoder&Decoder|  
|  *Dfaker*               |   64    |   128   |  4Conv+1Ups | 4Ups+3Residual+1Conv |  Decoder|  
|  *DFL-H128*             |  128    |  128    |  4Conv+1Ups |  3Ups+1Conv |  Input resolution|  

### Dataset Structure
The total size of the dataset files is ~82.4GB.  
The database contains three folders "DFDM_crf*" with videos in three subfolders:  
- DFDM_crf0 ~ 53.7G: 2,150 videos from 5 models with lossless qualities;
- DFDM_crf10 ~ 24.4G: 2,150 videos from 5 models with higher qualities;
- DFDM_crf23 ~ 4.3G: 2,150 videos from 5 models with lower qualities. 

File name: id**\_id**\_(scene)\_(model_name).mp4  
To extract frames/faces, we recommend to use FFMPEG, please refer to the [Faceswap](https://github.com/deepfakes/faceswap) [USAGE.md](https://github.com/deepfakes/faceswap/blob/master/USAGE.md#).

### Download
If you would like to access the *DFDM* dataset, please fill out this [google form](https://docs.google.com/forms/d/e/1FAIpQLSeM-1pJ13RyPVgF0bGRQtLiupwWDvALD6rKa_Oa8sIluIqtSA/viewform?vc=0&c=0&w=1&flr=0&usp=mail_form_link). The download link will be sent to you once the form is accepted (in 72 hours). If you have any questions, please send email to [dfdmdataset@gmail.com].

## Deepfake Model Attribution
We designed a simple and effective Deepfake video model attribution method based on Spatial and Temporal Attention (DMA-STA), and achieved an overall accuracy of ~70% in identifying the higher-quality Deepfakes in DFDM dataset.
### Usage
1. **Prerequisites**: our code requires PyTorch and Python 3.  
2. **Pre-trained model**: download the pretrained ResNet-50 model, put it into the 'data' folder, i.e., 'data/resnet50-19c8e357.pth'.  
3. **Data structure**: DF_class*/Video*/Frame1.png, ... (cropped face images).  
4. **Hyper-parameters**: you can modify 'config.py' to set more detailed hyper-parameters.  

# License and Citation
The DFDM database is released only for academic research. Researchers from educational institute are allowed to use this database freely for noncommercial purpose.

If you use this dataset, please cite the following paper:
```
@inproceedings{jia2022model,
  title={Model attribution of face-swap deepfake videos},
  author={Jia, Shan and Li, Xin and Lyu, Siwei},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={2356--2360},
  year={2022},
  organization={IEEE}
}
