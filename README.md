# Viola-Unet
A description of the winning solution: [Viola-Unet](https://arxiv.org/abs/2208.06313) for the validation dataset in the 2022 Intracranial Hemorrhage Segmentation challenge ([INSTANCE 2022](https://instance.grand-challenge.org/))
---
<img align="top" src="demo/125_0.923.gif" width="400"/> <img align="top" src="demo/105_0.881.gif" width="400"/>

## Test the pre-trained models in docker
1. Download the docker image [crainet.tar.gz](https://e1.pcloud.link/publink/show?code=XZTBy4ZYwtUXUhrCk4QfIMQCiPHl7KneUzk)
2. Prepare your input folder such as ```/home/yourname/Desktop/input``` (a folder contains all CT files for testing) will be synchronized with ```/input``` in the docker container, and output folder such as 
```/home/yourname/Desktop/predict``` (an empty folder used to save segmentation file) will be synchronized with ```/predict``` in the docker container. Note that, the filename of the segmentation mask file (```./predict```) is the same as the CT file (```./input```), the segmentation mask is a 3D zero-one array(0 stands for background, 1 stands for target), and the meta information of the segmentation mask file is consistent with that of original CT file (```*.nii.gz```). 
3. Run the program with commands as follows (assuming on Linux OS)
```
docker load < crainet.tar.gz
docker run --gpus "device=0" -name crainetst -v /home/yourname/Desktop/input:/input -v /home/yourname/Desktop/predict:/predict crainet:latest
```
4. The program will do following: first, read each CT file(```*.nii.gz```) in folder ```./input```, then, use pre-trained models (ensemble of viola-Unet and nnU-Net) to segment the CT scans one by one and save the segmented masks to ```./predict```. 

5. The structrue of the input and output folder for testing :
```
├── /home/yourname/Desktop/input
          │   ├── 144.nii.gz
          │   ├── 145.nii.gz
          │   ├── 146.nii.gz

├── /home/yourname/Desktop/predict
          │   ├── 144.nii.gz
          │   ├── 145.nii.gz
          │   ├── 146.nii.gz
```
6. What if only having CPU and Windows OS, you can Run the inference as following: 
```
docker load -i crainet.tar.gz
docker run --name crainetx -v D:\data\CT\test\input\:/input -v D:\data\CT\test\predict\:/predict crainet:latest
```
just replace ```D:\data\CT\test\input\``` and ```D:\data\CT\test\predict\``` with your own folder path. 
