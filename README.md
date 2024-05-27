# Viola-Unet V2
[Viola-Unet](https://arxiv.org/abs/2208.06313) is the winning solution for the validation dataset in the 2022 Intracranial Hemorrhage Segmentation challenge ([INSTANCE 2022](https://instance.grand-challenge.org/)). It’s a powerful AI model designed for segmenting intracranial hemorrhages (ICH) in head CT scans. We retrained the model with more data and released Vioal-Unet version 2 (viola_v2) for academical users. 
---
<img align="top" src="demo/125_0.923.gif" width="400"/> <img align="top" src="demo/105_0.881.gif" width="400"/>

## Test Pre-Trained Models in Docker
1. Download the pre-built Docker image: [viola_v2.tar.gz](https://e.pcloud.link/publink/show?code=XZID5MZvtia7EGYQypb0JDLiVu71p4kK4vy).
2. Prepare your input folder containing all CT files for test (e.g., ```/home/yourname/Desktop/input```) 
3. Create an empty output folder (e.g., ```/home/yourname/Desktop/predict```) 
4. Run the program with the following commands (assuming you're on Linux OS):
```
docker load < viola_v2.tar.gz
docker run --gpus "device=0" --name viola -e PYTHONUNBUFFERED=1 -v /home/yourname/Desktop/input:/input -v /home/yourname/Desktop/predict:/predict viola_v2:latest
```
The program will: 
1. Read each CT file (```*.nii.gz``` or ```*.nii``` in the input folder.
2. Use pre-trained models (ensemble of Viola_Unet and nnU-Net) to segment possible ICH from the CT scans.
3. Save the segmented masks to the output folder (with exactly the same name as input file)
4. Output inference messages to the terminal and save all messages to ```prediction_info.csv```.
Example inference message:
```
model nnUNet loaded successfully!
model Viola_s loaded successfully!

---------------start predicting input file: 002.nii.gz - 1/2 ----------------
Predicted lesion volume : 6.282 ml
Segmention was saved to file: 002.nii.gz
Cost time: 2.269 sec

---------------start predicting input file: 003.nii.gz - 2/2 ----------------
Predicted lesion volume : 0.428 ml
Segmentation was saved to the file: 003.nii.gz
Cost time: 2.335 sec

-------------------------Completed--------------------------------------------------
Predictions infor is saved to predictions_info.csv
```
## Running Inference on CPU and Windows OS
1. Load the Docker image:
```
docker load -i viola_v2.tar.gz
```
2. Run the inference with the following command:
```
docker run --name viola -v D:\data\CT\test\input\:/input -v D:\data\CT\test\predict\:/predict viola_v2:latest
```

### Folder Structure:
```
├── /home/yourname/Desktop/input
          ├── 144.nii.gz
          ├── 145.nii.gz
          ├── 146.nii.gz

├── /home/yourname/Desktop/predict
          ├── 144.nii.gz
          ├── 145.nii.gz
          ├── 146.nii.gz
          ├── predictions_info.csv
          
```

## Citation: 
Please consider citing [our work](https://arxiv.org/abs/2208.06313) if you find the code helps you

```
@inproceedings{liu2023ICH,
  title={Voxels Intersecting along Orthogonal Levels Attention U-Net for Intracerebral Haemorrhage Segmentation in Head CT},
  author={Qinghui Liu and Bradley J MacIntosh and Till Schellhorn and Karoline Skogen and KyrreEeg Emblem and Atle Bjørnerud},
  booktitle={Proceedings of ISBI 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  year={2023}
}
```
