import argparse, os
import time
import numpy as np
import torch
import pandas as pd

from load_model import load_model, infer_seg, nibout, infer_seg_3, patch_overlap, patch_size, spacing
from load_data import load_data, post_process, read_raw_image
from monai.transforms import SaveImaged
from monai.data import decollate_batch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ICH segmentation of a ct volume')
    parser.add_argument('--input_dir', default='', type=str, metavar='PATH',
                            help='this directory contains all test samples(ct volumes)')
    parser.add_argument('--predict_dir', default='', type=str, metavar='PATH',
                            help='segmentation file of each test sample should be stored in the directory')

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    models = [] # ensemble models, stack together
    
    models.append(load_model(network="nnUNet", device = device).eval())
    models.append(load_model(network="Viola_s", device = device).eval())


    test_file_list, dataloader = load_data(args.input_dir)

    out_file_name = "predictions_info.csv"
    filenames, pred_volums, infer_time = [], [], []
    
    with torch.no_grad():
        num_scans = len(dataloader)
        for i, d in enumerate(dataloader):
            path, filename = os.path.split(test_file_list[i]['image'])
            filenames.append(filename)
            
            raw_data = read_raw_image(test_file_list[i])
            raw_img = raw_data["image"]
            pixdims = raw_data["image_meta_dict"]["pixdim"][1:4]
            pix_volume = pixdims[0] * pixdims[1] * pixdims[2]  # mm^3
            new_pix_vol = spacing[0] * spacing[1] * spacing[2] 
            
            images = d["image"].to(device)
            print('\n---------------start predicting input file: {0} - {1}/{2} ----------------'.format(filename, i + 1, num_scans))
            # print("image size after preprocessed: ", images.size())

            start_time = time.time()
            pred_outputs = list()
            voting_ensemb = False
            for m in models:  # in this case, we only have one model
                pred = infer_seg(images, m, roi_size=patch_size, overlap=patch_overlap) 
                pred_outputs.append(pred)
                pred_vol = torch.sum(torch.argmax(torch.softmax(pred, 1), 1, keepdim=True)) * new_pix_vol / 1000.
                if pred_vol < 0.1:
                    print("The ICH region might be too small, the model is trying to do more augmentaion---")
                    voting_ensemb = True
                    # seem the ICH bleeds is too small, try to do more inference with augmentaion
                    # pred = infer_seg_3(images, m,  roi_size=(96, 96, 32), overlap=0.25)
                    # pred_outputs.append(pred)
                    pred = infer_seg_3(images, m, flip_axis=[1], roi_size=(96, 96, 32), overlap=0.25)
                    pred_outputs.append(pred)
                    pred = infer_seg_3(images, m, flip_axis=[2], roi_size=(96, 96, 32), overlap=0.25)
                    pred_outputs.append(pred)
                    

            if not voting_ensemb:
                d["pred"] = torch.mean(torch.stack(pred_outputs, dim=0), dim=0, keepdim=True).squeeze(0)
                d = [post_process(img) for img in decollate_batch(d)]
                d[0]["pred"] = torch.argmax(d[0]["pred"], 0, keepdim=True)
            else:
                num_pred = len(pred_outputs)
                voting_p = 0
                for i, p in enumerate(pred_outputs):
                    if i != num_pred-1:
                        d_copy = d.copy()
                        d_copy["pred"] = p # torch.stack(pred_outputs, dim=0).squeeze(1
                        d_copy = [post_process(img) for img in decollate_batch(d_copy)]
                        voting_p += torch.argmax(d_copy[0]["pred"], 0, keepdim=True)
                    else:
                        d["pred"] = p
                        # print(d["pred"].shape)
                        d = [post_process(img) for img in decollate_batch(d)]
                        d[0]["pred"] = torch.argmax(d[0]["pred"], 0, keepdim=True)
                        d[0]["pred"][voting_p >= 1] = 1
                        
            
            lesion_volume = torch.sum(d[0]["pred"]) * pix_volume / 1000. 
            pred_volums.append(lesion_volume.item())
            print('Predicted lesion volume : {:.3f} ml'.format(lesion_volume))

            d[0]["pred"] = d[0]["pred"].squeeze(0)
            nibout(
                d[0]["pred"].cpu().detach().numpy().astype(np.uint8),
                args.predict_dir, 
                test_file_list[i]['image']
                )
            infer_time.append(time.time() - start_time)
            print('Cost time: {:.3f} sec'.format(time.time() - start_time))
            
        df = pd.DataFrame({'Filename': filenames, 'Pre_volume (ml)': pred_volums, "Infer_time": infer_time})
        df_rounded = df.round(3)
        df_rounded.to_csv(os.path.join(args.predict_dir, out_file_name), index=False)
        print("\n-------------------------Completed--------------------------------------------------")
        print("Predictions infor is saved to", out_file_name)
