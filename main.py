import argparse, os
import time
import numpy as np
import torch
import pandas as pd
from skimage import draw

from load_model import load_model, infer_seg, nibout, infer_seg_3, patch_overlap, patch_size, spacing
from load_data import load_data, post_process, read_raw_image, pre_process_cls
from monai.transforms import SaveImaged
from monai.data import decollate_batch

from dense_net import load_detect_modes, get_cams, detect_ich
from img_utils import visualize_cam


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

    # --- -----  load detector models and CAM visualizations
    models_cls = load_detect_modes(device=device)
    cams, targets = get_cams(models=models_cls)


    test_file_list, dataloader = load_data(args.input_dir)

    # out_file_name = "predictions_info.csv"
    csv_file = os.path.join(args.predict_dir, "predictions_info.csv")
     
    with torch.no_grad():
        num_scans = len(dataloader)
        print('\n-------There are total "{0}" CT scans found in the input folder -----'.format(num_scans))
        for i, d in enumerate(dataloader):

            filenames, pred_volums, infer_time = [], [], []
            any_ich, edh, iph, ivh, sah, sdh = [], [], [], [], [], []
            
            path, filename = os.path.split(test_file_list[i]['image'])
            filenames.append(filename)
            
            raw_data = read_raw_image(test_file_list[i])
            raw_img = raw_data["image"]
            pixdims = raw_data["image_meta_dict"]["pixdim"][1:4]
            pix_volume = pixdims[0] * pixdims[1] * pixdims[2]  # mm^3
            new_pix_vol = spacing[0] * spacing[1] * spacing[2] 
            
            images = d["image"].to(device)
            print('\n--------start detect, classify, and segment ICH from "{0}" - {1}/{2} ----------------'.format(filename, i + 1, num_scans))

            pred_dict, idx_vis, visual_imgs = detect_ich(models=models_cls, cams=cams, targets=targets,
                                                         test_file=test_file_list[i],
                                                         pre_process_cls=pre_process_cls,
                                                         device=device, batch_size=12)
            any_ich.append(pred_dict['any_ICH'])
            edh.append(pred_dict['EDH'])
            iph.append(pred_dict['IPH'])
            ivh.append(pred_dict['IVH'])
            sah.append(pred_dict['SAH'])
            sdh.append(pred_dict['SDH'])
            print("Detected ICH-type (num of slices):\n", pred_dict)
            if idx_vis!= -1:
                visualize_cam(visual_imgs=visual_imgs, patient=filename, n_slice=idx_vis, path=args.predict_dir)

            print('\n--------start segmentation-------')
            # print("image size after preprocessed: ", images.size())

            start_time = time.time()
            pred_outputs = list()
            voting_ensemb = False
            for m in models:  # in this case, we only have one model
                pred = infer_seg(images, m, roi_size=patch_size, overlap=patch_overlap) 
                pred_outputs.append(pred)
                pred_vol = torch.sum(torch.argmax(torch.softmax(pred, 1), 1, keepdim=True)) * new_pix_vol / 1000.
                if pred_vol < 0.1:
                    if not voting_ensemb:
                        print("The ICH region might be too small, the model is trying to do more augmentaion---")
                        voting_ensemb = True
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
                for j, p in enumerate(pred_outputs):
                    if j != num_pred-1:
                        d_copy = d.copy()
                        d_copy["pred"] = p
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
            d[0]["pred"] = d[0]["pred"].cpu().detach().numpy().astype(np.uint8)

            if idx_vis != -1 and lesion_volume <= 0.:
                print('---Warning: Detector finds there are certain ICH, but Segmentation model could not find.')
                print('---Draw a circle in the mask file for warning purpose.----')
                r, c = d[0]["pred"][:, :, idx_vis].shape
                rr, cc = draw.circle_perimeter(r//2, c//2, radius=100, shape=d[0]["pred"][:, :, idx_vis].shape)
                d[0]["pred"][:, :, idx_vis][rr, cc] = 1

            nibout(
                d[0]["pred"],
                args.predict_dir, 
                test_file_list[i]['image']
                )
            infer_time.append(time.time() - start_time)
            print('Cost time: {:.3f} sec'.format(time.time() - start_time))

            if os.path.isfile(csv_file):
                pred_csv = pd.read_csv(csv_file)
            else:
                pred_csv = pd.DataFrame(columns = ['Filename', 'Pre_volume',
                                                   'any_ICH', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH',
                                                   "Infer_time"]
                                        )
            
            df = pd.DataFrame({'Filename': filenames, 'Pre_volume': pred_volums,
                               'any_ICH': any_ich, 'EDH': edh, 'IPH': iph, 'IVH': ivh, 'SAH':sah, 'SDH': sdh,
                               "Infer_time": infer_time})
            df_rounded = df.round(3)
            updated_df = pd.concat([pred_csv, df_rounded], ignore_index=True)

            # Write the updated DataFrame back to the CSV file
            updated_df.to_csv(csv_file, index=False)
            
        print("\n-------------------------Completed--------------------------------------------------")
        print("Predictions infor is saved to", csv_file)
