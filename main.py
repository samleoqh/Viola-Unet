import argparse, os
import time
import numpy as np
import torch

from load_model import load_model, infer_seg, nibout, infer_seg_3
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
 
    models.append(load_model(network="nnUNet", kf="kf0", device=device).eval())
    models.append(load_model(network="nnUNet", kf="kf1", device=device).eval())
    models.append(load_model(network="nnUNet", kf="kf2", device=device).eval())
    models.append(load_model(network="nnUNet", kf="kf3", device=device).eval())
    models.append(load_model(network="nnUNet", kf="kf4", device=device).eval())

    models.append(load_model(network="ViolaUNet_l", kf="kf0", device=device).eval())
    models.append(load_model(network="ViolaUNet_l", kf="kf1", device=device).eval())
    models.append(load_model(network="ViolaUNet_l", kf="kf2", device=device).eval())
    models.append(load_model(network="ViolaUNet_l", kf="kf3", device=device).eval())
    models.append(load_model(network="ViolaUNet_l", kf="kf4", device=device).eval())


    test_file_list, dataloader = load_data(args.input_dir)


    with torch.no_grad():
        num_scans = len(dataloader)
        for i, d in enumerate(dataloader):
            path, filename = os.path.split(test_file_list[i]['image'])
            raw_data = read_raw_image(test_file_list[i])
            raw_img = raw_data["image"]
            pixdims = raw_data["image_meta_dict"]["pixdim"][1:4]
            pix_volume = pixdims[0] * pixdims[1] * pixdims[2]  # mm^3
            images = d["image"].to(device)
            print('\n------------------start predicting input volume: {0} - {1}/{2} -------------------'.format(filename, i + 1, num_scans))
            # print("image size after preprocessed: ", images.size())
            _, _, h, w, z = images.size()
            print("h, w, z: ", images.size())

            max_size = h if h>w else w
            max_size = max_size if max_size > z else z

            overlap = 1-(max_size - 160)/(2*160)
            overlap = 0 if overlap<0 else round(overlap, 2)
            print("overlap: ", overlap)
            ###  make sure the last slize is z-axial, z must be smallest number
            reshape=None
            mid_reshape=None # switch w and z
            if h<z and h<w:
                print("We discovered some errors in the head information, tried to fix here but the predict will still not work well in this case...")
                reshape=(0, 1, 4, 3, 2)
                images=images.permute(reshape)
                # print("reshaped input", images.size())
            elif w<z and w<h:
                mid_reshape=(0, 1, 2, 4, 3)
                # images=images.permute(reshape)
                # we can fix the error head infor, but the model were trained with this error case, so at this time, we don't fix it
                # we leave this error for future work
                print("We discovered some errors in the head information without trying to fix it, so the predict will not work well in this case...")

            start_time = time.time()
            pred_outputs = list()
            for m in models:  # in this case, we only have one model
                pred = infer_seg(images, m, overlap=overlap)
                pred_outputs.append(pred)
                if mid_reshape is not None:
                    print("try using corrected oritentation to infer ...")
                    pred2 = infer_seg(images.permute(mid_reshape), m, overlap=overlap).permute(mid_reshape)
                    if torch.sum(torch.argmax(torch.softmax(pred, 1), 1))<torch.sum(torch.argmax(torch.softmax(pred2, 1), 1)):
                        pred_outputs.append(pred2)
                    else:
                        print("did not use the corrected oritentation ...")
                # # do aumentation if bleeds are too small or not found
                if torch.sum(torch.argmax(torch.softmax(pred, 1), 1, keepdim=True))<10.:
                    print('small object, trying to TTA boost ...')
                    pred = infer_seg_3(images, m, flip_axis=[2], overlap=overlap)
                    # print(torch.argmax(torch.softmax(pred, 1), 1, keepdim=True).size())
                    if torch.sum(torch.argmax(torch.softmax(pred, 1), 1, keepdim=True))>10.:
                        # pred_outputs.pop()
                        pred_outputs.append(pred)
                        print('augmented by flip2')
                    
                    pred = infer_seg_3(images, m, overlap=overlap, flip_axis=[1, 2], rot=1)
                    if torch.sum(torch.argmax(torch.softmax(pred, 1), 1, keepdim=True))>10.:
                        # pred_outputs.pop()
                        pred_outputs.append(pred)
                        print('augmented by flip 1-2 and rot 1')

            d["pred"] = torch.mean(torch.stack(pred_outputs, dim=0), dim=0, keepdim=True).squeeze(0)

            # print(d["pred"].size())
            if reshape is not None:
                d["pred"] = d["pred"].permute(reshape)
                # print('reshaped size: ',d["pred"].size())

            d = [post_process(img) for img in decollate_batch(d)]
            d[0]["pred"] = torch.argmax(d[0]["pred"], 0, keepdim=True)
            lesion_volume = torch.sum(d[0]["pred"]) * pix_volume / 1000. 
            print('Predicted lesion volume : {:.3f} ml'.format(lesion_volume))

            d[0]["pred"] = d[0]["pred"].squeeze(0)
            nibout(
                d[0]["pred"].cpu().detach().numpy().astype(np.uint8),
                args.predict_dir, 
                test_file_list[i]['image']
                )
            
            print('--------Cost time: {:.3f} sec --------'.format(time.time() - start_time))
    
            
