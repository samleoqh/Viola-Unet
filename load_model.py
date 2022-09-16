import os
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.transforms.utils import map_spatial_axes
from monai.data import decollate_batch


from viola_unet import ViolaUNet
from monai.networks.nets import DynUNet


wind_levels = [[0,100], [-15, 200],[-100, 1300]]
spacing = [0.45100001*2, 0.45100001*2, 4.99709511] 
patch_size = (160, 160, 32)
patch_overlap = 0.3 # 0.5 for last validation submission
sw_bt_size=1

# debug new setting
# patch_size = (192, 192, 24)
# patch_overlap = 0.75 # 0.5 for last validation submission


# # # last validation submission weights
# net_weights = {
#     "ViolaUNet_l":{
#         "kf0": "./best_ckpt1/viola/model_epoch_8640_dice_0.80106_lr_0.0000869017.pt",
#         "kf1": "./best_ckpt1/viola/model_epoch_4656_dice_0.76015_lr_0.0019869438.pt",  # new ft 0.76887
#         "kf2": "./best_ckpt1/viola/model_epoch_5940_dice_0.81959_lr_0.0000937899.pt",  # new ft 0.82135
#         "kf3": "./best_ckpt1/viola/model_epoch_37728_dice_0.78699_lr_0.0033099166.pt", # new ft 0.78784
#         "kf4": "./best_ckpt1/viola/model_epoch_12054_dice_0.78984_lr_0.0000753008.pt",
#     },
    
#     "nnUNet":{
#         "kf0": "./best_ckpt1/nnu/model_epoch_19296_dice_0.80530_lr_0.0042245472.pt",
#         "kf1": "./best_ckpt1/nnu/model_epoch_38412_dice_0.76024_lr_0.0022887478.pt",  # ft new 0.76780
#         "kf2": "./best_ckpt1/nnu/model_epoch_24651_dice_0.81655_lr_0.0037515006.pt", # ft new 0.81775
#         "kf3": "./best_ckpt1/nnu/model_epoch_1152_dice_0.79311_lr_0.0049999435.pt", 
#         "kf4": "./best_ckpt1/nnu/model_epoch_5292_dice_0.78911_lr_0.0049550524.pt",
#     },
# }


# # re-fine-tuned version after the last validation submission weights
net_weights = {
    "ViolaUNet_l":{
        "kf0": "./best_ckpt2/viola/model_epoch_8640_dice_0.80106_lr_0.0000869017.pt",
        "kf1": "./best_ckpt2/viola/model_epoch_5820_dice_0.76887_lr_0.0029100000.pt",  # new ft 0.76887
        "kf2": "./best_ckpt2/viola/model_epoch_297_dice_0.82135_lr_0.0001485000.pt",  # new ft 0.82135
        "kf3": "./best_ckpt2/viola/model_epoch_288_dice_0.78784_lr_0.0001440000.pt", # new ft 0.78784
        "kf4": "./best_ckpt2/viola/model_epoch_12054_dice_0.78984_lr_0.0000753008.pt",
    },
    
    "nnUNet":{
        "kf0": "./best_ckpt2/nnu/model_epoch_19296_dice_0.80530_lr_0.0042245472.pt",
        "kf1": "./best_ckpt2/nnu/model_epoch_6693_dice_0.76780_lr_0.0033465000.pt",  # ft new 0.76780
        "kf2": "./best_ckpt2/nnu/model_epoch_18414_dice_0.81775_lr_0.0047762088.pt", # ft new 0.81775
        "kf3": "./best_ckpt2/nnu/model_epoch_1152_dice_0.79311_lr_0.0049999435.pt", 
        "kf4": "./best_ckpt2/nnu/model_epoch_5292_dice_0.78911_lr_0.0049550524.pt",
    },
}


def load_model(network="ViolaUNet_l", kf="kf0", device='cpu', ckpt=True):
    if network == "ViolaUNet_l":
        model = ViolaUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=2,
            kernel_size=[[3, 3, 1], [3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 1], [1, 1, 1]],
            upsample_kernel_size=[[2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 1], [1, 1, 1]],
            filters=(32, 64, 96, 128, 192, 256, 320),
            dec_filters=(32, 64, 96, 128, 192, 256),
            norm_name=("BATCH", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout=0.2,
            deep_supervision=True,
            deep_supr_num=4,
            res_block=True, 
            trans_bias=True,
            viola_att = True,
            gated_att = False,
            sum_deep_supr = False,
        )
    elif network == 'nnUNet':
        model = DynUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=2,
            kernel_size=[[3, 3, 1], [3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 1], [1, 1, 1]],
            upsample_kernel_size=[[2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 1], [1, 1, 1]],
            filters=(32, 64, 96, 128, 192, 256, 320),
            dropout=0.2,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=True,
            deep_supr_num=4,
            res_block=True,
            trans_bias=True,
        )
    # elif network == 'ViolaUNet_s': # for paper figure 1
    #     model = ViolaUNet(
    #         spatial_dims=3,
    #         in_channels=3,
    #         out_channels=2,
    #         kernel_size=[[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 1]],
    #         strides=[[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]],
    #         upsample_kernel_size=[[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]],
    #         filters=(32, 64, 96, 128, 192, 256, 320),
    #         dec_filters=(32, 64, 96, 128, 128, 128),
    #         norm_name=("BATCH", {"affine": True}),
    #         act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    #         dropout=0.2,
    #         deep_supervision=True,
    #         deep_supr_num=2,
    #         res_block=False, 
    #         trans_bias=True,
    #         viola_att = True,
    #         gated_att = False,
    #         sum_deep_supr = False,
    #     )
    else:
        print("Not support the network currently - ", network)
        return None
    
    if ckpt and network != 'ViolaUNet_s':  # 
        pretrain = torch.load(net_weights[network][kf], map_location=device)
        model.load_state_dict(pretrain['state_dict'])
        print("model {}-{} loaded successfully!".format(network, kf))
    return model.to(device)


def infer_seg(images, model, 
    roi_size=patch_size, sw_batch_size=sw_bt_size, overlap=patch_overlap,
    flip_axis=-1, rot=0):
    if rot>0 and rot<4:
        val_outputs = sliding_window_inference(
            torch.stack([torch.rot90(k, rot, map_spatial_axes(k.ndim, (0, 1))) for k in decollate_batch(images)]),
            roi_size, sw_batch_size, model, overlap=overlap)
        val_outputs = torch.stack([torch.rot90(k, 4-rot, map_spatial_axes(k.ndim, (0, 1))) for k in decollate_batch(val_outputs)])
    elif flip_axis>=0 and flip_axis<3 :
        val_outputs = sliding_window_inference(
            torch.stack([torch.flip(k, map_spatial_axes(k.ndim, flip_axis)) for k in decollate_batch(images)]),
            roi_size, sw_batch_size, model, overlap=overlap)
        val_outputs = torch.stack([torch.flip(k, map_spatial_axes(k.ndim, flip_axis)) for k in decollate_batch(val_outputs)])

    else:
        val_outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, model, overlap=overlap)
    return val_outputs


def infer_seg_3(images, model, roi_size=patch_size, sw_batch_size=1, overlap=patch_overlap,
              flip_axis=[0, 1, 2], rot=-1):
    for axis in flip_axis:
        images = torch.stack([torch.flip(k, map_spatial_axes(k.ndim, axis)) for k in decollate_batch(images)])
    if rot > 0 and rot < 4:
        images = torch.stack([torch.rot90(k, rot, map_spatial_axes(k.ndim, (0, 1))) for k in decollate_batch(images)])

    val_outputs = sliding_window_inference(images, roi_size, sw_batch_size, model, overlap=overlap)

    if rot > 0 and rot < 4:
        val_outputs = torch.stack(
            [torch.rot90(k, 4 - rot, map_spatial_axes(k.ndim, (0, 1))) for k in decollate_batch(val_outputs)])
    for axis in flip_axis:
        val_outputs = torch.stack([torch.flip(k, map_spatial_axes(k.ndim, axis)) for k in decollate_batch(val_outputs)])

    return val_outputs


def nibout(segmentation, outputpath, imagepath):
    """
    save your predictions
    :param segmentation:Your prediction , the data type is "array".
    :param outputpath:The save path of prediction results.
    :param imagepath:The path of the image corresponding to the prediction result.
    :return:
    """
    # print(outputpath)
    path, filename = os.path.split(imagepath)
    print(filename)
    image = nib.load(imagepath)
    segmentation = nib.Nifti1Image(segmentation, image.affine)
    qform = image.get_qform()
    segmentation.set_qform(qform)
    sfrom = image.get_sform()
    segmentation.set_sform(sfrom)
    nib.save(segmentation, os.path.join(outputpath, filename))


# import time
if __name__ == '__main__':
    # _, channel, _, _, _ = input.shape
    # check model load and param size
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(network="ViolaUNet_l", device=device).eval()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    print(f'Trainable params: {sum([p.numel() for p in trainable_params])/1000**2.} M')
    # for kf in ["kf0", 'kf1', 'kf2', 'kf3', 'kf4']:
    #     model = load_model(network="ViolaUNet_s", kf=kf, device=device).eval()
    #     output = model(input)
    #     print(output.size())

    # ---  test inference speed -----------------------------
    # input = torch.randn(1, 3, 512, 512, 32).cuda()
    # input = torch.autograd.Variable(torch.sigmoid(torch.randn(1, 3, 512, 512, 32)), requires_grad=False).cuda()

    # output = model(input)
    # start_time = time.time()
    # output = infer_seg(input, model)
    # print('--------Cost time: {:.3f} sec --------'.format(time.time() - start_time))
