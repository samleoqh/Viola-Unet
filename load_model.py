import os
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.transforms.utils import map_spatial_axes
from monai.data import decollate_batch


from viola_unet import ViolaUNet
from monai.networks.nets import DynUNet


wind_levels = [[15,85], [-15, 200],[-100, 1300]]
spacing = [1., 1., 3.] 
patch_size = (160, 160, 32)
patch_overlap = 0.5 # 0.5 for last validation submission
sw_bt_size=1



# # re-fine-tuned version after the last validation submission weights
net_weights = {
    "nnUNet": "./best_ckpt2/nnu_v2.pt",      # best 192x192x32
    # "ViolaUNet_ss": "./viola2_ckpt/ss_f0.pt",      # best 160x32, 0.66
    "Viola_s": "./best_ckpt2/viola_v2.pt",      # best 192x192x32
}


def load_model(network="Viola_s", device='cpu'):
    if network == "Viola_s":
        model = ViolaUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=2,
            kernel_size=[[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            filters=(16, 32, 32, 64, 128),
            dec_filters=(16, 16, 32, 64),
            norm_name=("BATCH", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout=0.2,
            deep_supervision=False,
            deep_supr_num=2,
            res_block=True,
            trans_bias=True,
            viola_att=True,
            gated_att=False,
            sum_deep_supr=False,
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
    else:
        print("Not support the network currently - ", network)
        return None
    
    if net_weights[network] is not None:  # 
        pretrain = torch.load(net_weights[network], map_location=device)
        model.load_state_dict(pretrain)
        print("model {} loaded successfully!".format(network))
    else:
        print("model {} loaded failure, couldn't find the trained weights!".format(network))
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
    image = nib.load(imagepath)
    segmentation = nib.Nifti1Image(segmentation, image.affine)
    qform = image.get_qform()
    segmentation.set_qform(qform)
    sfrom = image.get_sform()
    segmentation.set_sform(sfrom)
    nib.save(segmentation, os.path.join(outputpath, filename))
    print('Segmentation was saved to the file:', filename)


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
