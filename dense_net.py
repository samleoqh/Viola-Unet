import os
import torch
import torch.nn as nn
from torchvision.models import DenseNet

import numpy as np
from PIL import Image

from img_utils import norm, cropping
from pytorch_grad_cam import  EigenCAM #,GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


pre_trained_models = {
    'dense_net121': {
        'f0':'./best_ckpt2/densenet121_f0.pth',
        # 'f1':'./best_ckpt2/densenet121_f1.pth',
        'f2':'./best_ckpt2/densenet121_f2.pth',
        # 'f3':'./best_ckpt2/densenet121_f3.pth',
        'f4':'./best_ckpt2/densenet121_f4.pth',
    }
}


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = DenseNet(32, (6, 12, 24, 16), 64).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)

        return self.sigmoid(x)


def load_detect_modes(net="dense_net121", device='cpu'):
    model_list = []
    for k in pre_trained_models[net].keys():
        if net == "dense_net121":
            model = DenseNet121()
        else:
            print('Not support this model {}!'.format(net))

        pretrain = torch.load(pre_trained_models[net][k], map_location=device)
        state_dict = pretrain["state_dict"]
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=False)
        model_list.append(model.eval().to(device))

    return model_list


def get_cams(models):
    cams = []
    for cam_model in models:
        # cam_model = models[1]
        dict(cam_model.named_modules())
        target_layers = [cam_model.densenet121.denseblock4.denselayer16.conv1]
        # cam = GradCAMPlusPlus(model=cam_model, target_layers=target_layers)
        cams.append(EigenCAM(model=cam_model, target_layers=target_layers))

    targets = [ClassifierOutputTarget(0)]
    return cams, targets


def detect_ich(models, cams, targets, test_file, pre_process_cls, device, batch_size = 12):
    with torch.no_grad():
        path, filename = os.path.split(test_file['image'])
        img_data = pre_process_cls(test_file)
        images = img_data["image"].permute(3,1,2,0)
        # print(images.shape)
        n_slices = images.size(0)
        img = images.numpy()
        imgs = []
        for j in range(0, n_slices):
            imgs.append(torch.from_numpy(norm(img=img[j, :, :, :]*255.)))
        images = torch.stack(imgs).permute(0,3,1,2).to(device)

        predictions = torch.FloatTensor([0, 0, 0, 0, 0, 0]).to(device)

        all_outputs = []
        for index in range(0, images.shape[0], batch_size):
            batch = images[index:min(index + batch_size, images.shape[0]), :]
            output = 0
            for m in models:
                output += m(batch)
            all_outputs.append(output/len(models))
        all_outputs = torch.cat(all_outputs, dim=0)
        # output = output/len(models)
        all_outputs[all_outputs > 0.5] = 1.
        all_outputs[all_outputs <= 0.5] = 0.
        predictions += all_outputs.sum(0)
        visual_imgs = []
        if predictions.sum() > 0:
            # print('patient {} ich- {}'.format(filename, predictions.data.cpu().numpy()))
            i_dxs = torch.nonzero(all_outputs.sum(1))[:,0]
            i_dx = i_dxs[i_dxs.shape[0]//2]
            # print('slice-{}: {}'.format(i_dx.cpu().numpy(), all_outputs[i_dx]))
            im = Image.fromarray(np.uint8(img[i_dx, :, :, 0] * 255), 'L')  # 'RGB'
            visual_imgs.append(im)
            # im.save('./demo/p{}_slice_{}.png'.format(filename, i_dx.cpu().numpy()))
            grayscale_cams = []
            with torch.enable_grad():
                # input= torch.tensor(images[[i_dx], :, :, :].clone(), requires_grad=True)
                for cam in cams:
                    grayscale_cams.append(cam(input_tensor=images[[i_dx], :, :, :], targets=targets)[0, :])
            # In this example grayscale_cam has only one image in the batch:
            for k, grayscale_cam in enumerate(grayscale_cams):
                visualization = show_cam_on_image(cropping(img[i_dx, :, :, :]), grayscale_cam, use_rgb=True)
                visual_imgs.append(Image.fromarray(visualization))
                # Image.fromarray(visualization).save('./demo/p{}_slice_{}_cam_model-{}.png'.format(filename, i_dx.cpu().numpy(), k))
        else:
            i_dx = -1

        pred_arry = predictions.data.cpu().numpy()
        ich_pred = dict()
        output = ['any_ICH', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH']
        for i, k in enumerate(output):
            ich_pred[k] = int(pred_arry[i])
        return ich_pred, i_dx, visual_imgs
