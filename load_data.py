# load data and pre-post precess
import os
from glob import glob
from load_model import load_model, wind_levels, spacing
import numpy as np

from monai.transforms import *
from monai.data import Dataset, DataLoader


pre_process = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CopyItemsd(keys=["image"], times=2, names=["img_2", "img_3"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=wind_levels[0][0], a_max=wind_levels[0][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ScaleIntensityRanged(
            keys=["img_2"], a_min=wind_levels[1][0], a_max=wind_levels[1][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ScaleIntensityRanged(
            keys=["img_3"], a_min=wind_levels[2][0], a_max=wind_levels[2][1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ConcatItemsd(['image', 'img_2', 'img_3'], name='image'),
        DeleteItemsd(['img_2', 'img_3']),

        
        Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear")),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RPI"),  # RPI, RAS   # error using orientationd for case 8 and 3
        # CastToTyped(keys=["image"], dtype=(np.float32)),
        # EnsureTyped(keys=["image", "label"]),
        # ToTensord(keys=["image"]),
    ]
)

post_process = Compose([
    EnsureTyped(keys="pred"),
    Activationsd(keys="pred", softmax=True),
    # KeepLargestConnectedComponent(applied_labels=1),
    Invertd(
        keys="pred",
        transform=pre_process,  # inference_transforms, test_transforms_3c
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
])


# reading raw head information from input scans
read_raw_image = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"],
                             a_min=0, a_max=100,
                             b_min=0., b_max=1., clip=True),
        EnsureTyped(keys=["image"])
    ]
)



def load_data(input_folder=''):
    images_nii = sorted(glob(os.path.join(input_folder, "*.nii*")))
    test_file_list = [{'image': img} for img in images_nii]

    test_dataset = Dataset(data=test_file_list, transform=pre_process)
    dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)  # only support bs=1, num_worker=0 for support Mac OS
    return test_file_list, dataloader