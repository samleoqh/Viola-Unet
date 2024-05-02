import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def norm(img, mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0):
    img = cropping(img)
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def cropping(image, ratio=0.85, size=(256, 256)):
    height, width, _ = image.shape
    locations = cv2.findNonZero(image[:, :, 2])

    if locations is not None:
        top_left = locations.min(axis=0)[0]
        bottom_right = locations.max(axis=0)[0]

        crop_x, crop_y = top_left[0], top_left[1]
        crop_width, crop_height = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]

        target_h = int(height * 0.9)
        target_w = int(width * 0.8)
        if target_w > crop_width: target_w = crop_width
        if target_h > crop_height: target_h = crop_height

        start_x = (width - target_w) // 2
        if start_x < crop_x: start_x = crop_x
        start_y = (height - target_h) // 2
        if start_y < crop_y: start_y = crop_y
        # zeros = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width, :]
        # zeros = image[crop_y:crop_y + target_h, start_x:start_x+target_w, :]
        zeros = image[start_y:start_y + target_h, start_x:start_x + target_w, :]

    else:
        target_h = int(height * ratio)
        target_w = int(width * ratio)

        start_x = (width - target_w) // 2
        start_y = (height - target_h) // 2

        zeros = image[start_y:start_y + target_h, start_x:start_x + target_w, :]
    # zeros = cv2.resize(zeros, (width,height))
    zeros = cv2.resize(zeros, size)
    return zeros


def visualize_cam(visual_imgs, patient, n_slice, path):
    n_plot = len(visual_imgs)
    fig, axes = plt.subplots(1, n_plot, figsize=(n_plot * 9, 12), sharex=False, sharey=False)
    for i, ax in enumerate(axes.flatten()):
        if i == 0:
            ax.set_ylabel(f'PRI (pixdim 1, 1, 3 mm), z-slice:{n_slice}', fontsize=24)
            ax.set_title(f"input_slice", fontsize=24)
            ax.imshow(visual_imgs[i], cmap='gray')
        else:
            ax.set_title(f"CAM_of_model-{i}", fontsize=24)
            ax.imshow(visual_imgs[i])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_name = '{0}_slice-{1}-cam.png'.format(patient, n_slice)
    plt.savefig(os.path.join(path, fig_name))
    plt.close(fig)
