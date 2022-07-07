import numpy as np
import os
import cv2




def show_cam_on_image(img:np.ndarray,
                      mask:np.ndarray,
                      use_rgb:bool=False,
                      colormap:int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255*mask),colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")


    cam = heatmap + img.squeeze()
    cam = cam / np.max(cam)
    return np.uint8(255*cam)


def scale_cam_image(cam,target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-8 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img,target_size)
        result.append(img)
    result = np.float32(result)

    return result