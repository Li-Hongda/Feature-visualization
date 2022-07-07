import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import cv2
from utils.gradcam import GradCAM
from utils.gradcam_plusplus import GradCAMPP
from utils.image_process import show_cam_on_image



parser = argparse.ArgumentParser()

parser.add_argument('--target_model',help='choose the model you want to visualize')
parser.add_argument('--img_path',default='demo/demo.png')
parser.add_argument('--result_path',default='result/')
parser.add_argument('--method',default='grad')
# parser.add_argument('--batch_size',type=int,default=10)
parser.add_argument('--checkpoint',help='the path of checkpoint for model')
parser.add_argument('--target_category',default=281,help='the target class for visualize')
parser.add_argument('--mode',type = str,choices=['demo','visualize'],help='visualize a single image for demo or a whole dataset for evaluation')
args = parser.parse_args()


def main():
    visualize_method = args.method
    target_model = models.resnet50(pretrained=True)
    target_layers = [target_model.layer4]

    # target_model = models.vgg16(pretrained=True)
    # target_layers = [target_model.features[-1]]

    target_image = cv2.imread(args.img_path,1)[:, :, ::-1]
    target_image_name = args.img_path.split('/')[-1].split('.')[0] + '_' + f"{args.method}" + '_' + f"{args.target_category}"+'_result.jpg'
    target_image = np.array(target_image,dtype=np.uint8)
    pre_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,0.456,0.406] ,std = [0.229,0.224,0.225])
        ]
    )
    img = pre_transforms(target_image.copy()).unsqueeze(0)
    if args.method == 'grad':
        cam = GradCAM(model = target_model, layers = target_layers)

    elif args.method == 'gradpp':
        cam = GradCAMPP(model = target_model, layers = target_layers)
    grayscale_cam = cam(input_tensor = img, target_category = args.target_category)
    grayscale_cam = grayscale_cam[0,:]
    visualization = show_cam_on_image(target_image.astype(np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    visualization = cv2.cvtColor(visualization,cv2.COLOR_RGB2BGR)
    # cv2.imshow('demo',visualization)
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    cv2.imwrite(os.path.join(args.result_path,target_image_name),visualization)


    

if __name__ == '__main__':
    main()
