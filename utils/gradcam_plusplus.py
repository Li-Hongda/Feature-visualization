import cv2
import numpy as np
from .activations_and_gradients import ActivationsAndGradients
from .image_process import scale_cam_image

class GradCAMPP:
    def __init__(self,
                model,
                layers
                ):
        self.model = model.eval()
        self.model = self.model.cuda()
        self.target_layers = layers
        # self.reshape_transform = reshape_transform
        self.activations_and_gradients = ActivationsAndGradients(
            self.model,self.target_layers
        )

    @staticmethod
    def get_cam_weights(activations,grads):

        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights


    @staticmethod
    def get_loss(output,target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i,target_category[i]]
        return loss

    def get_cam_image(self,activations,grads):
        weights = self.get_cam_weights(activations,grads)
        weighted_activations = weights[:,:,None,None] * activations
        cam = weighted_activations.sum(axis = 1)
        return cam

    @staticmethod
    def get_target_width_height(input):
        width,height = input.size(-1),input.size(-2)
        return width,height
    
    def compute_cam_per_layer(self,input_tensor):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_gradients.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_gradients.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []

        for layer_activations,layer_grads in zip(activations_list,grads_list):
            cam = self.get_cam_image(layer_activations,layer_grads)
            cam[cam < 0] = 0
            scaled = scale_cam_image(cam,target_size)
            # cam_per_target_layer.append(scaled[None,:,:])
            cam_per_target_layer.append(scaled[:,None,:])
        
        return cam_per_target_layer

    def aggregate_multi_layers(self,cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer,axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer,0)
        result = np.mean(cam_per_target_layer,axis=1)
        return scale_cam_image(result)

    def __call__(self,input_tensor,target_category = None):
        output = self.activations_and_gradients(input_tensor)
        if isinstance(target_category,int):
            target_category = [target_category] * input_tensor.size(0)
        
        if target_category is None:
            target_category = np.argmax(output.cpu.data.numpy(),axis=-1)
            print(f"category id:{target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output,target_category)
        loss.backward(retain_graph = True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_gradients.release()





        






            
