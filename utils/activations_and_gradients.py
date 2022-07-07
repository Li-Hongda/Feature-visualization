import os

class ActivationsAndGradients:
    """
        从目标层中提取感兴趣的类别激活值和梯度信息
    """
    def __init__(self,model,layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        for layer in layers:
            self.handles.append(
                layer.register_forward_hook(self.save_activations)
            )
            self.handles.append(
                layer.register_backward_hook(self.save_gradients)
            )
    
    def save_activations(self,module,input,output):
        activation = output
        # if self.reshape_transform is not None:
        #     activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradients(self,module,grad_input,grad_output):
        grad = grad_output[0]
        # if self.reshape_transform is not None:
        #     grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self,x):
        self.gradients = []
        self.activations = []
        return self.model(x.cuda())

    def release(self):
        for handle in self.handles:
            handle.remove()