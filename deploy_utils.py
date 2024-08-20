import torch
import timm
import numpy as np
import onnx
import onnxruntime
from PIL import Image
from custom_torch_module import setup_utils

def export_onnx(model, weight_path, export_path, input_size:list, device="cpu"):
    """
    Save model with weights as onnx file
    """
    torch.set_default_device(device)
    
    weights = torch.load(f=weight_path)
    model.load_state_dict(weights)
    model.eval()
    
    example_input = torch.empty(input_size)
    
    # 모델 변환
    torch.onnx.export(model,
                      example_input,
                      export_path,
                      export_params=True,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    print("[info] The model has succesfull exported.")
    print(f"[info] File Path : {export_path}")

class Onnx_deploy_model():
    def __init__(self, model_path, img_size):
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.transform = setup_utils.build_transform(img_size)
        
    def run(self, x, return_prob=True):
        """
        input : Image(PIL or Numpy)
        output : prob or logits
        """
        # img = Image.open(x).convert("RGB")
        x = self.transform(x).unsqueeze(dim=0)
        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        
        if return_prob:
            ort_outputs = softmax(ort_outputs)
        
        return ort_outputs.squeeze()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def softmax(x):
    max_num = np.max(x)   
    exp_a = np.exp(x - max_num) # to prevent OverFlow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y



