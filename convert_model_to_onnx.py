

import numpy as np
import torch
from net.cls_inception_v3 import InceptionV3
from utilities.model_util import load_pretrained_state_dict

def load_model(model_fpath, device):
  class Args(object):
    pretrained=False
    can_print=True
    in_channels=4
    num_classes=19
    ml_num_classes=20000

  class InceptionV3Simple(InceptionV3):
    def forward(self, data):
      data = super().forward({"image": data})
      return 1/(1 + torch.exp(-data['logits'])), data['feature_vector']
  
  model = InceptionV3Simple(Args(), feature_net='inception_v3', att_type='cbam')
  load_state_dict = torch.load(model_fpath, map_location=torch.device('cpu'))
  load_pretrained_state_dict(model, load_state_dict, strict=True)
  model = model.eval().to(device)
  return model


def export_onnx(model, device, file_path='bestfitting_inceptionv3_model2048.onnx'):
    import onnxruntime as ort
    # Switch the model to eval model
    model.eval()
    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 4, 128, 128)
    torch.onnx.export(model, example.to(device), file_path,
                    input_names=['image'],
                    output_names=['classes', 'features'],
                    dynamic_axes = {'image': [2, 3]}, # the width and height of the image can be changed
                    verbose=False, opset_version=11)
    ort_session = ort.InferenceSession(file_path)
    exported_results = ort_session.run(None, {'image': example.numpy().astype(np.float32)})
    with torch.no_grad():
        original_results = model(example.to(device))
    assert np.allclose(original_results[0].to('cpu').numpy(), exported_results[0]), f"{original_results[0].numpy()}, {exported_results[0]}"
    print(f'ONNX File {file_path} exported successfully')

model_path = '../../models/d0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds/fold0/12.00_ema.pth'
device = 'cuda:2'
model = load_model(model_path, device)
# Before export, we need to change the model to: 1) add sigmoid 2) make sure it take a tensor and return a tuple of outputs (i.e. probalbities and features)
export_onnx(model, device)