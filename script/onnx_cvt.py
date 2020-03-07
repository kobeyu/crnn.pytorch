import torch
import torchvision
import models.crnn as crnn

dummy_input = torch.randn(1, 1, 32, 100, device='cpu')

model_path = './data/crnn.pth'
model = crnn.CRNN(32, 1, 37, 256)
model_onnx_path = "./crnn.onnx"
output = torch.onnx.export(model, dummy_input, model_onnx_path, verbose=False)
