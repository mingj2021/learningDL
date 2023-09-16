import torch
import torchvision
from torch import Tensor, nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchsummary import summary
from collections import namedtuple

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = {}
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.block1 = model.features[0]
        self.block2 = model.features[1]
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return (x,)



if __name__ == '__main__':
    model = FeatureExtractor()
    # summary(model, (3, 512, 512))

    dummy_input = torch.randn(1, 3, 512, 512)

    input_names = [ "images" ]
    output_names = [ "output0" ]

    script_module = torch.jit.trace(model, dummy_input)
    script_module.save('efficientnet_v2_s.pt')
    # torch.onnx.export(model, dummy_input, "FeatureExtractor.onnx", verbose=True, input_names=input_names, output_names=output_names)
