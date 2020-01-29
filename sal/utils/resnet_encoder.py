__all__ = ['ResNetEncoder', 'resnet50encoder']
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, Bottleneck
from .pytorch_fixes import adapt_to_image_domain
from torch.autograd import Variable
import torch.nn as nn
class ResNetEncoder(ResNet):
    def forward(self, x):
        s0 = x
        x = self.conv1(s0)
        x = self.bn1(x)
        s1 = self.relu(x)
        x = self.maxpool(s1)
        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)

        s5 = self.layer4(s4)

        x = self.avgpool(s5)
        sX = x.view(x.size(0), -1)
        sC = self.fc(sX)

        return s0, s1, s2, s3, s4, s5, sX, sC


def resnet50encoder(pretrained=True,  **kwargs):
    """Constructs a ResNet-50 encoder that returns all the intermediate feature maps.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    
    model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model