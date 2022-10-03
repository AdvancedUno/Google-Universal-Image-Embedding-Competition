import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Model(nn.Module):
    def __init__(self, model_name_or_path, pretrained=True, n_classes=11):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.net = timm.create_model(model_name_or_path, pretrained=pretrained)
        self.out_features = self.net.fc.in_features
        self.global_pool = GeM(p_trainable=True)
        self.head = nn.Linear(in_features=self.out_features, out_features=n_classes)
        
    def forward(self, image):
        x = self.net.forward_features(image)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logits = self.head(x)
        return {
            'logits':logits,
            'embeddings':x
        }