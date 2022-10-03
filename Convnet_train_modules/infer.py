import torch
import timm
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image


class Model(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, checkpoint_path='src/runs/exp37/weights/best.pt', num_classes=11, image_size=224):
        super().__init__()
        model = timm.create_model(
            model_name=model_name, pretrained=True, 
            checkpoint_path=checkpoint_path, 
            num_classes=num_classes
        )
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.image_size = image_size
        
    def forward(self, image):
        image = transforms.functional.resize(image, size=[self.image_size, self.image_size])
        image = image / 255.0
        image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        
        with torch.no_grad():
            output = self.model(image)
        output = self.pool(output)
        output = torch.nn.functional.normalize(output)
        return output
    

if __name__ == "__main__":
    model = Model()
    image = Image.open('src/raw/artwork/image0001.jpeg').convert("RGB")
    convert_to_tensor = transforms.Compose([transforms.PILToTensor()])
    input_tensor = convert_to_tensor(image)
    input_batch = input_tensor.unsqueeze(0)
    embedding = torch.flatten(model(input_batch)[0]).cpu().data.numpy()
    model_scripted = torch.jit.script(model)
    model_scripted.save('saved_model.pt')