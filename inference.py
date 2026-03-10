import torch
from model import Model
from torchvision import transforms, datasets
from PIL import Image
from dataloader import Dataset
from torch.utils.data import DataLoader, random_split
import numpy as np

if __name__ == '__main__':
    
    model = Model(in_channels=3, num_classes=11)
    model.load_state_dict(torch.load('model_cloud.pt', weights_only=False)['model_state_dict'])

    transform = transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
        ])
    
    images_dir = '/Users/oliverlewis/Work/Clouds/CCSN_v2'
    classes = datasets.ImageFolder(images_dir, transform=transform).classes

    dataset = Dataset(root_dir=images_dir)

    img = Image.open("/Users/oliverlewis/Work/MLOps/ML_experiments/test.jpg").convert("RGB")
    tensor = transform(img)
 
    tensor = tensor.unsqueeze(0)
    predict = model(tensor)

    score = torch.nn.functional.softmax(predict, dim=1)

    score = score.detach().numpy()[0]

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(classes[np.argmax(score)], 100 * np.max(score))
    )