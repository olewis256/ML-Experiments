import torch
from torchvision import datasets, transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]