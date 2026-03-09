import torch

class Model(torch.nn.Module):

    def __init__(self, in_channels, num_classes):
        super(Model, self).__init__()
        self.conv1         = torch.nn.Conv2d(in_channels, 8,  kernel_size=3, padding=1)
        self.pool          = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2         = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  
        self.fc1           = torch.nn.Linear(16, num_classes)    

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.adaptive_pool(x)   
        x = torch.flatten(x, 1)     
        x = self.fc1(x)             
        return x