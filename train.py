import torch, os, wandb
from model import Model
from tqdm import tqdm
import numpy as np
from dataloader import Dataset
from torch.utils.data import DataLoader, random_split

def process_epoch(dataloader, flag, optimizer):
    if flag == 'train':
        model.train()
    else:
        model.eval()

    batch_iter = tqdm(enumerate(dataloader), total=len(dataloader))

    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []

    for j, (inputs, labels) in batch_iter:
    
        loss = None

        if flag == 'train':
            optimiser.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        if flag == 'train':
            loss.backward()
            optimiser.step()

        losses.append(loss.item())

    return np.ma.masked_invalid(losses).mean()

if __name__ == '__main__':

    seed = 1
    torch.manual_seed(seed)

    # wandb.init(project='mlops_test', name='a')
    # wanb.run.log_code(".")

    batch_size = 64
    epochs = 100

    images_dir = '/Users/oliverlewis/Work/Clouds/CCSN_v2'

    dataset = Dataset(root_dir=images_dir)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size   = int(0.1 * total)
    test_size  = total - train_size - val_size   

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=32, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=32, shuffle=False)

    example_input, example_output = next(iter(train_loader))

    model = Model(in_channels=len(example_input[0]), num_classes=len(dataset.dataset.classes))

    optimiser = torch.optim.Adam(model.parameters())

    test_losses = []

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1} | Train")

        model.train()

        train_loss = process_epoch(train_loader, 'train', optimiser)

        model.eval()

        test_loss = process_epoch(test_loader, 'test', optimiser)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        if epoch > 0 and test_loss < min(test_losses[:-1]):
            torch.save(model.state_dict(), f'model_cloud.pt')