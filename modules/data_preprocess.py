import torch

def calculate_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataset)
    std /= len(dataset)

    return mean, std