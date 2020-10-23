from datetime import datetime

import torchvision
import torch

from lib.dataset import Dataset
from lib.models import CNN
from lib.utils import train, calc_accurary

if __name__ == '__main__':
    dataDir = './data'
    datasetName = 'MNIST'
    modelsDir = './checkpoints'
    lr = 0.0001
    epochs = 2

    data = Dataset(datasetName, dataDir, torchvision.transforms.ToTensor())
    train_loader, val_loader, test_loader = data.loaders(
        train_size=50000, val_size=10000)

    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    loss, duration = train(cnn, epochs, train_loader, optimizer, criterion)

    currtime = datetime.now()
    torch.save(cnn, f'{modelsDir}/{currtime.strftime("%d%m%y%H%M%S")}.cpt')

    accuracy = calc_accurary(cnn, test_loader)
    print(f'\n\nAccuracy: {accuracy}\tTrain Time: {duration}\tLoss: {loss.item()}'.expandtabs())
