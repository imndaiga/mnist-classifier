import torch
from time import time


def train(model, epochs, train_loader, optimizer, criterion):
    print('\n\nTraining started!...')
    startTime = time()
    for epoch in range(1, epochs+1):
        for idx, batch in enumerate(train_loader):
            inputs, labels = batch
            preds = model(inputs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f'Epoch: {epoch}\tBatch: {idx}\tLoss: {loss.item()}'.expandtabs())
    duration = time() - startTime
    print(f'Training complete! Final loss: {loss.item()}')

    return loss, duration


def calc_accurary(model, test_loader):
    print('\n\nTesting started!...')
    num_correct = 0
    num_examples = len(test_loader.dataset)
    model.eval()

    for inputs, labels in test_loader:
        preds = model(inputs)
        preds = torch.max(preds, axis=1)
        preds = preds[1]
        num_correct += int(sum(preds == labels))
    
    accuracy = num_correct/num_examples * 100
    print(f'Testing complete! Accuracy: {accuracy}')

    return accuracy
