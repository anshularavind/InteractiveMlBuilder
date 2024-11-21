import torch
import torch.nn as nn

def train_model(model, epochs=10):
    # Define the loss function and optimizer
    dataset = model.dataset
    criterion = dataset.criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    # Train the model
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataset.train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, 5, i+1, len(dataset.train_loader), loss.item()))

    # Test
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataset.test_loader:
            outputs = model(images)
            new_correct, new_total = dataset.get_eval_numbers(outputs, labels)
            correct += new_correct
            total += new_total

        accuracy = correct / total
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * accuracy))

    return model, accuracy
