import torch
import torch.nn as nn
import time


def train_model(model, epochs=10):
    # Define the loss function and optimizer
    dataset = model.dataset
    criterion = dataset.criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

    print('Training model...')
    model.add_output_logs('Training model...')

    start = time.time()

    # Train the model
    model.train()
    output_increment = len(dataset.train_loader) // 10
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (x, y) in enumerate(dataset.train_loader):
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            epoch_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % output_increment == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, epochs, i+1, len(dataset.train_loader), loss.item()))
        epoch_loss /= len(dataset.train_loader)
        model.add_loss_logs(epoch_loss)

        # Test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in dataset.test_loader:
                outputs = model(x)
                new_correct, new_total = dataset.get_eval_numbers(outputs, y)
                correct += new_correct
                total += new_total

            accuracy = correct / total

            time_delta = time.time() - start
            minutes, seconds = divmod(time_delta, 60)
            time_elapsed_str = f'Time elapsed: {int(minutes)}m {int(seconds)}s'

            if epoch + 1 != epochs:
                output_str = time_elapsed_str + f'\nEpoch #{epoch + 1} {dataset.accuracy_descriptor}: {accuracy}'
                print(output_str)
                model.add_output_logs(output_str)
            else:
                output_str = time_elapsed_str + f'\nFinal {dataset.accuracy_descriptor}: {accuracy}'
                print(output_str)
                model.add_output_logs(output_str)

    return accuracy
