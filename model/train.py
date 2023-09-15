import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from .network import MNISTClassifier 
from .preprocess import get_data_loaders

def objective(trial):
    # Hyperparameters to be optimized
    batch_size = trial.suggest_int('batch_size', 32, 128)
    
    # Load data using the imported function
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)

    # Initialize model, criterion, and optimizer
    model = MNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    
    # Hyperparameters to be optimized
    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    num_epochs = 3  # Keep it small for hyperparameter optimization
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation accuracy (can be a separate dataset)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:  # Using validation loader for accuracy calculation
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy
