import torch
import copy
import matplotlib.pyplot as plt

def train_model(net, trainloader, valloader, epochs=50, lr=0.001, patience=5):
    """
    Train the CNN with early stopping and LR scheduler
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(net.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        net.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_losses.append(running_loss / len(trainloader))
        train_accs.append(100.*correct/total)
        
        # Validation
        net.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss_epoch = val_loss / len(valloader)
        val_losses.append(val_loss_epoch)
        val_accs.append(100.*val_correct/val_total)
        
        scheduler.step(val_loss_epoch)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_losses[-1]:.3f}, "
              f"Acc {train_accs[-1]:.2f}% | Val Loss {val_loss_epoch:.3f}, "
              f"Acc {val_accs[-1]:.2f}% | LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_wts = copy.deepcopy(net.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
    
    net.load_state_dict(best_model_wts)
    
    # Plot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss over Epochs")
    
    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend(); plt.title("Accuracy over Epochs")
    plt.show()
    
    return net

def test_model(net, testloader):
    """
    Evaluate model on test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    net.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
