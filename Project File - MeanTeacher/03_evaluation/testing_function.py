# Test the model
def test_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Get predictions
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f'Testing batch {batch_idx}/{len(test_loader)}')
    
    # Calculate metrics
    test_loss /= len(test_loader)
    accuracy = 100.0 * sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)
    
    return all_predictions, all_targets, test_loss, accuracy