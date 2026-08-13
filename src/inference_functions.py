import torch

def evaluate(model, dl, device, loss_fn=None):
    model.eval()

    total_loss = torch.tensor(0.0, device=device) if loss_fn else None
    correct = torch.tensor(0, device=device)
    total = 0
    with torch.inference_mode():
        for X_batch, y_batch in dl:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            preds = model(X_batch)

            if loss_fn:
                total_loss += loss_fn(preds, y_batch)

            # accumulate correct predictions and total predictions
            correct += (preds.argmax(dim=1) == y_batch).sum()
            total += y_batch.shape[0]

    # compute accuracy
    accuracy =  (correct / total).item()

    if loss_fn:
        return accuracy, (total_loss / len(dl)).item()

    return accuracy
