def train(model, loss_fn, optimizer, train_dl, device):
    model.train()

    total_loss = torch.tensor(0.0, device=device)
    for X_batch, y_batch in train_dl:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # 1. forward pass
        preds = model(X_batch)

        # 2. compute loss
        loss = loss_fn(preds, y_batch)

        # 3. reset gradients
        optimizer.zero_grad()

        # 4. compute gradients
        loss.backward()

        # 5. optimizer step
        optimizer.step()

        total_loss += loss

    # return average epoch loss
    return (total_loss / len(train_dl)).item()