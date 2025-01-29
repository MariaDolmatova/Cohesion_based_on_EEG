# Define the CNN model
class CNN_big(nn.Module):
    def __init__(self):
        super(CNN_big, self).__init__()
        self.conv1 = nn.Conv1d(40, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(32 * 30, 32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# Training function with early stopping


def train_model_early_stopping(model, train_loader, val_loader, optimizer, criterion, epochs, patience, min_delta):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    best_val_loss = float("inf")  # so that we can update the loss
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_preds, train_labels = [], []  # For calculating the scores...
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            output = model(X_batch).squeeze()  # output = batch size
            y_batch = y_batch.squeeze().float()

            loss = criterion(output, y_batch)

            optimizer.zero_grad()  # cleaning gradient
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)  # Total loss
            predicted = (output >= 0.5).long()
            train_labels.extend(y_batch.tolist())
            train_preds.extend(predicted.tolist())

        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_epoch_acc = accuracy_score(train_labels, train_preds)
        train_epoch_f1 = f1_score(train_labels, train_preds, average="binary")

        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)
        train_f1s.append(train_epoch_f1)

        model.eval()  # testing time
        val_running_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                output_val = model(X_val).squeeze()
                y_val = y_val.squeeze().float()
                val_loss = criterion(output_val, y_val)

                val_running_loss += val_loss.item() * X_val.size(0)
                predicted_val = (output_val >= 0.5).long()
                val_preds.extend(predicted_val.tolist())
                val_labels.extend(y_val.tolist())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = accuracy_score(val_labels, val_preds)
        val_epoch_f1 = f1_score(val_labels, val_preds, average="binary")

        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        val_f1s.append(val_epoch_f1)

        if val_epoch_loss < best_val_loss - min_delta:
            best_val_loss = val_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  >> Early stopping at epoch {epoch + 1}!")
                break

    model.load_state_dict(best_model_weights)
    return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s
