from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
# from model.model import CNNStocksModule

class CNNStocksModule(nn.Module):
    OUT_CHANNELS = 30  # Number of CNN channels
    KERNEL_SIZE = 10  # Size of CNN kernel

    def __init__(self, window_length: int):
        super(CNNStocksModule, self).__init__()

        assert window_length >= self.KERNEL_SIZE
        self.cnn = nn.Conv1d(
            1,  # In channel size
            self.OUT_CHANNELS,
            self.KERNEL_SIZE
        )

        num_scores = window_length - self.KERNEL_SIZE + 1

        # MaxPool kernel size is set such that we only output one value for each row/channel
        self.pool = nn.MaxPool1d(num_scores)

        self.linear = nn.Linear(self.OUT_CHANNELS, 1, bias=True)

    def forward(self, x):
        out = self.cnn(x.unsqueeze(1))
        out = self.pool(out).squeeze()
        out = torch.softmax(out, dim=1)
        out = self.linear(out).squeeze()
        return out
def train_model(df, window_size=30):
    df_returns = df['Close'].pct_change().dropna().values
    X, y = [], []

    for i in range(len(df_returns) - window_size - 1):
        X.append(df_returns[i:i + window_size])
        y.append(1 if df_returns[i + window_size] > 0 else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    years = pd.to_datetime(df.index[window_size + 1:]).year
    grouped_metrics = []

    unique_years = sorted(set(years))
    ticker_name = df.name if hasattr(df, 'name') else "stock"

    for year in unique_years:
        idx = years == year
        if idx.sum() < 10:
            continue

        X_train = torch.tensor(X[~idx])
        y_train = torch.tensor(y[~idx])
        X_test = torch.tensor(X[idx])
        y_test = torch.tensor(y[idx])

        model = CNNStocksModule(window_size)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train).squeeze()
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test).squeeze()
            test_preds = torch.sigmoid(test_logits).numpy()
            test_labels = y_test.numpy()

            pred_labels = (test_preds >= 0.5).astype(int)

            # Handle edge case: constant predictions
            if len(set(pred_labels)) > 1:
                r2 = np.corrcoef(pred_labels, test_labels)[0, 1] ** 2
                corr = np.corrcoef(pred_labels, test_labels)[0, 1]
            else:
                r2 = -1
                corr = 0

            mae = np.mean(np.abs(pred_labels - test_labels))

            print(f"{year} R²: {r2:.4f}, Corr: {corr:.4f}, MAE: {mae:.4f}")
            grouped_metrics.append({"Year": year, "R_Squared": r2, "Correlation": corr, "MAE": mae})

    # Save to CSV
    df_metrics = pd.DataFrame(grouped_metrics)
    results_filename = f"results_{ticker_name.replace('.', '_')}.csv"
    df_metrics.to_csv(results_filename, index=False)
    print(f"\n✅ Saved results to {results_filename}")
