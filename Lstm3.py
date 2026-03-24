# %%

import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


NODES = ["275", "363", "208", "207", "72"]

LEVEL_STEP = 24
RAIN_STEP = 24
FUTURE_RAIN = 36
PREDICT_STEPS = 36
SEQ_LEN = 36

HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 0.001
EPOCHS = 10
BATCH_SIZE = 128
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0:
        return float("nan")
    return 1 - np.sum((obs - sim) ** 2) / denominator


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, NUM_LAYERS, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, PREDICT_STEPS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def create_dataset(level: np.ndarray, rain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    total = len(level) - SEQ_LEN - PREDICT_STEPS + 1
    for i in range(max(total, 0)):
        x = np.zeros((SEQ_LEN, 3), dtype=np.float32)
        x[-LEVEL_STEP:, 0] = level[i + SEQ_LEN - LEVEL_STEP : i + SEQ_LEN].flatten()
        x[-RAIN_STEP:, 1] = rain[i + SEQ_LEN - RAIN_STEP : i + SEQ_LEN].flatten()
        x[:, 2] = rain[i + SEQ_LEN : i + SEQ_LEN + FUTURE_RAIN].flatten()
        y = level[i + SEQ_LEN : i + SEQ_LEN + PREDICT_STEPS].flatten()
        xs.append(x)
        ys.append(y.astype(np.float32))

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def find_excel_file(base_dir: Path, keywords: list[str]) -> Path:
    candidates = sorted(
        path for path in base_dir.glob("*.xlsx") if not path.name.startswith("~$")
    )
    for path in candidates:
        if all(keyword in path.name for keyword in keywords):
            return path
    raise FileNotFoundError(f"Could not find Excel file with keywords: {keywords}")


def normalize_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    columns = {str(column): column for column in df.columns}

    if "datetime" in columns:
        df = df.rename(columns={columns["datetime"]: "datetime"})
    elif "date" in columns:
        df = df.rename(columns={columns["date"]: "datetime"})
    elif all(part in columns for part in ["年", "月", "日", "小时"]):
        df["datetime"] = pd.to_datetime(
            df[["年", "月", "日", "小时"]].rename(columns={"小时": "hour"})
        )
    else:
        raise KeyError("Could not identify a datetime column.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def detect_rain_column(df: pd.DataFrame, node_columns: set[str]) -> str:
    for column in df.columns:
        column_str = str(column)
        if column_str in node_columns:
            continue
        if any(token in column_str.lower() for token in ["rain", "mm"]):
            return column
        if any(token in column_str for token in ["雨", "降水", "降雨"]):
            return column

    for column in df.columns:
        column_str = str(column)
        if column_str not in node_columns:
            return column

    raise KeyError("Could not identify the rain column.")


def load_merged_data(base_dir: Path) -> tuple[pd.DataFrame, str]:
    flow_file = find_excel_file(base_dir, ["液位"])
    rain_file = find_excel_file(base_dir, ["5分钟", "降雨"])

    df_flow = pd.read_excel(flow_file)
    df_rain = pd.read_excel(rain_file)

    df_flow = normalize_datetime_column(df_flow)
    df_rain = normalize_datetime_column(df_rain)

    df = pd.merge(df_flow, df_rain, on="datetime", how="inner")
    df = df.sort_values("datetime").set_index("datetime")
    df.columns = df.columns.astype(str)

    missing_nodes = [node for node in NODES if node not in df.columns]
    if missing_nodes:
        raise KeyError(f"Missing node columns: {missing_nodes}")

    rain_column = detect_rain_column(df, node_columns=set(NODES))
    return df, rain_column


def inverse_transform_2d(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> LSTMModel:
    model = LSTMModel(3, HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    num_batches = int(np.ceil(len(x_train_t) / BATCH_SIZE))
    if num_batches == 0:
        raise ValueError("Training set is empty. Please check the sequence length and split size.")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * BATCH_SIZE
            end = (i + 1) * BATCH_SIZE
            x_batch = x_train_t[start:end]
            y_batch = y_train_t[start:end]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {epoch_loss / num_batches:.6f}")

    return model


def plot_node_prediction(
    node: str,
    time_index: pd.Index,
    true_series: np.ndarray,
    pred_series: np.ndarray,
) -> None:
    plt.figure(figsize=(15, 5), dpi=300)
    plt.plot(time_index, true_series, color="black", linewidth=2.2, label="Real")
    plt.plot(time_index, pred_series, color="#d62728", linewidth=1.8, label=f"Pred Node {node}")
    plt.title(f"Node {node} Water Level Prediction (t+1)")
    plt.xlabel("Datetime")
    plt.ylabel("Water Level (m)")
    plt.legend(fontsize=10)
    plt.xticks(rotation=30)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_random_hours(
    node: str,
    time_index: pd.Index,
    true_series: np.ndarray,
    pred_series: np.ndarray,
    num_hours: int = 3,
) -> None:
    steps_per_hour = 12
    max_start = len(true_series) - steps_per_hour + 1
    if max_start <= 0:
        return

    num_samples = min(num_hours, max_start)
    random_starts = np.random.choice(range(max_start), num_samples, replace=False)

    plt.figure(figsize=(15, 8), dpi=300)
    for j, start in enumerate(sorted(random_starts)):
        end = start + steps_per_hour
        plt.subplot(num_samples, 1, j + 1)
        plt.plot(time_index[start:end], true_series[start:end], color="black", linewidth=2.5, label="Real")
        plt.plot(time_index[start:end], pred_series[start:end], color="#d62728", linewidth=1.8, label="Pred")
        plt.title(f"Node {node} - Random Hour {j + 1}")
        plt.ylabel("Water Level (m)")
        plt.grid(alpha=0.2)
        if j == 0:
            plt.legend(fontsize=9)

    plt.xlabel("Datetime")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_random_segment(
    node: str,
    model: LSTMModel,
    scaler_level: MinMaxScaler,
    level_test: np.ndarray,
    rain_test_scaled: np.ndarray,
    val_end: int,
    df_index: pd.Index,
) -> None:
    max_start = len(level_test) - SEQ_LEN - PREDICT_STEPS + 1
    if max_start <= 0:
        return

    start_idx = np.random.randint(0, max_start)
    end_idx = start_idx + SEQ_LEN + PREDICT_STEPS

    level_segment = scaler_level.transform(level_test[start_idx:end_idx])
    rain_segment = rain_test_scaled[start_idx:end_idx]

    x_test_seg, _ = create_dataset(level_segment, rain_segment)
    with torch.no_grad():
        pred_seg = model(torch.tensor(x_test_seg, dtype=torch.float32)).numpy()

    pred_seg = inverse_transform_2d(scaler_level, pred_seg)[0]
    true_seg = level_test[start_idx + SEQ_LEN : start_idx + SEQ_LEN + PREDICT_STEPS, 0]
    time_index_seg = df_index[val_end + start_idx + SEQ_LEN : val_end + start_idx + SEQ_LEN + PREDICT_STEPS]

    plt.figure(figsize=(14, 5), dpi=300)
    plt.plot(time_index_seg, true_seg, color="black", linewidth=2.2, label="Real")
    plt.plot(time_index_seg, pred_seg, color="#d62728", linewidth=1.8, label="Pred")

    y_min = min(true_seg.min(), pred_seg.min())
    y_max = max(true_seg.max(), pred_seg.max())
    y_range = y_max - y_min
    margin = y_range * 0.05 if y_range > 0 else 0.1
    plt.ylim(y_min - margin, y_max + margin)

    plt.title(f"Node {node} - 3-Hour Random Test Segment Prediction")
    plt.xlabel("Datetime")
    plt.ylabel("Water Level (m)")
    plt.legend(fontsize=10)
    plt.xticks(rotation=30)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def main() -> None:
    set_seed(SEED)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "models_multi"
    output_dir.mkdir(exist_ok=True)

    df, rain_column = load_merged_data(base_dir)

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    models: dict[str, LSTMModel] = {}
    level_scalers: dict[str, MinMaxScaler] = {}
    rain_scalers: dict[str, MinMaxScaler] = {}
    results = []

    for node in NODES:
        print(f"\n======================\nTraining Node: {node}\n======================")

        level = df[node].to_numpy(dtype=np.float32).reshape(-1, 1)
        rain = df[rain_column].to_numpy(dtype=np.float32).reshape(-1, 1)

        level_train = level[:train_end]
        level_val = level[train_end:val_end]
        level_test = level[val_end:]

        rain_train = rain[:train_end]
        rain_val = rain[train_end:val_end]
        rain_test = rain[val_end:]

        scaler_level = MinMaxScaler()
        scaler_rain = MinMaxScaler()

        level_train_scaled = scaler_level.fit_transform(level_train)
        level_val_scaled = scaler_level.transform(level_val)
        level_test_scaled = scaler_level.transform(level_test)

        rain_train_scaled = scaler_rain.fit_transform(rain_train)
        rain_val_scaled = scaler_rain.transform(rain_val)
        rain_test_scaled = scaler_rain.transform(rain_test)

        x_train, y_train = create_dataset(level_train_scaled, rain_train_scaled)
        x_val, y_val = create_dataset(level_val_scaled, rain_val_scaled)
        x_test, y_test = create_dataset(level_test_scaled, rain_test_scaled)

        if len(x_val) == 0 or len(x_test) == 0:
            raise ValueError(f"Validation or test set is empty for node {node}.")

        model = train_model(x_train, y_train)

        model.eval()
        with torch.no_grad():
            pred_val = model(torch.tensor(x_val, dtype=torch.float32)).numpy()
            pred_test = model(torch.tensor(x_test, dtype=torch.float32)).numpy()

        pred_val = inverse_transform_2d(scaler_level, pred_val)
        true_val = inverse_transform_2d(scaler_level, y_val)

        rmse = np.sqrt(mean_squared_error(true_val.flatten(), pred_val.flatten()))
        mae = mean_absolute_error(true_val.flatten(), pred_val.flatten())
        score_nse = nse(true_val.flatten(), pred_val.flatten())

        print(f"Node {node} Validation -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, NSE: {score_nse:.4f}")

        torch.save(model.state_dict(), output_dir / f"model_node_{node}.pth")
        joblib.dump(scaler_level, output_dir / f"scaler_level_{node}.pkl")
        joblib.dump(scaler_rain, output_dir / f"scaler_rain_{node}.pkl")

        models[node] = model
        level_scalers[node] = scaler_level
        rain_scalers[node] = scaler_rain
        results.append([node, rmse, mae, score_nse])

        pred_test = inverse_transform_2d(scaler_level, pred_test)
        true_test = inverse_transform_2d(scaler_level, y_test)
        time_index_test = df.index[val_end + SEQ_LEN : val_end + SEQ_LEN + len(pred_test)]

        plot_node_prediction(node, time_index_test, true_test[:, 0], pred_test[:, 0])
        plot_random_hours(node, time_index_test, true_test[:, 0], pred_test[:, 0])
        plot_random_segment(node, model, scaler_level, level_test, rain_test_scaled, val_end, df.index)

    results_df = pd.DataFrame(results, columns=["Node", "RMSE", "MAE", "NSE"])
    print("\nValidation Results")
    print(results_df)
    results_df.to_excel(output_dir / "validation_results.xlsx", index=False)


if __name__ == "__main__":
    main()

# %%
