# %%
import os
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


NODE = "275"

LEVEL_STEPS = [12, 24, 36]
RAIN_STEPS = [12, 24, 36]

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
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            NUM_LAYERS,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, PREDICT_STEPS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def create_dataset(
    level: np.ndarray,
    rain: np.ndarray,
    level_steps: int,
    rain_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    total = len(level) - SEQ_LEN - PREDICT_STEPS + 1
    for i in range(max(total, 0)):
        x = np.zeros((SEQ_LEN, 3), dtype=np.float32)

        level_hist = level[i + SEQ_LEN - level_steps : i + SEQ_LEN]
        rain_hist = rain[i + SEQ_LEN - rain_steps : i + SEQ_LEN]
        rain_future = rain[i + SEQ_LEN : i + SEQ_LEN + FUTURE_RAIN]
        y = level[i + SEQ_LEN : i + SEQ_LEN + PREDICT_STEPS]

        x[-level_steps:, 0] = level_hist.flatten()
        x[-rain_steps:, 1] = rain_hist.flatten()
        x[:, 2] = rain_future.flatten()

        xs.append(x)
        ys.append(y.flatten().astype(np.float32))

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


def detect_rain_column(df: pd.DataFrame, excluded_columns: set[str]) -> str:
    for column in df.columns:
        column_str = str(column)
        if column_str in excluded_columns:
            continue
        if any(token in column_str.lower() for token in ["rain", "mm"]):
            return column
        if any(token in column_str for token in ["雨", "降水", "降雨"]):
            return column

    for column in df.columns:
        column_str = str(column)
        if column_str not in excluded_columns:
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

    rain_column = detect_rain_column(df, excluded_columns={NODE})
    if NODE not in df.columns:
        raise KeyError(f"Node column {NODE!r} was not found. Available columns: {list(df.columns)}")

    return df, rain_column


def inverse_transform_2d(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> LSTMModel:
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


def evaluate_multi_horizon(true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, float]:
    metrics = {}
    for horizon, steps in [("1h", 12), ("2h", 24), ("3h", 36)]:
        true_slice = true_values[:, :steps].flatten()
        pred_slice = pred_values[:, :steps].flatten()
        metrics[f"RMSE_{horizon}"] = np.sqrt(mean_squared_error(true_slice, pred_slice))
        metrics[f"MAE_{horizon}"] = mean_absolute_error(true_slice, pred_slice)
        metrics[f"NSE_{horizon}"] = nse(true_slice, pred_slice)
    return metrics


def plot_metric_bars(results_df: pd.DataFrame) -> None:
    for metric in ["RMSE_1h", "RMSE_2h", "RMSE_3h"]:
        plt.figure(figsize=(10, 5))
        plt.bar(results_df["Model"], results_df[metric])
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_first_step_comparison(
    time_index: pd.Index,
    true_series: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    best_model: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]

    plt.figure(figsize=(15, 5), dpi=300)
    plt.plot(time_index, true_series, color="black", linewidth=2.5, label="Real")

    for i, (key, pred) in enumerate(predictions_dict.items()):
        plt.plot(
            time_index,
            pred,
            label=key,
            linewidth=2 if key == best_model else 1.2,
            alpha=1.0 if key == best_model else 0.6,
            color="#d62728" if key == best_model else colors[i % len(colors)],
        )

    plt.title(f"Node {NODE} Water Level Prediction Comparison (t+1)")
    plt.xlabel("Datetime")
    plt.ylabel("Water Level")
    plt.legend(ncol=3, fontsize=9, frameon=True)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_random_hours(
    time_index: pd.Index,
    true_series: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    num_hours: int = 3,
) -> None:
    steps_per_hour = 12
    max_start = len(true_series) - steps_per_hour + 1
    if max_start <= 0:
        return

    num_samples = min(num_hours, max_start)
    random_starts = np.random.choice(range(max_start), num_samples, replace=False)

    plt.figure(figsize=(15, 8), dpi=300)
    for i, start in enumerate(sorted(random_starts)):
        end = start + steps_per_hour
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(time_index[start:end], true_series[start:end], color="black", linewidth=2.5, label="Real")

        for key, pred in predictions_dict.items():
            plt.plot(time_index[start:end], pred[start:end], linewidth=1.3, alpha=0.7, label=key if i == 0 else None)

        plt.title(f"Random Test Hour {i + 1}")
        plt.ylabel("Water Level")
        plt.grid(alpha=0.2)
        if i == 0:
            plt.legend(fontsize=8, ncol=4)

    plt.xlabel("Datetime")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def main() -> None:
    set_seed(SEED)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "models"
    output_dir.mkdir(exist_ok=True)

    df, rain_column = load_merged_data(base_dir)

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    level = df[NODE].to_numpy(dtype=np.float32).reshape(-1, 1)
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

    joblib.dump(scaler_level, output_dir / "scaler_level.pkl")
    joblib.dump(scaler_rain, output_dir / "scaler_rain.pkl")

    results = []
    predictions_dict: dict[str, np.ndarray] = {}
    best_model_name = None
    best_rmse = float("inf")
    time_index = None
    true_series = None

    for level_step in LEVEL_STEPS:
        for rain_step in RAIN_STEPS:
            model_name = f"L{level_step}_R{rain_step}"
            print(f"\n======================")
            print(f"Training {model_name}")
            print("======================")

            x_train, y_train = create_dataset(level_train_scaled, rain_train_scaled, level_step, rain_step)
            x_val, y_val = create_dataset(level_val_scaled, rain_val_scaled, level_step, rain_step)
            x_test, y_test = create_dataset(level_test_scaled, rain_test_scaled, level_step, rain_step)

            if len(x_val) == 0 or len(x_test) == 0:
                raise ValueError("Validation or test set is empty. Please reduce SEQ_LEN/PREDICT_STEPS or use more data.")

            model = train_model(x_train, y_train)

            model.eval()
            with torch.no_grad():
                pred_val = model(torch.tensor(x_val, dtype=torch.float32)).numpy()
                pred_test = model(torch.tensor(x_test, dtype=torch.float32)).numpy()

            pred_val = inverse_transform_2d(scaler_level, pred_val)
            true_val = inverse_transform_2d(scaler_level, y_val)

            metrics = evaluate_multi_horizon(true_val, pred_val)
            results.append(
                [
                    model_name,
                    metrics["RMSE_1h"],
                    metrics["RMSE_2h"],
                    metrics["RMSE_3h"],
                    metrics["MAE_1h"],
                    metrics["MAE_2h"],
                    metrics["MAE_3h"],
                    metrics["NSE_1h"],
                    metrics["NSE_2h"],
                    metrics["NSE_3h"],
                ]
            )

            if metrics["RMSE_3h"] < best_rmse:
                best_rmse = metrics["RMSE_3h"]
                best_model_name = model_name

            torch.save(model.state_dict(), output_dir / f"model_{model_name}.pth")

            pred_test = inverse_transform_2d(scaler_level, pred_test)
            true_test = inverse_transform_2d(scaler_level, y_test)

            predictions_dict[model_name] = pred_test[:, 0]
            time_index = df.index[val_end + SEQ_LEN : val_end + SEQ_LEN + len(pred_test)]
            true_series = true_test[:, 0]

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "RMSE_1h",
            "RMSE_2h",
            "RMSE_3h",
            "MAE_1h",
            "MAE_2h",
            "MAE_3h",
            "NSE_1h",
            "NSE_2h",
            "NSE_3h",
        ],
    )
    print("\nValidation Results")
    print(results_df)
    results_df.to_excel(output_dir / "evaluation_results.xlsx", index=False)

    plot_metric_bars(results_df)

    if time_index is not None and true_series is not None and best_model_name is not None:
        plot_first_step_comparison(time_index, true_series, predictions_dict, best_model_name)
        plot_random_hours(time_index, true_series, predictions_dict)

        print(f"\nBest model by RMSE_3h: {best_model_name} ({best_rmse:.4f})")


if __name__ == "__main__":
    main()

# %%
