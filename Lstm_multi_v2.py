# %%
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class Config:
    # Train one model per node.
    nodes: tuple[str, ...] = ("275", "363", "208", "207", "72")
    # Try a longer history window alongside the original 3-hour window.
    history_windows: tuple[int, ...] = (36, 72)
    future_rain: int = 36
    predict_steps: int = 36
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 40
    batch_size: int = 128
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    patience: int = 6
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    peak_weight: float = 3.0
    peak_percentile: float = 0.90
    output_dir: str = "models_multi_v2_optimized"


CONFIG = Config()


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
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class PeakWeightedMSELoss(nn.Module):
    def __init__(self, peak_threshold: float, peak_weight: float) -> None:
        super().__init__()
        self.peak_threshold = peak_threshold
        self.peak_weight = peak_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(target)
        weights = torch.where(target >= self.peak_threshold, self.peak_weight, weights)
        return torch.mean(weights * (pred - target) ** 2)


def find_excel_file(base_dir: Path, keywords: list[str]) -> Path:
    candidates = sorted(
        path for path in base_dir.glob("*.xlsx") if not path.name.startswith("~$")
    )
    for path in candidates:
        if all(keyword in path.name for keyword in keywords):
            return path
    raise FileNotFoundError(f"Could not find Excel file with keywords: {keywords}")


def normalize_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = {str(column).strip().lower(): column for column in df.columns}

    if "datetime" in renamed_columns:
        df = df.rename(columns={renamed_columns["datetime"]: "datetime"})
    elif "date" in renamed_columns:
        df = df.rename(columns={renamed_columns["date"]: "datetime"})
    elif all(part in renamed_columns for part in ["年", "月", "日", "小时"]):
        time_parts = df[
            [
                renamed_columns["年"],
                renamed_columns["月"],
                renamed_columns["日"],
                renamed_columns["小时"],
            ]
        ].rename(columns={renamed_columns["小时"]: "hour"})
        df["datetime"] = pd.to_datetime(time_parts)
    else:
        raise KeyError(
            f"Could not identify a datetime column. Available columns: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def detect_rain_column(df: pd.DataFrame, node_columns: set[str]) -> str:
    for column in df.columns:
        column_str = str(column)
        if column_str in node_columns:
            continue
        column_lower = column_str.lower()
        if "rain" in column_lower or "mm" in column_lower:
            return column_str
        if "降水" in column_str or "降雨" in column_str:
            return column_str

    for column in df.columns:
        column_str = str(column)
        if column_str not in node_columns:
            return column_str

    raise KeyError("Could not identify the rain column.")


def load_merged_data(base_dir: Path, config: Config) -> tuple[pd.DataFrame, str]:
    level_file = find_excel_file(base_dir, ["节点液位"])
    rain_file = find_excel_file(base_dir, ["5分钟", "降雨"])

    df_level = pd.read_excel(level_file)
    df_rain = pd.read_excel(rain_file)

    df_level = normalize_datetime_column(df_level)
    df_rain = normalize_datetime_column(df_rain)

    df_level.columns = [str(column) for column in df_level.columns]
    df_rain.columns = [str(column) for column in df_rain.columns]

    df = pd.merge(df_level, df_rain, on="datetime", how="inner")
    df = df.sort_values("datetime").set_index("datetime")

    missing_nodes = [node for node in config.nodes if node not in df.columns]
    if missing_nodes:
        raise KeyError(f"Missing node columns: {missing_nodes}")

    rain_column = detect_rain_column(df, node_columns=set(config.nodes))
    return df, rain_column


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values.flatten())
    rolled = series.rolling(window=window, min_periods=1).sum()
    return rolled.to_numpy(dtype=np.float32).reshape(-1, 1)


def first_difference(values: np.ndarray) -> np.ndarray:
    diff = np.diff(values.flatten(), prepend=values.flatten()[0])
    return diff.astype(np.float32).reshape(-1, 1)


def build_feature_frame(level: np.ndarray, rain: np.ndarray) -> np.ndarray:
    # Features are derived in code, so the Excel files can stay unchanged.
    level_diff = first_difference(level)
    rain_acc_1h = rolling_sum(rain, 12)
    rain_acc_3h = rolling_sum(rain, 36)
    return np.concatenate(
        [
            level.astype(np.float32),
            rain.astype(np.float32),
            level_diff,
            rain_acc_1h,
            rain_acc_3h,
        ],
        axis=1,
    )


def split_series(values: np.ndarray, train_end: int, val_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return values[:train_end], values[train_end:val_end], values[val_end:]


def create_dataset(
    features: np.ndarray,
    future_rain: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    predict_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    total = len(target) - seq_len - predict_steps + 1
    for i in range(max(total, 0)):
        history_block = features[i : i + seq_len]
        rain_future = future_rain[i + seq_len : i + seq_len + predict_steps].flatten()
        y = target[i + seq_len : i + seq_len + predict_steps].flatten()

        if len(rain_future) != predict_steps:
            continue

        # Keep the full history block, and place future rain on the tail part only.
        x = np.zeros((seq_len, history_block.shape[1] + 1), dtype=np.float32)
        x[:, :-1] = history_block.astype(np.float32)
        x[-predict_steps:, -1] = rain_future.astype(np.float32)
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def inverse_transform_2d(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def build_model(input_size: int, config: Config) -> LSTMModel:
    return LSTMModel(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.predict_steps,
        dropout=config.dropout,
    )


def predict(model: LSTMModel, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32)
        return model(x_t).cpu().numpy()


def evaluate_predictions(true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(true_values.flatten(), pred_values.flatten()))),
        "MAE": float(mean_absolute_error(true_values.flatten(), pred_values.flatten())),
        "NSE": float(nse(true_values.flatten(), pred_values.flatten())),
    }


def train_model(
    model: LSTMModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
) -> tuple[LSTMModel, float, int]:
    peak_threshold = float(np.quantile(y_train, config.peak_percentile))
    criterion = PeakWeightedMSELoss(
        peak_threshold=peak_threshold,
        peak_weight=config.peak_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    num_batches = int(np.ceil(len(x_train_t) / config.batch_size))
    if num_batches == 0:
        raise ValueError("Training set is empty. Please check data size and sequence settings.")

    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for i in range(num_batches):
            start = i * config.batch_size
            end = (i + 1) * config.batch_size
            x_batch = x_train_t[start:end]
            y_batch = y_train_t[start:end]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t).cpu().numpy()

        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        scheduler.step(val_rmse)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"TrainLoss: {epoch_loss / num_batches:.6f} "
            f"ValRMSE(scaled): {val_rmse:.6f} "
            f"LR: {current_lr:.6f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}, best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val_rmse, best_epoch


def save_t1_plot(
    node: str,
    time_index: pd.Index,
    true_series: np.ndarray,
    pred_series: np.ndarray,
    output_dir: Path,
    history_window: int,
) -> None:
    plt.figure(figsize=(15, 5), dpi=300)
    plt.plot(time_index, true_series, color="black", linewidth=2.2, label="Real")
    plt.plot(time_index, pred_series, color="#d62728", linewidth=1.8, label="Pred")
    plt.title(f"Node {node} Water Level Prediction (t+1, history={history_window})")
    plt.xlabel("Datetime")
    plt.ylabel("Water Level")
    plt.legend(fontsize=10)
    plt.xticks(rotation=30)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"node_{node}_t1_comparison.png")
    plt.close()


def save_random_hour_plot(
    node: str,
    time_index: pd.Index,
    true_series: np.ndarray,
    pred_series: np.ndarray,
    output_dir: Path,
    history_window: int,
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
        plt.title(f"Node {node} - Random Hour {j + 1} (history={history_window})")
        plt.ylabel("Water Level")
        plt.grid(alpha=0.2)
        if j == 0:
            plt.legend(fontsize=9)

    plt.xlabel("Datetime")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / f"node_{node}_random_hours.png")
    plt.close()


def save_random_segment_plot(
    node: str,
    time_index: pd.Index,
    true_segment: np.ndarray,
    pred_segment: np.ndarray,
    output_dir: Path,
    history_window: int,
) -> None:
    plt.figure(figsize=(14, 5), dpi=300)
    plt.plot(time_index, true_segment, color="black", linewidth=2.2, label="Real")
    plt.plot(time_index, pred_segment, color="#d62728", linewidth=1.8, label="Pred")

    y_min = min(true_segment.min(), pred_segment.min())
    y_max = max(true_segment.max(), pred_segment.max())
    y_range = y_max - y_min
    margin = y_range * 0.05 if y_range > 0 else 0.1
    plt.ylim(y_min - margin, y_max + margin)

    plt.title(f"Node {node} - Random 3 Hour Segment (history={history_window})")
    plt.xlabel("Datetime")
    plt.ylabel("Water Level")
    plt.legend(fontsize=10)
    plt.xticks(rotation=30)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"node_{node}_random_segment.png")
    plt.close()


def prepare_scaled_splits(
    feature_frame: np.ndarray,
    rain: np.ndarray,
    level: np.ndarray,
    train_end: int,
    val_end: int,
) -> dict[str, np.ndarray | MinMaxScaler]:
    feature_train, feature_val, feature_test = split_series(feature_frame, train_end, val_end)
    rain_train, rain_val, rain_test = split_series(rain, train_end, val_end)
    level_train, level_val, level_test = split_series(level, train_end, val_end)

    feature_scaler = MinMaxScaler()
    rain_scaler = MinMaxScaler()
    level_scaler = MinMaxScaler()

    feature_train_scaled = feature_scaler.fit_transform(feature_train)
    feature_val_scaled = feature_scaler.transform(feature_val)
    feature_test_scaled = feature_scaler.transform(feature_test)

    rain_train_scaled = rain_scaler.fit_transform(rain_train)
    rain_val_scaled = rain_scaler.transform(rain_val)
    rain_test_scaled = rain_scaler.transform(rain_test)

    level_train_scaled = level_scaler.fit_transform(level_train)
    level_val_scaled = level_scaler.transform(level_val)
    level_test_scaled = level_scaler.transform(level_test)

    return {
        "feature_scaler": feature_scaler,
        "rain_scaler": rain_scaler,
        "level_scaler": level_scaler,
        "feature_train_scaled": feature_train_scaled,
        "feature_val_scaled": feature_val_scaled,
        "feature_test_scaled": feature_test_scaled,
        "rain_train_scaled": rain_train_scaled,
        "rain_val_scaled": rain_val_scaled,
        "rain_test_scaled": rain_test_scaled,
        "level_train_scaled": level_train_scaled,
        "level_val_scaled": level_val_scaled,
        "level_test_scaled": level_test_scaled,
        "level_test_raw": level_test,
    }


def choose_best_history_window(
    node: str,
    df: pd.DataFrame,
    rain_column: str,
    config: Config,
) -> dict[str, object]:
    n = len(df)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    level = df[node].to_numpy(dtype=np.float32).reshape(-1, 1)
    rain = df[rain_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    feature_frame = build_feature_frame(level, rain)
    scaled = prepare_scaled_splits(feature_frame, rain, level, train_end, val_end)

    best_run: dict[str, object] | None = None
    for history_window in config.history_windows:
        x_train, y_train = create_dataset(
            scaled["feature_train_scaled"],  # type: ignore[arg-type]
            scaled["rain_train_scaled"],  # type: ignore[arg-type]
            scaled["level_train_scaled"],  # type: ignore[arg-type]
            seq_len=history_window,
            predict_steps=config.predict_steps,
        )
        x_val, y_val = create_dataset(
            scaled["feature_val_scaled"],  # type: ignore[arg-type]
            scaled["rain_val_scaled"],  # type: ignore[arg-type]
            scaled["level_val_scaled"],  # type: ignore[arg-type]
            seq_len=history_window,
            predict_steps=config.predict_steps,
        )
        x_test, y_test = create_dataset(
            scaled["feature_test_scaled"],  # type: ignore[arg-type]
            scaled["rain_test_scaled"],  # type: ignore[arg-type]
            scaled["level_test_scaled"],  # type: ignore[arg-type]
            seq_len=history_window,
            predict_steps=config.predict_steps,
        )

        if len(x_train) == 0:
            raise ValueError(f"Training set is empty for node {node}, history {history_window}.")
        if len(x_val) == 0 or len(x_test) == 0:
            raise ValueError(f"Validation or test set is empty for node {node}, history {history_window}.")

        print(
            f"Node {node}, history {history_window} -> "
            f"train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}"
        )

        model = build_model(input_size=x_train.shape[-1], config=config)
        model, best_val_rmse_scaled, best_epoch = train_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
        )

        pred_val = predict(model, x_val)
        pred_test = predict(model, x_test)

        level_scaler = scaled["level_scaler"]  # type: ignore[assignment]
        true_val = inverse_transform_2d(level_scaler, y_val)
        pred_val = inverse_transform_2d(level_scaler, pred_val)
        true_test = inverse_transform_2d(level_scaler, y_test)
        pred_test = inverse_transform_2d(level_scaler, pred_test)

        val_metrics = evaluate_predictions(true_val, pred_val)
        test_metrics = evaluate_predictions(true_test, pred_test)
        print(
            f"Node {node}, history {history_window} -> "
            f"Val RMSE: {val_metrics['RMSE']:.4f}, "
            f"Test RMSE: {test_metrics['RMSE']:.4f}, "
            f"best epoch: {best_epoch}"
        )

        run_info = {
            "history_window": history_window,
            "model": model,
            "best_epoch": best_epoch,
            "best_val_rmse_scaled": best_val_rmse_scaled,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "pred_test": pred_test,
            "true_test": true_test,
            "level_scaler": level_scaler,
            "feature_scaler": scaled["feature_scaler"],
            "rain_scaler": scaled["rain_scaler"],
            "val_end": val_end,
        }
        if best_run is None or val_metrics["RMSE"] < best_run["val_metrics"]["RMSE"]:  # type: ignore[index]
            best_run = run_info

    if best_run is None:
        raise RuntimeError(f"No valid training run for node {node}.")
    return best_run


def train_single_node(
    node: str,
    df: pd.DataFrame,
    rain_column: str,
    config: Config,
    output_dir: Path,
) -> dict[str, float | int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_run = choose_best_history_window(node, df, rain_column, config)

    model = best_run["model"]
    history_window = int(best_run["history_window"])
    best_epoch = int(best_run["best_epoch"])
    val_metrics = best_run["val_metrics"]
    test_metrics = best_run["test_metrics"]
    pred_test = best_run["pred_test"]
    true_test = best_run["true_test"]

    torch.save(model.state_dict(), output_dir / f"model_node_{node}.pth")
    joblib.dump(best_run["level_scaler"], output_dir / f"scaler_level_{node}.pkl")
    joblib.dump(best_run["rain_scaler"], output_dir / f"scaler_rain_{node}.pkl")
    joblib.dump(best_run["feature_scaler"], output_dir / f"scaler_feature_{node}.pkl")

    val_end = int(best_run["val_end"])
    time_index_test = df.index[val_end + history_window : val_end + history_window + len(pred_test)]
    true_t1 = true_test[:, 0]
    pred_t1 = pred_test[:, 0]

    if len(time_index_test) != len(true_t1):
        raise ValueError(
            f"Time index length mismatch for node {node}: "
            f"{len(time_index_test)} vs {len(true_t1)}"
        )

    save_t1_plot(node, time_index_test, true_t1, pred_t1, output_dir, history_window)
    save_random_hour_plot(node, time_index_test, true_t1, pred_t1, output_dir, history_window)

    random_start = np.random.randint(0, len(pred_test))
    segment_time = time_index_test[random_start : random_start + config.predict_steps]
    true_segment = true_test[random_start]
    pred_segment = pred_test[random_start]
    if len(segment_time) == config.predict_steps:
        save_random_segment_plot(node, segment_time, true_segment, pred_segment, output_dir, history_window)

    return {
        "Node": node,
        "Best_History_Window": history_window,
        "Best_Epoch": best_epoch,
        "Val_RMSE_3h": float(val_metrics["RMSE"]),
        "Val_MAE_3h": float(val_metrics["MAE"]),
        "Val_NSE_3h": float(val_metrics["NSE"]),
        "Test_RMSE_3h": float(test_metrics["RMSE"]),
        "Test_MAE_3h": float(test_metrics["MAE"]),
        "Test_NSE_3h": float(test_metrics["NSE"]),
    }


def main() -> None:
    set_seed(CONFIG.seed)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / CONFIG.output_dir
    output_dir.mkdir(exist_ok=True)

    df, rain_column = load_merged_data(base_dir, CONFIG)
    print(f"Merged data shape: {df.shape}")
    print(f"Rain column: {rain_column}")

    results: list[dict[str, float | int | str]] = []
    for node in CONFIG.nodes:
        print(f"\n======================\nTraining Node: {node}\n======================")
        metrics_row = train_single_node(node, df, rain_column, CONFIG, output_dir)
        print(
            f"Node {node} best history: {metrics_row['Best_History_Window']}, "
            f"best epoch: {metrics_row['Best_Epoch']}, "
            f"Val RMSE_3h: {metrics_row['Val_RMSE_3h']:.4f}, "
            f"Val MAE_3h: {metrics_row['Val_MAE_3h']:.4f}, "
            f"Val NSE_3h: {metrics_row['Val_NSE_3h']:.4f}"
        )
        results.append(metrics_row)

    results_df = pd.DataFrame(results)
    results_df.to_excel(output_dir / "validation_results.xlsx", index=False)
    print("\nValidation Results")
    print(results_df)


if __name__ == "__main__":
    main()

# %%
