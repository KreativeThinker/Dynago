import os

import numpy as np
import pandas as pd

RAW_CSV = "dynago/data/raw.csv"
NORM_CSV = "dynago/data/normalized.csv"


def load_data():
    """Loads raw data and handles duplicate columns."""
    raw_df = pd.read_csv(RAW_CSV)

    # Remove duplicate gesture_index column (keep only the first occurrence)
    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]

    # Verify we have exactly 63 coordinate columns + 1 label column
    coord_columns = [col for col in raw_df.columns if col not in ["gesture_index"]]
    if len(coord_columns) != 63:
        raise ValueError(f"Expected 63 coordinate columns, found {len(coord_columns)}")

    if os.path.exists(NORM_CSV) and os.path.getsize(NORM_CSV) > 0:
        norm_df = pd.read_csv(NORM_CSV)
    else:
        norm_df = pd.DataFrame(columns=raw_df.columns)

    return raw_df, norm_df


def normalize_landmarks(new_data):
    """Normalizes landmark data with proper column handling."""
    # Extract coordinates (first 63 columns)
    landmarks = new_data.iloc[:, :63].values
    labels = new_data.iloc[:, -1].values  # Last column is gesture_index

    num_landmarks = 21
    landmarks = landmarks.reshape(-1, num_landmarks, 3)

    normalized_landmarks = []
    for hand in landmarks:
        wrist = hand[0]
        hand -= wrist  # Center

        max_dist = np.max(np.linalg.norm(hand, axis=1))
        hand /= max_dist if max_dist > 0 else 1

        normalized_landmarks.append(hand.flatten())

    # Recreate column names for coordinates
    coord_cols = []
    for i in range(1, 22):
        coord_cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    norm_df = pd.DataFrame(normalized_landmarks, columns=coord_cols)
    norm_df["gesture_index"] = labels
    return norm_df


def normalize():
    """Processes new raw data with duplicate column handling."""
    try:
        raw_df, norm_df = load_data()
        last_processed_index = len(norm_df)
        new_data = raw_df.iloc[last_processed_index:]

        if new_data.empty:
            print("No new data to process.")
            return

        norm_new_data = normalize_landmarks(new_data)
        norm_new_data.to_csv(
            NORM_CSV, mode="a", header=not os.path.exists(NORM_CSV), index=False
        )
        print(f"Processed {len(norm_new_data)} new entries.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Debug info:")
        print(f"Columns in raw data: {raw_df.columns.tolist()}")
        print(f"Data shape: {raw_df.shape}")


if __name__ == "__main__":
    normalize()
