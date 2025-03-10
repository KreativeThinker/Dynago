import pandas as pd
import numpy as np
import os

RAW_CSV = "dynago/data/raw.csv"
NORM_CSV = "dynago/data/normalized.csv"


def load_data():
    """Loads raw and existing normalized data, handling empty files."""
    raw_df = pd.read_csv(RAW_CSV)

    if os.path.exists(NORM_CSV) and os.path.getsize(NORM_CSV) > 0:
        norm_df = pd.read_csv(NORM_CSV)
    else:
        norm_df = pd.DataFrame(columns=raw_df.columns)  # Empty DataFrame

    return raw_df, norm_df


def normalize_landmarks(new_data):
    """Normalizes landmark data (centering + scaling)."""
    landmarks = new_data.iloc[:, :-1].values
    labels = new_data.iloc[:, -1].values

    num_landmarks = 21
    landmarks = landmarks.reshape(-1, num_landmarks, 3)

    normalized_landmarks = []
    for hand in landmarks:
        wrist = hand[0]
        hand -= wrist  # Center at wrist

        max_dist = np.max(np.linalg.norm(hand, axis=1))  # Max distance for scaling
        hand /= max_dist if max_dist > 0 else 1  # Scale

        normalized_landmarks.append(hand.flatten())

    norm_df = pd.DataFrame(normalized_landmarks)
    norm_df["gesture"] = labels
    return norm_df


def normalize():
    """Processes only new raw data and appends to normalized CSV."""
    raw_df, norm_df = load_data()

    last_processed_index = len(norm_df)
    new_data = raw_df.iloc[last_processed_index:]  # Only new entries

    if new_data.empty:
        print("No new data to process.")
        return

    norm_new_data = normalize_landmarks(new_data)
    norm_new_data.to_csv(
        NORM_CSV, mode="a", header=not os.path.exists(NORM_CSV), index=False
    )
    print(f"Processed {len(norm_new_data)} new entries.")


if __name__ == "__main__":
    normalize()
