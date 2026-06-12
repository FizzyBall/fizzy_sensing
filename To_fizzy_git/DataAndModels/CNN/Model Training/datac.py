"""
data.py — Load gesture CSV recordings and prepare them for CNN training.

What this script does, in plain English:
1. Finds every CSV file in the 'recordings' folder.
2. Reads each one into a table.
3. Keeps only the columns that have real data.
4. Cuts each recording into fixed-size "windows" (short clips).
5. Labels each window based on the filename (Idle, Shake, Tap, UpDown).
6. Packages everything into a PyTorch Dataset that the CNN can train on.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from Trainsettings import version, LABELS, FEATURE_COLUMNS



# Folder where your CSV files live. All the recording folders sit inside 'recordings_folder'.
# Each recording folder should be called "recordings_v{version}", where {version} is a number
# which corresponds to what version of data you wanna train.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, 'all_recordings', f'recordings_v{version}')

# Folder where the normalization stats (mean / std) get saved.
# This script lives in 'Model Training', but the 'Mean, std' folder sits one
# level up in the CNN_final root, so we step out of this folder with '..'.
MEANSTD_DIR = os.path.join(SCRIPT_DIR, '..', 'Mean, std')

# Define what classes are needed to classify, and what number each class gets. 
# There are already pre defined class names which were used to train the previous models.

# How long each window is, in samples.
WINDOW_SIZE = 64

# Stride for each window
WINDOW_STRIDE = 32


#Function to read and load one CSV file as a NumPy array of shape (number of rows, number of features)
def load_one_csv(file_path):

    #Panda reads the CSV
    df = pd.read_csv(file_path)

    # Pick the collums that are wanted and defined above in the code
    df = df[FEATURE_COLUMNS]

    # Convert to a NumPy array of 32-bit floats, this is what pytorch needs
    return df.to_numpy(dtype=np.float32)



#Function to cut one long recording into many fixed-size windows (short clips)
def cut_into_windows(recording, window_size, stride):
    """
    Take one full recording (shape: time × features) and cut it
    into many overlapping windows (shape: window_size × features).

    Example: a 1000-row recording with window_size=200, stride=100
    gives us 9 windows.
    """

    # Empty list to collect the windows, and a pointer that starts at the top
    windows = []
    start = 0

    # Keep sliding the window down as long as a full window still fits
    while start + window_size <= len(recording):

        # Take the slice of rows from 'start' up to 'start + window_size'
        window = recording[start : start + window_size]

        # Add this window to our list
        windows.append(window)

        # Move the start down by 'stride' rows (overlap if stride < window_size)
        start += stride

    # If the recording was too short to fit even one window, return an empty array
    if len(windows) == 0:
        return np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float32)

    # Stack all the separate windows into one big array of shape (num_windows, window_size, features)
    return np.stack(windows)


# =============================================================
# STEP 4: The Dataset class — what PyTorch will use for training
# =============================================================


class GestureDataset(Dataset):
    """
    A PyTorch Dataset that holds all our gesture windows + their labels.

    Once built, you can ask it for example #5 with: dataset[5]
    and you'll get back (window_tensor, label).
    """

    def __init__(self):

        # Two empty lists where we collect all windows and all labels, then combine them at the end
        all_windows = []
        all_labels = []

        # Get a list of the paths to every CSV file inside the recordings folder
        all_csv_paths = glob.glob(os.path.join(RECORDINGS_DIR, '*.csv'))

        # Go through each gesture type one by one (name + its number, e.g. 'shake' -> 1)
        for gesture_name, label_number in LABELS.items():

            # Find every CSV whose filename contains this gesture name
            # e.g. for 'idle' this matches 'idle_001.csv', 'idle_002.csv', ...

            # pattern = os.path.join(RECORDINGS_DIR, f'*{gesture_name}*.csv')
            # file_paths = sorted(glob.glob(pattern))

            file_paths = sorted([
                p for p in all_csv_paths
                if gesture_name.lower() in os.path.basename(p).lower()
            ])

            # Print how many files we found for this gesture (handy sanity check)
            print(f'Found {len(file_paths)} files for "{gesture_name}"')

            # Load every CSV for this gesture into its own NumPy array
            recordings = [load_one_csv(p) for p in file_paths]

            # Glue all those recordings together into one long signal (stacked top to bottom)
            combined = np.concatenate(recordings, axis=0)

            # Slide windows over that long signal so no data is thrown away
            windows = cut_into_windows(combined, WINDOW_SIZE, WINDOW_STRIDE)

            # Make a label for every window: just this gesture's number repeated
            labels = np.full(len(windows), label_number, dtype=np.int64)

            # Save this gesture's windows and labels into the big lists
            all_windows.append(windows)
            all_labels.append(labels)

        # Stack every gesture's windows and labels into two big arrays
        # X shape: (total_windows, window_size, num_features) e.g. (N, 64, 6)
        # y shape: (total_windows,)
        X = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # PyTorch's Conv1d expects shape (batch, channels, time)
        # Right now we have (batch, time, channels), so we swap the last two axes.
        X = X.transpose(0, 2, 1)   # now shape: (N, 6, 64)

        # Normalize: subtract mean, divide by std deviation, per channel.
        # This makes training much more stable — all features end up
        # roughly in the same range (mean 0, std 1).
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8   # +tiny number to avoid /0
        X = (X - mean) / std

        # Save the mean and std to disk so the live/real-time code can normalize the same way.
        # They go in the 'Mean, std' folder, named by the data version they belong to.
        # If they already exist for this data version, leave them alone (don't overwrite).
        os.makedirs(MEANSTD_DIR, exist_ok=True)
        mean_path = os.path.join(MEANSTD_DIR, f'imu_mean_v{version}.npy')
        std_path = os.path.join(MEANSTD_DIR, f'imu_std_v{version}.npy')
        if os.path.exists(mean_path) and os.path.exists(std_path):
            print(f'Mean/std for data v{version} already exist, keeping {mean_path} and {std_path}')
        else:
            np.save(mean_path, mean)
            np.save(std_path, std)
            print(f'Saved {mean_path} and {std_path}')

        print(f'mean = {mean}')
        print(f'std = {std}')

        # Convert the final NumPy arrays into PyTorch tensors and store them on the object
        self.X = torch.from_numpy(X).float()   # inputs as floats
        self.y = torch.from_numpy(y).long()    # labels as integers

    def __len__(self):
        # Tells PyTorch how many examples (windows) we have in total
        return len(self.y)

    def __getitem__(self, index):
        # Tells PyTorch how to fetch example number `index`: returns (window, label)
        return self.X[index], self.y[index]




#Here the file is run to test it

if __name__ == '__main__':
    # Build the dataset
    dataset = GestureDataset()

    print()
    print(f'Total windows: {len(dataset)}')
    print(f'Input shape:   {dataset.X.shape}   (windows, channels, timesteps)')
    print(f'Label shape:   {dataset.y.shape}')
    print(f'Examples per class: {torch.bincount(dataset.y).tolist()}')
    print(f'   (in order: Idle, Shake, Tap, Spin)')

    # Wrap in a DataLoader — this is what you'll loop over during training.
    # batch_size=32 means it gives you 32 windows at a time.
    # shuffle=True means it mixes them up each epoch (important for training).
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Grab one batch just to confirm it works
    batch_X, batch_y = next(iter(loader))
    print()
    print(f'One batch of inputs: {batch_X.shape}')
    print(f'One batch of labels: {batch_y.shape}')