import yaml 
import numpy as np
import os
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

val_size = params['preprocess']['val_size']
seed = params['preprocess']['seed']

x_train = np.load("data/raw/x_train.npy")
x_test = np.load("data/raw/x_test.npy")
y_train = np.load("data/raw/y_train.npy")
y_test = np.load("data/raw/y_test.npy")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size = val_size,
    random_state = seed,
    stratify = y_train
)

os.makedirs("data/processed", exist_ok= True)

np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/y_train.npy", y_train)

