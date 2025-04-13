# train_and_evaluate.py

from data_preprocessing import load_cifar100_data
from mobilenetv2_baseline import build_mobilenetv2_baseline
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os, json

train_gen, val_gen = load_cifar100_data()

model = build_mobilenetv2_baseline()

if not os.path.exists("models"):
    os.makedirs("models")

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("models/mobilenetv2_baseline.keras", save_best_only=True)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=callbacks)

# Save metrics
if not os.path.exists("results"):
    os.makedirs("results")

with open("results/baseline_metrics.json", "w") as f:
    json.dump(history.history, f)