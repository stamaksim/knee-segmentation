import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GROUP_NUMBER = "4"
EXPECTED_LABELED = 100
EXPECTED_TEST = 50

xrays_dir, masks_dir, csvs_dir = "data/xrays/", "data/masks/", "data/CSVs/"
os.makedirs(csvs_dir, exist_ok=True)

all_images = sorted(
    (f for f in os.listdir(xrays_dir) if f.endswith(".png")),
    key=lambda f: int(os.path.splitext(f)[0]),   # numeric sort
)

labeled_data, test_data = [], []
for img_name in all_images:
    xray_path = os.path.join(xrays_dir, img_name)
    mask_path = os.path.join(masks_dir, img_name)
    if os.path.exists(mask_path):
        labeled_data.append({"xrays": xray_path, "masks": mask_path})
    else:
        test_data.append({"xrays": xray_path, "masks": "None"})

# fail loudly instead of silently producing wrong CSVs
assert len(labeled_data) == EXPECTED_LABELED, f"labeled: {len(labeled_data)}"
assert len(test_data) == EXPECTED_TEST, f"test: {len(test_data)}"

pd.DataFrame(labeled_data + test_data).to_csv(csvs_dir + "dataset.csv", index=False)

np.random.seed(42)
np.random.shuffle(labeled_data)
n_train = int(len(labeled_data) * 0.8)
train_data, val_data = labeled_data[:n_train], labeled_data[n_train:]

pd.DataFrame(train_data).to_csv(csvs_dir + "train.csv", index=False)
pd.DataFrame(val_data).to_csv(csvs_dir + "val.csv", index=False)
pd.DataFrame(test_data).to_csv(csvs_dir + "test.csv", index=False)

counts = {"dataset": len(labeled_data) + len(test_data), "train": len(train_data),
          "val": len(val_data), "test": len(test_data)}
print(counts)

plt.bar(counts.keys(), counts.values())
for i, v in enumerate(counts.values()):
    plt.text(i, v + 1, str(v), ha="center")
plt.ylabel("Number of samples")
plt.title("Samples per CSV file")
plt.savefig("csv_counts.png", dpi=150)

zip_name = f"csv_files_{GROUP_NUMBER}.zip"
with zipfile.ZipFile(zip_name, "w") as z:
    for name in ["dataset", "train", "val", "test"]:
        z.write(csvs_dir + f"{name}.csv", arcname=f"{name}.csv")
print("Done:", zip_name)
