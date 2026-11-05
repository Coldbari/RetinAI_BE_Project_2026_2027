import os
import random
import shutil
from collections import defaultdict

# =========================
# CONFIG
# =========================

SOURCE_FOLDER = "rop_dataset_working_copy"
OUTPUT_FOLDER = "dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# =========================
# CREATE OUTPUT FOLDERS
# =========================

splits = ["train", "val", "test"]
classes = ["DG0", "DG1"]

for split in splits:
    for cls in classes:
        os.makedirs(
            os.path.join(OUTPUT_FOLDER, split, cls),
            exist_ok=True
        )

# =========================
# GROUP FILES BY PATIENT ID
# =========================

patient_files = defaultdict(list)

for root, _, filenames in os.walk(SOURCE_FOLDER):
    for filename in filenames:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        parts = filename.split("_")

        # Patient ID is first part
        patient_id = parts[0]

        patient_files[patient_id].append((root, filename))

# =========================
# SPLIT PATIENTS
# =========================

patient_ids = list(patient_files.keys())

random.seed(RANDOM_SEED)
random.shuffle(patient_ids)

total_patients = len(patient_ids)

train_end = int(total_patients * TRAIN_RATIO)
val_end = train_end + int(total_patients * VAL_RATIO)

train_patients = patient_ids[:train_end]
val_patients = patient_ids[train_end:val_end]
test_patients = patient_ids[val_end:]

# =========================
# COPY FILES
# =========================

def copy_patient_files(patient_list, split_name):

    count = 0

    for patient_id in patient_list:

        for root, filename in patient_files[patient_id]:

            parts = filename.split("_")

            # Find DG class
            dg_part = [p for p in parts if p.startswith("DG")][0]

            class_name = dg_part

            src_path = os.path.join(root, filename)

            dst_path = os.path.join(
                OUTPUT_FOLDER,
                split_name,
                class_name,
                filename
            )

            shutil.copy2(src_path, dst_path)

            count += 1

    print(f"{split_name.upper()} → {count} images")


copy_patient_files(train_patients, "train")
copy_patient_files(val_patients, "val")
copy_patient_files(test_patients, "test")

# =========================
# FINAL SUMMARY
# =========================

print("\nDataset split completed successfully.")

print(f"\nTotal Patients: {total_patients}")
print(f"Train Patients: {len(train_patients)}")
print(f"Validation Patients: {len(val_patients)}")
print(f"Test Patients: {len(test_patients)}")
