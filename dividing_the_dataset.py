import os, shutil, random

FULL_DATA = "./planr_dataset_2/plantvillage dataset/color"

other_plants = [
    f"{FULL_DATA}/Tomato___healthy",
    f"{FULL_DATA}/Tomato___Early_blight",
    f"{FULL_DATA}/Potato___healthy",
    f"{FULL_DATA}/Potato___Late_blight",
    f"{FULL_DATA}/Grape___healthy",
    f"{FULL_DATA}/Grape___Black_rot",
    f"{FULL_DATA}/Peach___healthy",
    f"{FULL_DATA}/Peach___Bacterial_spot",
    f"{FULL_DATA}/Pepper,bell___healthy",
    f"{FULL_DATA}/Pepper,bell___Bacterial_spot",
    f"{FULL_DATA}/Strawberry___healthy",
    f"{FULL_DATA}/Strawberry___Leaf_scorch",
    f"{FULL_DATA}/Soybean___healthy",
    f"{FULL_DATA}/Raspberry___healthy",
]

# Clean and recreate folders
shutil.rmtree("./corn_or_not", ignore_errors=True)
os.makedirs("./corn_or_not/corn", exist_ok=True)
os.makedirs("./corn_or_not/not_corn", exist_ok=True)

# Copy corn images
corn_folders = [f for f in os.listdir(FULL_DATA) if "Corn" in f]
for folder in corn_folders:
    src = f"{FULL_DATA}/{folder}"
    for img in os.listdir(src):
        shutil.copy(f"{src}/{img}", f"./corn_or_not/corn/{folder}_{img}")

corn_count = len(os.listdir('./corn_or_not/corn'))
print(f"Corn images: {corn_count}")

# Collect all not_corn images first
all_not_corn = []
for src in other_plants:
    if not os.path.exists(src):
        print(f"⚠️  Skipping: {src}")
        continue
    folder_name = os.path.basename(src)
    for img in os.listdir(src):
        all_not_corn.append((f"{src}/{img}", f"./corn_or_not/not_corn/{folder_name}_{img}"))

# Randomly pick same count as corn
random.shuffle(all_not_corn)
selected = all_not_corn[:corn_count]  # take only 3852

for src_path, dst_path in selected:
    shutil.copy(src_path, dst_path)

not_corn_count = len(os.listdir('./corn_or_not/not_corn'))
print(f"Not-corn images: {not_corn_count}")
print(f"\n✓ Balanced! Corn: {corn_count} | Not-corn: {not_corn_count}")