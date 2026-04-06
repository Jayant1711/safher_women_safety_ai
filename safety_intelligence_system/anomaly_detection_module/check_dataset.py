import kagglehub
import os

print("Downloading/Locating Geolife Dataset via Kagglehub...")
path = kagglehub.dataset_download("arashnic/microsoft-geolife-gps-trajectory-dataset")
print(f"Dataset securely staged at: {path}")

# List first 15 files found
file_count = 0
first_file = None

for root, dirs, files in os.walk(path):
    for f in files:
         if file_count == 0:
             first_file = os.path.join(root, f)
         if file_count < 15:
             print(os.path.join(root, f))
         file_count += 1

print(f"\nTotal files found: {file_count}")

if first_file:
    print(f"\n--- Previewing first few lines of {first_file} ---")
    try:
        with open(first_file, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                 print(line.strip())
                 if i > 10: break
    except Exception as e:
        print(f"Could not read file text: {e}")
