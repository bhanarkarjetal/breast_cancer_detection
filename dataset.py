import shutil
import os
import kagglehub

# Download latest version from kaggle
# It will return the path where the dataset is downloaded and unzipped
local_path_dir = kagglehub.dataset_download("hayder17/breast-cancer-detection")

# create new folder to copy files from 'local_path_dir'
new_folder = "breast_cancer_data"

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# copying all the files from 'local_path_dir'
shutil.copytree(local_path_dir, new_folder, dirs_exist_ok=True)

# remove source directory
shutil.rmtree(local_path_dir)
