import os
import random
import shutil
import pandas as pd

root = 'dataset/all_data/'
energy_path = os.path.join(root, 'energy')
local_path = os.path.join(root, 'local')
test_path = os.path.join('dataset/test')
train_path = os.path.join('dataset/train')

local_files = os.listdir(local_path)
global_file = pd.read_csv(os.path.join(root, 'global.csv'))

os.makedirs(test_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(os.path.join(test_path, 'energy'), exist_ok=True)
os.makedirs(os.path.join(test_path, 'local'), exist_ok=True)
os.makedirs(os.path.join(train_path, 'energy'), exist_ok=True)
os.makedirs(os.path.join(train_path, 'local'), exist_ok=True)

file_groups = {}

for file in local_files:
    group = file[:2]
    if group not in file_groups:
        file_groups[group] = []
    file_groups[group].append(file)

random.seed(50)

test_files = set()
for group, files in file_groups.items():
    test_n = int(len(files) * 0.15)
    test_files.update(random.sample(files, test_n))
    for file in files:
        if file in test_files:
            shutil.copy(os.path.join(local_path, file), os.path.join(test_path, 'local', file))
            shutil.copy(os.path.join(energy_path, file), os.path.join(test_path, 'energy', file))
        else:
            shutil.copy(os.path.join(local_path, file), os.path.join(train_path, 'local', file))
            shutil.copy(os.path.join(energy_path, file), os.path.join(train_path, 'energy', file))

test_names = []
for file in test_files:
    test_names.append(file.split('.')[0])

# global_file
test_global = global_file[global_file['name'].isin(test_names)]
train_global = global_file[~global_file['name'].isin(test_names)]

test_global.to_csv(os.path.join(test_path, 'global.csv'), index=False)
train_global.to_csv(os.path.join(train_path, 'global.csv'), index=False)
