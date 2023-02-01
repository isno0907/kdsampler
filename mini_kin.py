import pickle
import os
import pandas as pd
from tqdm import tqdm
label_map = pd.read_table("data/kinetics400/annotations/label_map_k400.txt", header=None)
mini_train = pd.read_table("data/mini_kinetics/mini_kinetics_train.txt", header=None, sep=' ')
mini_val = pd.read_table("data/mini_kinetics/mini_kinetics_val.txt", header=None, sep=' ')
kin_train = pd.read_csv("./data/kinetics400/annotations/kinetics_train.csv")
kin_test = pd.read_csv("./data/kinetics400/annotations/kinetics_test.csv")
kin_val = pd.read_csv("./data/kinetics400/annotations/kinetics_val.csv")

from pathlib import Path
import os
train_root = Path("data/kinetics400/rawframes_train")
file_lists = []
print("train")
for i in tqdm(range(len(mini_train)), colour="blue"):
# for i in range(20):
    id = mini_train.iloc[i][0]
    data = kin_train[kin_train.youtube_id == id]
    start= "%s" % ("%06d" % int(data.time_start.item()))
    end = "%s" % ("%06d" % int(data.time_end.item()))
    rel_path = train_root / data.label.item().replace(" ","_") / f"{data.youtube_id.item()}_{start}_{end}"
    new_path = os.path.join(data.label.item().replace(" ","_"), f"{data.youtube_id.item()}_{start}_{end}")
    if os.path.exists(rel_path):
        length = len(os.listdir(rel_path))
        label = mini_train.iloc[i][2]
        file_lists.append(f"{new_path} {length} {label}\n")
with open("data/mini_kinetics/new_mini_kinetics_train.txt", 'w') as f:
    f.writelines(file_lists)

from pathlib import Path
import os
val_root = Path("data/kinetics400/rawframes_val")
file_lists = []
print("validation")
for i in tqdm(range(len(mini_val)), colour="red"):
    id = mini_val.iloc[i][0]
    data = kin_val[kin_val.youtube_id == id]
    start= "%s" % ("%06d" % int(data.time_start.item()))
    end = "%s" % ("%06d" % int(data.time_end.item()))
    rel_path = val_root / data.label.item().replace(" ","_") / f"{data.youtube_id.item()}_{start}_{end}"
    new_path = os.path.join(data.label.item().replace(" ","_"), f"{data.youtube_id.item()}_{start}_{end}")
    if os.path.exists(rel_path):
        length = len(os.listdir(rel_path))
        label = mini_val.iloc[i][2]
        file_lists.append(f"{new_path} {length} {label}\n")
        # break
with open("data/mini_kinetics/new_mini_kinetics_val.txt", 'w') as f:
    f.writelines(file_lists)
