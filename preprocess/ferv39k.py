# *_*coding:utf-8 *_*
import os
import pandas as pd

# change 'data_path' to yours
data_path = 'path/to/your/dataset'
split_dir = os.path.join(data_path, '4_setups/All_scenes')
video_dir = os.path.join(data_path, '2_ClipsforFaceCrop')

save_dir = f'../saved/data/ferv39k/all_scenes'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

label2idx = {'Happy': 0, 'Sad': 1, 'Neutral': 2, 'Angry': 3, 'Surprise': 4, 'Disgust': 5, 'Fear': 6}

# read
train_split_file = os.path.join(split_dir, f'train_All.csv')
df = pd.read_csv(train_split_file, header=None, delimiter=' ')
train_label_dict = dict(zip(df[0], df[1]))

test_split_file = os.path.join(split_dir, f'test_All.csv')
df = pd.read_csv(test_split_file, header=None, delimiter=' ')
test_label_dict = dict(zip(df[0], df[1]))

train_label_dict = {os.path.join(video_dir, v):label2idx[l] for v,l in train_label_dict.items()}
test_label_dict = {os.path.join(video_dir, v):label2idx[l] for v,l in test_label_dict.items()}

# write
new_train_split_file = os.path.join(save_dir, f'train.csv')
df = pd.DataFrame(train_label_dict.items())
df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

new_test_split_file = os.path.join(save_dir, f'test.csv')
df = pd.DataFrame(test_label_dict.items())
df.to_csv(new_test_split_file, header=None, index=False, sep=' ')

## val == test, simply specify in the code, do not generate the csv file
