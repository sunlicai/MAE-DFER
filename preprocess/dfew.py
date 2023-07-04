# *_*coding:utf-8 *_*
import os
import pandas as pd
import sys

# change 'data_path' to yours
data_path = 'path/to/your/dataset'
split_dir = os.path.join(data_path, 'EmoLabel_DataSplit')
video_dir = os.path.join(data_path, 'Clip/jpg_256')
assert len(sys.argv) > 1, 'Error: please specify split number (1-5)!'
split = f'{sys.argv[1]}'

save_dir = f'../saved/data/dfew/org/split0{split}' # org: original frames in jpg_256, not temporally aligned in jpg_224_16f
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read
train_split_file = os.path.join(split_dir, f'train(single-labeled)/set_{split}.csv')
df = pd.read_csv(train_split_file)
train_label_dict = dict(zip(df.video_name, df.label))

test_split_file = os.path.join(split_dir, f'test(single-labeled)/set_{split}.csv')
df = pd.read_csv(test_split_file)
test_label_dict = dict(zip(df.video_name, df.label))

train_label_dict = {os.path.join(video_dir, f'{v:05d}'):(l-1) for v,l in train_label_dict.items()}
test_label_dict = {os.path.join(video_dir, f'{v:05d}'):(l-1) for v,l in test_label_dict.items()}

# write
new_train_split_file = os.path.join(save_dir, f'train.csv')
df = pd.DataFrame(train_label_dict.items())
df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

new_test_split_file = os.path.join(save_dir, f'test.csv')
df = pd.DataFrame(test_label_dict.items())
df.to_csv(new_test_split_file, header=None, index=False, sep=' ')

## val == test
new_val_split_file = os.path.join(save_dir, f'val.csv')
df = pd.DataFrame(test_label_dict.items())
df.to_csv(new_val_split_file, header=None, index=False, sep=' ')
