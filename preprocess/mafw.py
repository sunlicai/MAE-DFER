# *_*coding:utf-8 *_*
import os
import pandas as pd
import sys

# change 'data_path' to yours
data_path = 'path/to/your/dataset'
split_dir = os.path.join(data_path, 'split/single/no_caption')
video_dir = os.path.join(data_path, 'data/frames')
assert len(sys.argv) > 1, 'Error: please specify split number (1-5)!'
split = f'{sys.argv[1]}'

save_dir = f'../saved/data/mafw/single/split0{split}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'anxiety', 'helplessness', 'disappointment']
label2idx = {l:idx for idx, l in enumerate(labels)}

# read
train_split_file = os.path.join(split_dir, f'set_{split}/train.txt')
df = pd.read_csv(train_split_file, header=None, delimiter=' ')
train_label_dict = dict(zip(df[0], df[1]))

test_split_file = os.path.join(split_dir, f'set_{split}/test.txt')
df = pd.read_csv(test_split_file, header=None, delimiter=' ')
test_label_dict = dict(zip(df[0], df[1]))

train_label_dict = {os.path.join(video_dir, f"{v.split('.')[0]}"):label2idx[l] for v,l in train_label_dict.items()}
test_label_dict = {os.path.join(video_dir,  f"{v.split('.')[0]}"):label2idx[l] for v,l in test_label_dict.items()}

# write
new_train_split_file = os.path.join(save_dir, f'train.csv')
df = pd.DataFrame(train_label_dict.items())
df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

new_test_split_file = os.path.join(save_dir, f'test.csv')
df = pd.DataFrame(test_label_dict.items())
df.to_csv(new_test_split_file, header=None, index=False, sep=' ')

## val == test, simply specify in the code, do not generate the csv file