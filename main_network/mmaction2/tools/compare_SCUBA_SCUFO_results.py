import pickle as pkl
import numpy as np
import argparse

# list_file = '/export/xxx/xxx/work/dataset_config/UCF101-24/lists_generated/v1/replacebackground_with_box_for_UCF101_test_place365/101classes/testlist01.txt'
# # list_file = '/export/xxx/xxx/work/dataset_config/JHMDB/lists_generated/v1/replacebackground_with_mask_for_HMDB51_test_place365/51classes/testlist01.txt'
# res1_file = '/export/xxx/xxx/work/video_classification/stillmix/mmaction2/work_dir/results1.pkl'
# res2_file = '/export/xxx/xxx/work/video_classification/stillmix/mmaction2/work_dir/results2.pkl'

def parse_args():
    parser = argparse.ArgumentParser(description='Compare')
    parser.add_argument('--list_file', default=None)
    parser.add_argument('--res1_file', default=None)
    parser.add_argument('--res2_file', default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.list_file) as f:
        data = f.readlines()
    labels = []
    for item in data:
        labels.append(int(item.strip().split()[2]))

    res1 = pkl.load(open(args.res1_file, 'rb'))
    res2 = pkl.load(open(args.res2_file, 'rb'))

    count = 0
    for ii in range(len(labels)):
        if np.argmax(res1[ii]) == labels[ii] and np.argmax(res2[ii]) != labels[ii]:
            count += 1
    print(count/len(labels))

if __name__ == '__main__':
    main()