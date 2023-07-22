import pickle as pkl
import numpy as np
import argparse

# list_file = '/export/xxx/xxx/work/dataset_config/UCF101-24/lists_generated/v1/replacebackground_with_box_for_UCF101_test_place365/101classes/testlist01.txt'
# # list_file = '/export/xxx/xxx/work/dataset_config/JHMDB/lists_generated/v1/replacebackground_with_mask_for_HMDB51_test_place365/51classes/testlist01.txt'
# res1_file = '/export/xxx/xxx/work/video_classification/stillmix/mmaction2/work_dir/results1.pkl'
# res2_file = '/export/xxx/xxx/work/video_classification/stillmix/mmaction2/work_dir/results2.pkl'

def parse_args():
    parser = argparse.ArgumentParser(description='Compare')
    parser.add_argument('--dynamic_list_file', default=None)
    parser.add_argument('--static_list_file', default=None)
    parser.add_argument('--res_file', default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.dynamic_list_file) as f:
        data = f.readlines()
    dynamic_labels = []
    for item in data:
        dynamic_labels.append(int(item.strip().split()[-1]))
    with open(args.static_list_file) as f:
        data = f.readlines()
    static_labels = []
    for item in data:
        static_labels.append(int(item.strip().split()[-1]))

    res = pkl.load(open(args.res_file, 'rb'))

    dynamic_count = 0
    static_count = 0
    dynamic_more_count = 0
    for ii in range(len(dynamic_labels)):
        if np.argmax(res[ii]) == dynamic_labels[ii]:
            dynamic_count += 1
        if np.argmax(res[ii]) == static_labels[ii]:
            static_count += 1
        if res[ii][dynamic_labels[ii]] > res[ii][static_labels[ii]]:
            dynamic_more_count += 1
    print(dynamic_count/len(dynamic_labels), static_count/len(dynamic_labels), dynamic_more_count/len(dynamic_labels))

if __name__ == '__main__':
    main()

# python ../tools/compare_mix_SCUB_SCUF_results.py --dynamic_list_file /xxx/xxx/work/dataset_config/Kinetics_400/lists_generated/v3/mix_dynamic_static_same_bg_for_test_stripe/val/dynamic_testlist01.txt --static_list_file /xxx/xxx/work/dataset_config/Kinetics_400/lists_generated/v3/mix_dynamic_static_same_bg_for_test_stripe/val/static_testlist01.txt --res_file results.pkl