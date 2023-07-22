CUDA_VISIBLE_DEVICES=3 python replacebackground_with_box_for_UCF101_test.py \
--generate_path /export/home/xxx/xxx/datasets/UCF101-24/generated/v2/replacebackground_with_box_for_UCF101_test_place365/generated_videos \
--write_file_path /export/home/xxx/xxx/work/dataset_config/UCF101-24/lists_generated/v2/replacebackground_with_box_for_UCF101_test_place365/101classes/testlist01.txt \
--list_file /export/home/xxx/xxx/work/dataset_config/UCF101/lists/testlist01.txt \
--background_images_path /export/home/xxx/xxx/datasets/place365/test_256 \
--multiple_per_video 5 --sample_rate 1.0