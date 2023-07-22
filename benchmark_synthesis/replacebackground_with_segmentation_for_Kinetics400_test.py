import os
import imageio
from PIL import Image
import numpy as np
import scipy.io
import random
import pickle
from datetime import datetime
import shutil

random.seed(int(datetime.now().timestamp()*1000%100000))
np.random.seed(int(datetime.now().timestamp()*1000%100000))

multiple_per_video = 1

frame_path = '/media/storage3/xxx/xxx/datasets/Kinetics_400/tmp_frames'


##CFFM trained on vspw dataset
human_class = 60
max_class = 130

# valid and smooth
area_valid_threshold = 0.1
length_valid_threshold = 0.9
smooth_range = 1


def stripe_img():
    crop_size = 256
    image_size = int(crop_size * 1.5)
    bg_color = np.random.rand(3)

    img = np.zeros((image_size,image_size,3))
    img[:,:,0] = bg_color[0]
    img[:,:,1] = bg_color[1]
    img[:,:,2] = bg_color[2]

    linewidth = 5 + 15 * np.random.rand(1)
    shift = (3 + 2 * np.random.rand(1)) * linewidth

    ## y=Asin(wx+t)+B
    w = 2 * np.pi / (np.random.rand(1) * 1000 + 4)
    t = (np.random.rand(1) * 200 - 100) * w
    A = 10 + np.random.rand(1) * 100
    B = 128

    x = np.arange(image_size)
    y = A*np.sin(w*x+t)+B
    valid_index = np.where((y>0) & (y<image_size))
    valid_x = x[valid_index]
    valid_y = y[valid_index]

    for ind in range(valid_x.shape[0]):
        xx = valid_x[ind]
        yy = valid_y[ind]
        min_yy = int(max(0, yy - linewidth))
        max_yy = int(min(255, yy + linewidth))

        img[min_yy:max_yy,xx,0] = 1 - bg_color[0]
        img[min_yy:max_yy,xx,1] = 1 - bg_color[1]
        img[min_yy:max_yy,xx,2] = 1 - bg_color[2]

        num_shift = 1
        while(int(max_yy-num_shift*shift)>0):
            img[max(0,int(min_yy-num_shift*shift)):int(max_yy-num_shift*shift),xx,0] = 1 - bg_color[0]
            img[max(0,int(min_yy-num_shift*shift)):int(max_yy-num_shift*shift),xx,1] = 1 - bg_color[1]
            img[max(0,int(min_yy-num_shift*shift)):int(max_yy-num_shift*shift),xx,2] = 1 - bg_color[2]
            num_shift += 1

        num_shift = 1
        while(int(min_yy+num_shift*shift)<image_size):
            img[int(min_yy+num_shift*shift):min(image_size, int(max_yy+num_shift*shift)),xx,0] = 1 - bg_color[0]
            img[int(min_yy+num_shift*shift):min(image_size, int(max_yy+num_shift*shift)),xx,1] = 1 - bg_color[1]
            img[int(min_yy+num_shift*shift):min(image_size, int(max_yy+num_shift*shift)),xx,2] = 1 - bg_color[2]
            num_shift += 1
    
    IMG = Image.fromarray(np.uint8(img*255))

    rotate_angle = np.random.rand(1) * 360
    IMG = IMG.rotate(rotate_angle, Image.NEAREST, expand = 1)
    width, height = IMG.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = (width + crop_size) // 2
    bottom = (height + crop_size) // 2
    IMG = IMG.crop((left, top, right, bottom))

    return np.asarray(IMG)


def generate(mask_path, saliency_path, generate_path, write_file_name, list_file, background_images_path, start, end):
    if not os.path.isdir(generate_path):
        os.mkdir(generate_path)
    write_file = open(write_file_name, 'w')

    keys = []
    labels = {}
    with open(list_file) as f:
        data = f.readlines()
    for item in data[start:end]:
        video = item.strip().split()[0].split('.')[0]
        keys.append(video)
        labels[video] = item.strip().split()[1]

    if background_images_path != 'stripe':
        all_background_images = os.listdir(background_images_path)

    for video in keys:
        video_id = video.split('/')[-1]
        video_path = os.path.join(frame_path, video)
        curr_mask_path = video_path.replace(frame_path, mask_path)
        curr_saliency_path = video_path.replace(frame_path, saliency_path)
        frames = os.listdir(video_path)
        num_frames = len([f for f in frames if f.endswith('.jpg')])
        assert num_frames == len([f for f in os.listdir(curr_mask_path) if f.endswith('.png')])
        assert num_frames == len([f for f in os.listdir(curr_saliency_path) if f.endswith('.png')])
        background_image_list = []
        for number in range(multiple_per_video):
            # create generation path
            generate_video_path = os.path.join(generate_path, video_id+'__%d'%number)
            print(generate_video_path)
            if not os.path.isdir(generate_video_path):
                os.mkdir(generate_video_path)
            # background image list
            if background_images_path != 'stripe':
                sample_background_image = random.sample(all_background_images, 1)[0]
                curr_background_image = imageio.imread(os.path.join(background_images_path, sample_background_image))
                while len(curr_background_image.shape) < 3:
                    sample_background_image = random.sample(all_background_images, 1)[0]
                    curr_background_image = imageio.imread(os.path.join(background_images_path, sample_background_image))
                if curr_background_image.shape[2] > 3:
                    curr_background_image = curr_background_image[:,:,:3]
            else:
                curr_background_image = stripe_img()
            background_image_list.append(curr_background_image)
        # generate frames
        valid_num = 0
        for curr_frame_name in frames:
            if curr_frame_name.endswith('.jpg'):
                ## frame_index
                frame_index = int(curr_frame_name.split('.')[0])
                ## read frame
                curr_frame_path = os.path.join(video_path, curr_frame_name)
                curr_frame = imageio.imread(curr_frame_path).astype('float')
                ## mask from human segmentation model
                valid_segmentation_mask = True
                cumulative_human_mask = []
                # smooth
                for smooth_ind in range(max(1,frame_index-smooth_range), min(num_frames,frame_index+smooth_range)+1):
                    mask_file = os.path.join(curr_mask_path, '%05d.png'%smooth_ind)
                    # human class mask
                    human_mask = np.array(Image.open(mask_file))
                    human_mask[human_mask == human_class] = max_class
                    human_mask[human_mask != max_class] = 0.0
                    human_mask[human_mask == max_class] = 1.0
                    cumulative_human_mask.append(human_mask)
                human_mask = np.sum(np.stack(cumulative_human_mask, axis=0), axis=0)
                human_mask[human_mask > 1.0] = 1.0
                segmentation_mask = human_mask
                # validness
                if np.sum(segmentation_mask) < area_valid_threshold * np.prod(segmentation_mask.shape):
                    valid_segmentation_mask = False
                ## mask from salency object model
                valid_saliency_mask = True
                cumulative_saliency_mask = []
                # smooth
                for smooth_ind in range(max(1,frame_index-smooth_range), min(num_frames,frame_index+smooth_range)+1):
                    mask_file = os.path.join(curr_saliency_path, '%05d.png'%smooth_ind)
                    saliency_mask = np.array(Image.open(mask_file))
                    cumulative_saliency_mask.append(saliency_mask)
                saliency_mask = np.sum(np.stack(cumulative_saliency_mask, axis=0), axis=0)
                saliency_mask[saliency_mask > 1.0] = 1.0
                # validness
                if np.sum(saliency_mask) < area_valid_threshold * np.prod(saliency_mask.shape):
                    valid_saliency_mask = False
                ## combination of the two masks
                curr_mask = saliency_mask + segmentation_mask
                curr_mask[curr_mask > 1.0] = 1.0
                if np.sum(curr_mask) >= area_valid_threshold * np.prod(segmentation_mask.shape):
                    valid_num += 1
                curr_mask = np.expand_dims(curr_mask, axis=-1)
                for number in range(multiple_per_video):
                    # resize background image
                    frame_shape = (curr_frame.shape[1], curr_frame.shape[0])
                    resize_background_images = np.array(Image.fromarray(background_image_list[number]).resize(frame_shape)).astype('float')
                    # generate image
                    generate_frame = (curr_frame * curr_mask + resize_background_images * (1 - curr_mask)).astype('uint8')
                    generate_frame_path = os.path.join(generate_path, video_id+'__%d'%number, '{:06d}.jpg'.format(frame_index))
                    imageio.imwrite(generate_frame_path, generate_frame)
        if valid_num < length_valid_threshold * num_frames:
            for number in range(multiple_per_video):
                shutil.rmtree(os.path.join(generate_path, video_id+'__%d'%number))
        else:
            for number in range(multiple_per_video):
                original_label = labels[video]
                write_item = '{} {:d} {}\n'.format(video+'__%d'%number, num_frames, original_label)
                write_file.write(write_item)
    write_file.close()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--mask_path',type=str, help='')
    parser.add_argument('--saliency_path',type=str, help='')
    parser.add_argument('--generate_path',type=str, help='')
    parser.add_argument('--write_file_name',type=str, help='')
    parser.add_argument('--list_file',type=str, help='')
    parser.add_argument('--background_images_path',type=str, help='')
    parser.add_argument('--multiple_per_video',type=int, help='')
    parser.add_argument('--start',type=int, help='')
    parser.add_argument('--end', type=int, help='')
    args = parser.parse_args()
    generate(mask_path=args.mask_path, saliency_path=args.saliency_path, generate_path=args.generate_path, write_file_name=args.write_file_name, list_file=args.list_file, background_images_path=args.background_images_path, start=args.start, end=args.end)
