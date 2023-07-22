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

frame_path = '/media/storage3/xxx/xxx/datasets/UCF101-24/rgb-images'
box_path = '/media/storage3/xxx/xxx/datasets/UCF101-24/UCF101v2-GT.pkl'
box_file = open(box_path, 'rb')
box_data = pickle.load(box_file, encoding='latin1')


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


def generate(generate_path, write_file_name, list_file, sample_rate, background_images_path, multiple_per_video):
    if not os.path.isdir(generate_path):
        os.mkdir(generate_path)
    write_file = open(write_file_name, 'w')

    keys = []
    labels = {}
    with open(list_file) as f:
        data = f.readlines()
    for item in data:
        if np.random.rand(1) <= sample_rate:
            video_id = item.strip().split()[0]
            keys.append(video_id)
            labels[video_id] = item.strip().split()[2]

    if background_images_path != 'stripe':
        all_background_images = os.listdir(background_images_path)

    categories = os.listdir(frame_path)
    for ca in categories:
        ca_path = os.path.join(frame_path, ca)
        videos = os.listdir(ca_path)
        for video in videos:
            if video in keys:
                video_path = os.path.join(ca_path, video)
                frames = os.listdir(video_path)
                num_frames = len([f for f in frames if f.endswith('.jpg')])
                ## have gttubes annotations and check class index
                if '{}/{}'.format(ca, video) in box_data['gttubes'].keys() and list(box_data['gttubes']['{}/{}'.format(ca, video)].keys())[0] == box_data['labels'].index(ca):
                    box_list = box_data['gttubes']['{}/{}'.format(ca, video)][box_data['labels'].index(ca)]
                    box = box_list[0]
                    if len(box_list) > 1:
                        for box_ind in range(1, len(box_list)):
                            box = np.concatenate([box, box_list[box_ind]], axis=0)
                    anno_frame_index = list(box[:,0])
                    anno_frame_index_range = [int(min(anno_frame_index)), int(max(anno_frame_index))]
                    frame_index_shift = int(min(anno_frame_index)) - 1
                    background_image_list = []
                    for number in range(multiple_per_video):
                        # create genetation path
                        generate_video_path = os.path.join(generate_path, video+'__%d'%number)
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
                    for curr_frame_name in frames:
                        if curr_frame_name.endswith('.jpg'):
                            ## choose annotated frames
                            frame_index = int(curr_frame_name.split('.')[0])
                            if frame_index >= anno_frame_index_range[0] and frame_index <= anno_frame_index_range[1]:
                                curr_frame_path = os.path.join(video_path, curr_frame_name)
                                curr_frame = imageio.imread(curr_frame_path).astype('float')
                                curr_box_mask = np.zeros((curr_frame.shape[0], curr_frame.shape[1]))
                                if frame_index in anno_frame_index:
                                    curr_box = box[anno_frame_index.index(frame_index),:]
                                    assert curr_box[0] == frame_index
                                    curr_box_mask[int(curr_box[2]):int(curr_box[4]),int(curr_box[1]):int(curr_box[3])] = 1.0
                                curr_box_mask = np.expand_dims(curr_box_mask, axis=-1)
                                for number in range(multiple_per_video):
                                    # resize background image
                                    frame_shape = (curr_frame.shape[1], curr_frame.shape[0])
                                    resize_background_images = np.array(Image.fromarray(background_image_list[number]).resize(frame_shape)).astype('float')
                                    # generate image
                                    generate_frame = (curr_frame * curr_box_mask + resize_background_images * (1 - curr_box_mask)).astype('uint8')
                                    generate_frame_path = os.path.join(generate_path, video+'__%d'%number, 'frame{:06d}.jpg'.format(frame_index - frame_index_shift))
                                    imageio.imwrite(generate_frame_path, generate_frame)
                    for number in range(multiple_per_video):
                        original_label = labels[video]
                        write_item = '{} {:d} {}\n'.format(video+'__%d'%number, anno_frame_index_range[1] - anno_frame_index_range[0] + 1, original_label)
                        write_file.write(write_item)
    write_file.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--generate_path',type=str, help='')
    parser.add_argument('--sample_rate',type=float, help='')
    parser.add_argument('--write_file_name',type=str, help='')
    parser.add_argument('--list_file',type=str, help='')
    parser.add_argument('--background_images_path',type=str, help='')
    parser.add_argument('--multiple_per_video',type=int, help='')
    args = parser.parse_args()
    generate(generate_path=args.generate_path, write_file_name=args.write_file_name, list_file=args.list_file, sample_rate=args.sample_rate, background_images_path=args.background_images_path, multiple_per_video=args.multiple_per_video)
