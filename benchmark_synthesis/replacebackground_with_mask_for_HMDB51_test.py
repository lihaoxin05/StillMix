import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import imageio
from PIL import Image
import numpy as np
import scipy.io
import random
from datetime import datetime

random.seed(int(datetime.now().timestamp()*1000%100000))
np.random.seed(int(datetime.now().timestamp()*1000%100000))

frame_path = '/export/home/xxx/xxx/datasets/JHMDB/Rename_Images'
mask_path = '/export/home/xxx/xxx/datasets/JHMDB/puppet_mask'


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


def generate(list_file, sample_rate, generate_path, background_images_path, write_file_name, multiple_per_video):
    if not os.path.isdir(generate_path):
        os.mkdir(generate_path)
    if background_images_path != 'stripe':
        all_background_images = os.listdir(background_images_path)
    write_file = open(write_file_name, 'w')

    keys = []
    infos = {}
    with open(list_file) as f:
        data = f.readlines()
    for item in data:
        if np.random.rand(1) <= sample_rate:
            video_id = item.strip().split()[0]
            keys.append(video_id)
            infos[video_id] = item.strip()

    categories = os.listdir(frame_path)
    for ca in categories:
        ca_path = os.path.join(frame_path, ca)
        videos = os.listdir(ca_path)
        for video in videos:
            if video in keys:
                video_path = os.path.join(ca_path, video)
                frames = os.listdir(video_path)
                num_frames = len([f for f in frames if f.endswith('.png')])
                mask_file_path = os.path.join(mask_path, ca, video, 'puppet_mask.mat')
                if os.path.isfile(mask_file_path):
                    mask_file = scipy.io.loadmat(mask_file_path)
                    for number in range(multiple_per_video):
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

                        generate_video_path = os.path.join(generate_path, video+'__%d'%number)
                        print(generate_video_path)
                        if not os.path.isdir(generate_video_path):
                            os.mkdir(generate_video_path)
                            for curr_frame_name in frames:
                                if curr_frame_name.endswith('.png'):
                                    curr_frame_index = int(curr_frame_name.split('.')[0])-1
                                    if curr_frame_index < mask_file['part_mask'].shape[-1]:
                                        curr_frame_path = os.path.join(video_path, curr_frame_name)
                                        curr_frame = imageio.imread(curr_frame_path).astype('float')
                                        frame_shape = (curr_frame.shape[1], curr_frame.shape[0])
                                        curr_mask = np.expand_dims(mask_file['part_mask'][:,:,curr_frame_index].astype('float'), axis=-1)
                                        resize_background_image = np.array(Image.fromarray(curr_background_image).resize(frame_shape)).astype('float')
                                        generate_frame = (curr_frame * curr_mask + resize_background_image * (1 - curr_mask)).astype('uint8')
                                        generate_frame_path = os.path.join(generate_video_path, 'frame{:06d}.jpg'.format(int(curr_frame_name.split('.')[0])))
                                        imageio.imwrite(generate_frame_path, generate_frame)
                        original_item = infos[video]
                        label = original_item.split()[2]
                        write_item = '{} {:d} {}\n'.format(video+'__%d'%number, min(num_frames, mask_file['part_mask'].shape[-1]), label)
                        write_file.write(write_item)
    write_file.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--list_file',type=str, help='')
    parser.add_argument('--sample_rate',type=float)
    parser.add_argument('--generate_path',type=str, help='')
    parser.add_argument('--background_images_path',type=str, help='')
    parser.add_argument('--write_file_name',type=str, help='')
    parser.add_argument('--multiple_per_video',type=int, help='')
    args = parser.parse_args()
    generate(args.list_file, args.sample_rate, args.generate_path, args.background_images_path, args.write_file_name, args.multiple_per_video)
