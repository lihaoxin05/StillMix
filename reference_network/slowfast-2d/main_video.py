import os
import copy
import pickle
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import decord

channel_factor = 8

def change_channel(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            if child.in_channels > channel_factor:
                setattr(model, child_name, nn.Conv2d(child.in_channels // channel_factor, child.out_channels // channel_factor, kernel_size=child.kernel_size, padding=child.padding, stride=child.stride, bias=child.bias))
            else:
                setattr(model, child_name, nn.Conv2d(child.in_channels, child.out_channels // channel_factor, kernel_size=child.kernel_size, padding=child.padding, stride=child.stride, bias=child.bias))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, nn.BatchNorm2d(child.num_features // channel_factor, eps=child.eps, momentum=child.momentum, affine=child.affine, track_running_stats=child.track_running_stats))
        else:
            change_channel(child)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class VideoRecord(object):
    def __init__(self, row):
        self._data = row
        self.path =  self._data[0]
        self.label = int(self._data[1])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root_path, list_file, transform=None, mode='video', number_of_frame=1):
        self.data_root_path = data_root_path
        self.list_file = list_file
        self.transform = transform
        self.mode = mode
        self.number_of_frame = number_of_frame
        self._parse_list()

    def _load_image(self, video_path, choose_num_frames, frame_index=None):
        try:
            video_full_path = os.path.join(self.data_root_path, video_path)
            video = decord.VideoReader(video_full_path)
            if frame_index is not None:
                frames = video.get_batch([frame_index])
            else:
                num_frames = len(video)
                choose_index = list(np.random.choice(num_frames, choose_num_frames))
                frames = video.get_batch([choose_index])
            frames = frames.asnumpy()
            return_frames = []
            for ii in range(frames.shape[0]):
                return_frames.append(Image.fromarray(frames[ii], 'RGB'))
            return return_frames
        except Exception:
            print('error loading image:', os.path.join(self.data_root_path, video_path))
            assert False

    def _parse_list(self):
        tmp = [x.strip().split() for x in open(self.list_file)]
        if self.mode == 'video':
            self.image_list = [VideoRecord(item) for item in tmp]
        elif self.mode == 'frame':
            self.image_list = []
            for item in tmp:
                record = VideoRecord(item)
                video_full_path = os.path.join(self.data_root_path, record.path)
                video = decord.VideoReader(video_full_path)
                num_frame = len(video)
                for ii in range(0,num_frame,20):
                    record_c = copy.deepcopy(record)
                    record_c.frame_index = ii
                    self.image_list.append(record_c)
        print('image number:%d' % (len(self.image_list)))

    def __getitem__(self, index):
        record = self.image_list[index]
        if self.mode == 'video':
            image = self._load_image(record.path, self.number_of_frame)
            return_path = record.path
        elif self.mode == 'frame':
            image = self._load_image(record.path, self.number_of_frame, record.frame_index)
            return_path = record.path + '//%d'%record.frame_index
        process_data = []
        for img in image:
            process_data.append(self.transform(img))
        process_data = torch.stack(process_data, dim=0).squeeze(0)
        return return_path, process_data, record.label

    def __len__(self):
        return len(self.image_list)


class Model(nn.Module):
    def __init__(self, arch, num_class, pretrain=True):
        super(Model, self).__init__()
        self.arch = arch
        self.num_class = num_class
        self.pretrain = pretrain
        self.prepare_model()
        
    def prepare_model(self):
        self.slow_path = getattr(models, self.arch)(pretrained=self.pretrain)
        self.fast_path = getattr(models, self.arch)(pretrained=False)
        feature_dim = self.slow_path.fc.in_features + self.fast_path.fc.in_features
        change_channel(self.fast_path)
        lateral_names = ['conv1','layer1','layer2','layer3']
        lateral_channels = []
        for name in lateral_names:
            if name == 'conv1':
                in_channels = getattr(self.fast_path, name).out_channels
            else:
                in_channels = getattr(self.fast_path, name)[-1].bn3.num_features
            lateral_channels.append(in_channels)
            setattr(self, name+'_lateral', \
            nn.Sequential(nn.Conv2d(in_channels, in_channels*2, kernel_size=1, stride=1, bias=False), \
            nn.BatchNorm2d(in_channels*2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True), \
            nn.ReLU())
            )
        add_channel_names = ['layer1','layer2','layer3','layer4']
        for ii in range(len(add_channel_names)):
            name = add_channel_names[ii]
            add_channels = lateral_channels[ii]
            conv_kernel = getattr(self.slow_path,name)[0].conv1
            new_conv_kernel = nn.Conv2d(conv_kernel.in_channels+add_channels*2, conv_kernel.out_channels, kernel_size=conv_kernel.kernel_size, padding=conv_kernel.padding, stride=conv_kernel.stride, bias=conv_kernel.bias)
            new_conv_kernel.weight.data[:,:conv_kernel.in_channels,:,:] = conv_kernel.weight.data
            getattr(self.slow_path,name)[0].conv1 = new_conv_kernel
            downsample_kernel = getattr(self.slow_path,name)[0].downsample[0]
            new_downsample_kernel = nn.Conv2d(downsample_kernel.in_channels+add_channels*2, downsample_kernel.out_channels, kernel_size=downsample_kernel.kernel_size, padding=downsample_kernel.padding, stride=downsample_kernel.stride, bias=downsample_kernel.bias)
            new_downsample_kernel.weight.data[:,:downsample_kernel.in_channels,:,:] = downsample_kernel.weight.data
            getattr(self.slow_path,name)[0].downsample[0] = new_downsample_kernel
        # self.fast_path = nn.Sequential(*(list(self.fast_path.children())[:-1]))
        # self.slow_path = nn.Sequential(*(list(self.slow_path.children())[:-1]))
        feature_dim = self.slow_path.fc.in_features + self.fast_path.fc.in_features // channel_factor
        self.slow_path.fc = Identity()
        self.fast_path.fc = Identity()
        self.fc = nn.Linear(feature_dim, self.num_class)

    def forward(self, input):
        x_fast = self.fast_path.conv1(input)
        x_fast = self.fast_path.bn1(x_fast)
        x_fast = self.fast_path.relu(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)
        x_slow = self.slow_path.conv1(input)
        x_slow = self.slow_path.bn1(x_slow)
        x_slow = self.slow_path.relu(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)
        x_fast_lateral = self.conv1_lateral(x_fast)
        x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
        for name in ['layer1','layer2','layer3','layer4']:
            x_fast = getattr(self.fast_path, name)(x_fast)
            x_slow = getattr(self.slow_path, name)(x_slow)
            if name != 'layer4':
                x_fast_lateral = getattr(self, name+'_lateral')(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
        x_slow = self.slow_path.avgpool(x_slow)
        x_fast = self.fast_path.avgpool(x_fast)
        output = torch.cat((x_slow, x_fast), dim=1).squeeze(-1).squeeze(-1)
        output = self.fc(output)
        return output


def load_split_train_test(datadir, train_list, test_list, batch_size):
    train_transforms = transforms.Compose([transforms.Resize((256,256)),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])
    train_data = Dataset(datadir, train_list, transform=train_transforms)
    test_data = Dataset(datadir, test_list, transform=test_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
    return trainloader, testloader


def main(args):
    if args.train:
        trainloader, testloader = load_split_train_test(args.data_dir, args.train_list, args.val_list, args.batch_size)
        model = Model(args.arch, args.num_class, args.pretrain)
        for param in model.parameters():
            param.requires_grad = True
        print(model)

        model.cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1)

        training_loss = []
        training_acc = []
        val_loss = []
        val_acc = []
        for epoch in range(args.epochs):
            running_training_loss = 0
            running_training_acc = 0
            running_num = 0
            for _, inputs, labels in trainloader:
                optimizer.zero_grad()
                outputs = model.forward(inputs.cuda())
                loss = criterion(outputs, labels.cuda())
                running_training_loss += loss.item() * inputs.shape[0]
                top_class = torch.argmax(outputs, dim=-1)
                equals = top_class == labels.cuda().view(*top_class.shape)
                running_training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                running_num += inputs.shape[0]
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch % args.print_every_epoch == 0:
                test_loss = 0
                accuracy = 0
                sample_count = 0
                model.eval()
                with torch.no_grad():
                    for _, inputs, labels in testloader:
                        outputs = model.forward(inputs.cuda())
                        batch_loss = criterion(outputs, labels.cuda())
                        top_class = torch.argmax(outputs, dim=-1)
                        equals = top_class == labels.cuda().view(*top_class.shape)
                        accuracy += torch.sum(equals.type(torch.FloatTensor)).item()
                        test_loss += batch_loss.item() * inputs.shape[0]
                        sample_count += inputs.shape[0]
                    accuracy /= sample_count
                print(f"Epoch {epoch+1}/{args.epochs}.. "
                        f"Train loss: {running_training_loss/running_num:.3f}.. "
                        f"Test loss: {test_loss/sample_count:.3f}.. "
                        f"Test accuracy: {accuracy:.3f}")
                training_loss.append(running_training_loss/running_num)
                training_acc.append(running_training_acc/running_num)
                val_loss.append(test_loss/sample_count)
                val_acc.append(accuracy)
                model.train()
                if epoch % 10 == 0 or epoch == args.epochs - 1:
                    torch.save(model, '%s.pth'%(args.model_name))
        np.save('training-statistic-%s.npy'%args.model_name, np.array([training_loss, training_acc, val_loss, val_acc]))

    if args.test:
        test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        test_data = Dataset(args.data_dir, args.test_list, transform=test_transforms, mode=args.test_mode, number_of_frame=args.video_mode_frame)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=4, shuffle=False)
        
        model=torch.load('%s.pth'%args.model_name)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        accuracy = 0
        test_loss = 0
        sample_count = 0
        if args.save_test_results:
            save_results = {}
        with torch.no_grad():
            for path, inputs, labels in testloader:
                if len(inputs.shape) == 5:
                    reshape = True
                    batch_size, nframe, c, h, w = inputs.shape
                    inputs = inputs.view(batch_size*nframe, c, h, w)
                else:
                    reshape = False
                sample_count += len(path)
                inputs = inputs.cuda()
                outputs = model.forward(inputs)
                if reshape:
                    _, hid = outputs.shape
                    outputs = outputs.view(batch_size, nframe, hid).mean(1).contiguous()
                batch_loss = criterion(outputs, labels.cuda())
                test_loss += batch_loss * len(path)
                top_class = torch.argmax(outputs, dim=-1)
                probs = F.softmax(outputs/args.softmax_temp, dim=-1)
                for i in range(len(path)):
                    equals = top_class[i] == labels[i].cuda()
                    accuracy += equals.type(torch.FloatTensor)
                    if args.save_test_results:
                        save_results[path[i]] = [probs[i][labels[i].cuda()].detach().cpu().item(), probs[i][top_class[i]].detach().cpu().item()]
        accuracy /= sample_count
        test_loss /= sample_count
        print(accuracy)
        print(f"Test accuracy: {torch.mean(accuracy):.3f}")
        print(test_loss)
        print(f"Test loss: {torch.mean(test_loss):.3f}")
        if args.save_test_results:
            args.result_flie_name = args.result_flie_name.replace('.pkl', '_%s_cali%.2f.pkl'%(args.model_name, args.softmax_temp))
            with open(args.result_flie_name, 'wb') as f:
                pickle.dump(save_results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='some description')
    parser.add_argument('--num_class', type=int, default=101, help='')
    parser.add_argument('--data_dir', type=str, default='/home/xxx/xxx/datasets/UCF101/jpegs_256', help='')
    parser.add_argument('--train_list', type=str, default='/home/xxx/xxx/work/dataset_config/UCF101/lists/trainlist01.txt', help='')
    parser.add_argument('--val_list', type=str, default='/home/xxx/xxx/work/dataset_config/UCF101/lists/trainlist01.txt', help='')
    parser.add_argument('--test_list', type=str, default='/home/xxx/xxx/work/dataset_config/UCF101/lists/trainlist01.txt', help='')
    parser.add_argument('--train', default=False, action="store_true", help='')
    parser.add_argument('--test', default=False, action="store_true", help='')
    parser.add_argument('--test_mode', type=str, default='frame', help='')
    parser.add_argument('--video_mode_frame', type=int, default=1, help='')
    parser.add_argument('--softmax_temp', type=float, default=1.0, help='')
    parser.add_argument('--save_test_results', default=False, action="store_true", help='')
    parser.add_argument('--arch', type=str, default='resnet50', help='')
    parser.add_argument('--pretrain', default=False, action="store_true", help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--lr', type=float, default=0.0001,help='')
    parser.add_argument('--epochs', type=int, default=300, help='')
    parser.add_argument('--lr_step', type=int, default=[6, 9], nargs="+", help='')
    parser.add_argument('--wd', type=float, default=1e-4, help='')
    parser.add_argument('--print_every_epoch', type=int, default=1, help='')
    parser.add_argument('--model_name', type=str, default='ucf101-split01-resnet50-nopretrain', help='')
    parser.add_argument('--result_flie_name', type=str, default='./results.pkl', help='')
    args = parser.parse_args()
    main(args)
