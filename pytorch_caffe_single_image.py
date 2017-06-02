from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True # nfsmax
import torchvision.models as models
import io
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn as nn

## one time fix for bug in torch model zoo and jcjohnson's caffe conversion
# vgg_torch_caffe = torch.load('/home/nvidia/cnn_weights/pytorch/vgg16-00b39a1b.pth')
# vgg_torch_caffe['classifier.0.weight'] = vgg_torch_caffe['classifier.1.weight']
# vgg_torch_caffe['classifier.0.bias'] = vgg_torch_caffe['classifier.1.bias']
# vgg_torch_caffe['classifier.3.weight'] = vgg_torch_caffe['classifier.4.weight']
# vgg_torch_caffe['classifier.3.bias'] = vgg_torch_caffe['classifier.4.bias']
# del vgg_torch_caffe['classifier.4.weight']
# del vgg_torch_caffe['classifier.4.bias']
# del vgg_torch_caffe['classifier.1.weight']
# del vgg_torch_caffe['classifier.1.bias']

# torch.save(vgg_torch_caffe, "/home/nvidia/cnn_weights/pytorch/vgg16-00b39a1b.pth")

vgg16 = models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('/home/nvidia/cnn_weights/pytorch/vgg16-00b39a1b.pth'))

MEAN_IMAGE = np.array([103.939, 116.779, 123.68])
MEAN_IMAGE = MEAN_IMAGE.reshape([3,1,1])

def prep_image(url):
	ext = url.split('.')[-1]
	im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
	# Resize so smallest dim = 256, preserving aspect ratio
	h, w, _ = im.shape
	if h < w:
		im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
	else:
		im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
	# Central crop to 224x224
	h, w, _ = im.shape
	im = im[h//2-112:h//2+112, w//2-112:w//2+112]
	rawim = np.copy(im).astype('uint8')
	# Shuffle axes to c01
	im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
	# Convert to BGR
	im = im[::-1, :, :]
	im = im - MEAN_IMAGE
	im = im.astype('float32')
	x = torch.from_numpy(im)
	x = x.contiguous()
	return x

import urllib

index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
image_urls = index.split('<br>')

x = prep_image(image_urls[1])

# model.features
# Sequential (
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU (inplace)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU (inplace)
#   (4): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU (inplace)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU (inplace)
#   (9): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU (inplace)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU (inplace)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU (inplace)
#   (16): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): ReLU (inplace)
#   (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU (inplace)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU (inplace)
#   (23): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#   (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (25): ReLU (inplace)
#   (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (27): ReLU (inplace)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU (inplace)
#   (30): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
# )

# model = vgg16.cuda()
model_bla = nn.Sequential(*list(model.features)[:4])
x_to_pass = Variable(x.cuda(), volatile=True)
x_to_pass = x_to_pass.unsqueeze(0)

y = model(x_to_pass)
y_numpy = y.data.cpu().numpy()