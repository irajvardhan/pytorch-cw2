
# coding: utf-8

# In[1]:


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data as D
import os

import argparse
import cv2
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import torch.nn.functional as F
from skimage.transform import resize
from PIL import Image

from torch.utils.data.sampler import SubsetRandomSampler
from  torch.optim import lr_scheduler
from sal.nn_network import SaliencyModel
from sal.utils.resnet_encoder import resnet50encoder
from  torch.optim import lr_scheduler

import os


# In[5]:


NUM_CHANNELS = 1
PATCH_SIZE = 14 # it is 64 for 224x224 image
IMG_W = 28 # 224 originally
IMG_H = 28 # 224 originally
# MEANS = np.array([0.1307])
# STDS = np.array([0.3081])
MEANS = np.array([0.])
STDS = np.array([1.])

epsilons = [0, .05, .1, .15, .2, .25, .3]
use_cuda=True

dataset = 'fmnist'
source_class = 9
target_class = (source_class+1)%10


# In[6]:


num_epochs = 1
lr         = 1e-3
momentum   = 0.9
w_decay    = 1e-5
step_size  = 5
gamma      = 0.5

channel_dims=[64,256,512,1024,2048]
intr_model= SaliencyModel(resnet50encoder(pretrained=False), channel_dims)

criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(intr_model.parameters(), lr=lr, weight_decay=w_decay)
scheduler_nn=lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_intr_model = './model/'+dataset+'/'+str(target_class)+'/newnet_new.pth'

intr_model.eval()
intr_model = torch.load(pretrained_intr_model, map_location=device)
intr_model.cuda()


# In[7]:


class Net(nn.Module):
    def __init__(self, features, num_classes, init_weights=True):
        super(Net, self).__init__()
        
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(True),
            nn.Linear(500, num_classes)
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        # x are the logits values
        return x 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# In[8]:


"""
torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
"""

def make_layers(cfg, in_channels, kernel_size, stride, padding, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


"""
Refer VGG19_bn configurationh here: 
https://github.com/pytorch/vision/blob/76702a03d6cc2e4f431bfd1914d5e301c07bd489/torchvision/models/vgg.py#L63
"""
cfgs = {
    #'E': [64, 64, 'M',128, 128, 'M',256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',512, 512, 512, 512, 'M'],
    'E': [20, 'M', 50, 'M']
}

model_layers = make_layers(cfgs['E'],in_channels=1, kernel_size=5, stride=1, padding=0, batch_norm=False)


# In[9]:


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                # Store the features and register hook to save gradients for the target layer
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        # Retrieve the saved gradients for the target layer
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output

class GradCam:
    """
    This class computes the Grad-CAM mask for the specified index.
    """
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def __call__(self, image_tensor, index=None):
        features, output = self.extractor(image_tensor)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        # Compute the one-hot tensor corresponding to the index
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        # Get the gradients and features to compute Grad-CAM
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_W, IMG_W))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


# In[10]:


class CustomDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, path, train=True):
        """ Intialize the dataset
        """
        if train:
            data_path = os.path.join(path,'x_train.npy')
            targets_path = os.path.join(path,'y_train.npy')
        else:
            data_path = os.path.join(path,'x_test.npy')
            targets_path = os.path.join(path,'y_test.npy')

        self.path = data_path
        self.data = np.load(data_path)
        self.targets = np.load(targets_path)
        #self.transform = transforms.ToTensor()
        self.transform = transforms.Compose([
                       transforms.ToTensor()
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])
        self.len = np.shape(self.data)[0]
        
    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        data = self.data[index]
        #image = Image.fromarray(data)
        
        
        target = int(self.targets[index])
        
        #data = (data * 255).astype(np.uint8)
        #data = data.reshape(28,28)
        #image = Image.fromarray((data * 255).astype(np.uint8))
        #image = Image.fromarray(data.astype(np.uint8))
        
        return self.transform(data), target

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


# In[11]:


directory = './data/fmnist'

IS_DATA_READY = True
assert(IS_DATA_READY == True)

x_train = np.load(directory + '/x_train.npy')
y_train = np.load(directory + '/y_train.npy')
x_test = np.load(directory + '/x_test.npy')
y_test = np.load(directory + '/y_test.npy')
print('x_train shape : {}'.format(x_train.shape))
print('y_train shape : {}'.format(y_train.shape))
print('x_test shape : {}'.format(x_test.shape))
print('y_test shape : {}'.format(y_test.shape))


# Simple dataset. Only save path to image and load it and transform to tensor when call __getitem__.
filepath = './data/fmnist/'
train_set = CustomDS(filepath, train=True)
test_set = CustomDS(filepath, train=False)

# total images in set
print(train_set.len)
print(test_set.len)


# In[12]:


# main method
## Training settings
# input batch size for training (default: 64)
BATCH_SIZE = 20

# input batch size for testing (default: 1000)
TEST_BATCH_SIZE = 1

# number of epochs to train
EPOCHS = 10

#learning rate (default: 0.01)
LR = 0.01

#SGD momentum (default: 0.5)
MOMENTUM = 0.5

# how many batches to wait before logging training status
LOG_INTERVAL = 10

SAVE_MODEL = True
SEED = 1
NO_CUDA = False
USE_CUDA = not NO_CUDA and torch.cuda.is_available()

NUM_CLASSES=10

torch.manual_seed(SEED)

device = torch.device("cuda" if USE_CUDA else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}


# In[13]:


import torch
import operator as op
import functools as ft


'''reduce_* helper functions reduce tensors on all dimensions but the first.
They are intended to be used on batched tensors where dim 0 is the batch dim.
'''


def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def reduce_mean(x, keepdim=True):
    numel = ft.reduce(op.mul, x.size()[1:])
    x = reduce_sum(x, keepdim=keepdim)
    return x / numel


def reduce_min(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.min(a, keepdim=keepdim)[0]
    return x


def reduce_max(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.max(a, keepdim=keepdim)[0]
    return x


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def l2r_dist(x, y, keepdim=True, eps=1e-8):
    d = (x - y)**2
    d = reduce_sum(d, keepdim=keepdim)
    d += eps  # to prevent infinite gradient at 0
    return d.sqrt()


def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)


def l1_dist(x, y, keepdim=True):
    d = torch.abs(x - y)
    return reduce_sum(d, keepdim=keepdim)


def l2_norm(x, keepdim=True):
    norm = reduce_sum(x*x, keepdim=keepdim)
    return norm.sqrt()


def l1_norm(x, keepdim=True):
    return reduce_sum(x.abs(), keepdim=keepdim)


def rescale(x, x_min=-1., x_max=1.):
    return x * (x_max - x_min) + x_min


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


# In[14]:


"""PyTorch Carlini and Wagner L2 attack algorithm.

Based on paper by Carlini & Wagner, https://arxiv.org/abs/1608.04644 and a reference implementation at
https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
"""
import os
import sys
import torch
import numpy as np
from torch import optim
from torch import autograd
#from .helpers import *


class AttackCarliniWagnerL2:

    def __init__(self, targeted=True, search_steps=None, max_steps=None, cuda=True, debug=False):
        self.debug = debug
        self.targeted = targeted
        #self.num_classes = 1000
        self.num_classes = 10
        self.confidence = 0  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = 0.1  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 5
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or 1000
        self.abort_early = True
        self.clip_min = -1.
        self.clip_max = 1.
        self.cuda = cuda
        self.clamp_fn = 'tanh'  # set to something else perform a simple clamp instead of tanh
        self.init_rand = False  # an experiment, does a random starting point help?

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const, output_exp, target_exp):
        # compute the probability of the label class versus the maximum other
        
        # for the targeted attack, real will contain the current logit values for the targeted class
        # This basically tell us what is the current probability of the image being classified as the target class
        # multiplying by one hot encoded target ensures that other (index != target) logit values become 0
        # sum(1) simply gives us the logit value of the target class
        real = (target * output).sum(1)
        real_exp = (target_exp * output_exp).sum(1)
        
        # indices other than target class will have their logit values, target index will have -10000
        # takes the maximum value when we suppress the logit of the targeted class
        # this will give the logit of the most likely other class
        # in the first run, this would most likely be the prob of the true class
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        other_exp = ((1. - target_exp) * output_exp - target_exp * 10000.).max(1)[0]
        
#         print('output: ', output)
#         print('real: ', real)
#         print('other: ', other)
        
#         print('dist shape: ', dist.shape)
#         print('dist: ', dist)
        
#         print('output exp: ', output_exp)
#         print('real exp: ', real_exp)
#         print('other exp: ', other_exp)
        
        
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        
        loss1 = torch.sum(scale_const * loss1)
        loss2 = dist.sum()
        
        loss3 = torch.clamp(other_exp - real_exp + self.confidence, min=0.) 
        loss3 = torch.sum(scale_const * loss3)
        
        #print('loss2 which is dist.sum: ', loss2)

        loss = loss1 + loss2 + loss3
        #loss = loss3
        #loss = loss1 + loss2
        
#         print('loss1: ',loss1)
#         print('loss2: ',loss2)
#         print('loss3: ', loss3)
#         print('total loss is: ', loss)
        
        return loss

    def _optimize(self, optimizer, model, exp_model, intr_model, input_var,
                  modifier_var, target, target_var, target_var_exp, 
                  scale_const_var, gradcam, input_orig=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        if self.clamp_fn == 'tanh':
            input_adv = tanh_rescale(modifier_var + input_var, self.clip_min, self.clip_max)
        else:
            input_adv = torch.clamp(modifier_var + input_var, self.clip_min, self.clip_max)
        
        # input_adv is between [-1,1]
        # output logits from the base model f(.)
        output = model(input_adv)
#         index_base = np.argmax(output.cpu().data.numpy())
#         index_base_prob = torch.nn.functional.softmax(output)[0][index_base]
#         print('Classification F(.) of the input is class: {} with probability:{}'.format(index_base, index_base_prob))

        # distance to the original input data
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim=False)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim=False)
            
        # obtain the gradcam output for the adversarial image tensor
        input_tensor = nn.Upsample(scale_factor=8, mode='bilinear')(input_adv)
        input_tensor = input_tensor.to(device)
        output_intr = intr_model(input_tensor) # shape : torch.Size([1, 1, 224, 224])
        ave_p = nn.AvgPool2d(kernel_size=8, stride=None, padding=0)
        input_intr_tensor = ave_p(output_intr) #[1,1,28,28]

        # obtain prediction
        # Normalize -1 to 1....0 to 1 # TODO: see if this is not compatible with tanh thing done to input 
        input_intr_tensor = (input_intr_tensor - torch.min(input_intr_tensor))/(torch.max(input_intr_tensor) - torch.min(input_intr_tensor))
        output_exp = exp_model(input_intr_tensor)
#         index = np.argmax(output_exp.cpu().data.numpy())
#         index_prob = torch.nn.functional.softmax(output_exp)[0][index]
#         print('Classification G(.) of the explanation is class: {} with probability:{}'.format(index, index_prob))
        
        loss = self._loss(output, target_var, dist, scale_const_var, output_exp, target_var_exp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #loss_np = loss.data[0] #throws error
        loss_np = loss.data
        dist_np = dist.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        output_exp_np = output_exp.data.cpu().numpy()
        input_adv_np = input_adv.data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
        return loss_np, dist_np, output_np, output_exp_np, input_adv_np

    def run(self, model, exp_model, intr_model, input, target, gradcam, batch_idx=0):
        batch_size = input.size(0)
        # print('batch size: ', batch_size)
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = input.permute(0, 2, 3, 1).cpu().numpy()

        # setup input (image) variable, clamp/scale as necessary
        if self.clamp_fn == 'tanh':
            # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
            # this as per the reference implementation or can we skip the arctanh?
            input_var = autograd.Variable(torch_arctanh(input), requires_grad=False)
            input_orig = tanh_rescale(input_var, self.clip_min, self.clip_max)
        else:
            input_var = autograd.Variable(input, requires_grad=False)
            input_orig = None

        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        
        #target_onehot will have a 1 at the index of the targeted class (in the case of targeted attack)
        #print('target_onehot: ', target_onehot)
        
        target_var = autograd.Variable(target_onehot, requires_grad=False)
        
        
        ## setup the target variable for the explanation based model
        # target for the explanation based model is the negative class 0 i.e., one-hot target is [1, 0]
        target_exp = torch.tensor([0]).cuda()
        target_onehot_exp = torch.zeros(target_exp.size() + (2,)) # 2 is the num of classes in exp model
         
        if self.cuda:
            target_onehot_exp = target_onehot_exp.cuda()
        target_onehot_exp.scatter_(1, target_exp.unsqueeze(1), 1.)
        
        #target_onehot will have a 1 at the index of the targeted class (in the case of targeted attack)
#         print('target_onehot for base model: ', target_onehot)
#         print('target_onehot for exp model: ', target_onehot_exp)
        
        target_var_exp = autograd.Variable(target_onehot_exp, requires_grad=False)


        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        if self.init_rand:
            # Experiment with a non-zero starting point...
            modifier = torch.normal(means=modifier, std=0.001)
        if self.cuda:
            modifier = modifier.cuda()
        modifier_var = autograd.Variable(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=0.0005)

        for search_step in range(self.binary_search_steps):
            
            #print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if self.debug:
                #print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = autograd.Variable(scale_const_tensor, requires_grad=False)

            prev_loss = 1e6
            for step in range(self.max_steps):
                # perform the attack
                loss, dist, output, output_exp, adv_img = self._optimize(
                    optimizer,
                    model,
                    exp_model,
                    intr_model,
                    input_var,
                    modifier_var,
                    target,
                    target_var,
                    target_var_exp,
                    scale_const_var,
                    gradcam,
                    input_orig)

                if step % 100 == 0 or step == self.max_steps - 1:
                    print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, modifier mean: {3:.5e}'.format(step, loss, dist.mean(), modifier_var.data.mean()))

                if self.abort_early and step % (self.max_steps // 10) == 0:
                    if loss > prev_loss * .9999:
                        print('Aborting early...')
                        break
                    prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    target_exp_label = 0
                    output_logits = output[i]
                    output_exp_logits = output_exp[i]
                    output_label = np.argmax(output_logits)
                    output_exp_label = np.argmax(output_exp_logits)
                    
                    di = dist[i]
                    if self.debug:
                        if step % 100 == 0:
                            print('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    
                    if di < best_l2[i] and self._compare(output_logits, target_label) and self._compare(output_exp_logits, target_exp_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                    
                    if di < o_best_l2[i] and self._compare(output_logits, target_label) and self._compare(output_exp_logits, target_exp_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[i] = adv_img[i]
                    
                    #print('o_best_l2: ', o_best_l2[i])
                    cond1 = di < o_best_l2[i]
                    cond2 = self._compare(output_logits, target_label)
                    cond3 = self._compare(output_exp_logits, target_exp_label)
                    #print('cond1: {} and cond2:{} and cond3: {}'.format(cond1, cond2, cond3))
                
                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if self.debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if self.debug:
                        print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop
            
            # added by raj
            if batch_success == batch_size:
                break
                
        return o_best_attack, o_best_l2


# In[15]:


TARGETED = True
MAX_STEPS = 1000
SEARCH_STEPS = 6
NO_CUDA = False
DEBUG = False

attack = AttackCarliniWagnerL2(
        targeted=TARGETED,
        max_steps=MAX_STEPS,
        search_steps=SEARCH_STEPS,
        cuda=not NO_CUDA,
        debug=DEBUG)


# In[16]:


train_loader = D.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
test_loader = D.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# # Initialize the network
pretrained_model = "model/fmnist/v2/fmnist_cnn.pt"
# Initialize the network
model = Net(model_layers, num_classes=NUM_CLASSES).cuda()

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

##########################################


# # Initialize the explanation model
NUM_CLASSES_EXP = 2 
class_ind = 6
model_dir = 'data/defender/fmnist/exp_model_data/for_target/' + str(class_ind) + '/model'
pretrained_exp_model = model_dir+"/exp_model.pt"

exp_model = Net(model_layers, num_classes=NUM_CLASSES_EXP).cuda()

# Load the pretrained model
exp_model.load_state_dict(torch.load(pretrained_exp_model))

# Set the model in evaluation mode. In this case this is for the Dropout layers
exp_model.eval()


# In[17]:


desired_source_label = source_class
input_tensors = []
input_labels = []
for batch_idx, (input_tensor, input_label) in enumerate(test_loader):
    for i, label in enumerate(input_label):
        if label == desired_source_label:
            input_tensors.append(input_tensor[i].cpu().numpy())
            
input_tensors = torch.from_numpy(np.array(input_tensors))
input_label = torch.from_numpy(np.array([desired_source_label]*BATCH_SIZE))


# In[ ]:


loader = train_loader

iterations = 1
i = 20
thresh_on_iterations = 5

success_record = []

gradcam = GradCam(model, target_layer_names=["4"])

best_l2s = []

input_label = torch.from_numpy(np.array([desired_source_label]*BATCH_SIZE))

while i < 100:
    input_tensor = input_tensors[i:i+BATCH_SIZE]
    
    print('\n\n-------iteration: {}----------'.format(iterations))
    
    # clean image
    input_tensor = input_tensor.float().cuda()
    
    # original label for the clean image
    input_label = input_label.cuda() ## input_label.item() will give you the scalar label
    
    pred_input = model(input_tensor)
    pred_prob_input = F.softmax(pred_input, dim=1)
    
    # if the attack is targeted, we will target the next class modulo number of classes
    if TARGETED == True:
        target = (input_label+1)%10
    # else the target is kept as the original label as per attack design
    else:
        target = input_label
    
    # result obtained is a numpy array
    adversarial_img, best_l2 = attack.run(model, exp_model, intr_model, input_tensor, target, gradcam, batch_idx)
    
    # reshape
    adversarial_img = np.transpose(adversarial_img, (0,3,1,2))
    
    #adv_imgs.append(adversarial_img)
    best_l2s.extend(best_l2)
    
    # conver to torch tensor
    adversarial_tensor = torch.from_numpy(adversarial_img).cuda()
    
    # obtain the prediction by the model
    pred_adv = model(adversarial_tensor)
    pred_prob_adv = F.softmax(pred_adv, dim=1)
    pred_class_adv = torch.argmax(pred_prob_adv, dim=1)
    
    if TARGETED:
        result = pred_class_adv == target
    else:
        result = pred_class_adv != input_label
    
    result_np = result.cpu().numpy()
    success_record.extend(result_np)
    
    # log results
    success_record_npy = np.array(success_record)
    best_l2s_npy = np.array(best_l2s)
    ind_succ = np.where(success_record_npy==True)[0]
    succ_l2s = best_l2s_npy[ind_succ]
    print('succ:{} fail:{} mean L2 of succ attacks:{}'.format(succ_l2s.shape[0], success_record_npy.shape[0] - succ_l2s.shape[0], np.mean(succ_l2s)))
    
    target_str = str(target[0].item())
    folderpath = 'results/adaptive/' + dataset + '/' + target_str
    
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    
    # save batch result
    print('saving to ',folderpath)
    np.save(folderpath+'/batch_' + str(iterations) + '_result.npy',result_np)
    np.save(folderpath+'/batch_' + str(iterations) + '_best_l2s.npy',best_l2)
    
    
    iterations += 1
    i += BATCH_SIZE
    if iterations == thresh_on_iterations:
        break


# In[ ]:


# save overall result
np.save(folderpath+'/success_record.npy',success_record)
np.save(folderpath+'/best_l2s.npy',best_l2s)

