#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import time
import copy
import numpy as np
import math

# JETSON?
jetson = True

sys.argv = ['options', '--classes', '40', '--dataroot', 'modelnet40-normal_numpy/',
           '--checkpoints_dir', 'checkpoints/']
if jetson:
  sys.argv += ['--batch_size', '4']
from modelnet.options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from models.classifier import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from util.visualizer import Visualizer

from scipy import stats


# In[2]:


testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)


# In[3]:


model_probabilities_list = []


models_dir = [f for f in os.listdir('checkpoints') if os.path.isdir(os.path.join('checkpoints', f))]
models_dir = [m for m in models_dir if m != 'train']

for model_name in models_dir:
    print('Evaluating ' + model_name)
    if 'model' not in model_name:
        continue
    # create model, optionally load pre-trained model
    model = Model(opt)
    model_path = os.path.join(opt.checkpoints_dir, model_name)
    checkpoints = [x for x in os.listdir(model_path) if os.path.splitext(x)[-1] == '.pth']
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[0]))[-2:]
    for item in checkpoints:
        if item.split('_')[-1] == 'classifier.pth':
            ckp_classifier = item
        if item.split('_')[-1] == 'encoder.pth':
            ckp_encoder = item
    classifier_ckp_path = os.path.join(model_path, ckp_classifier)
    encoder_ckp_path = os.path.join(model_path, ckp_encoder)
    model.encoder.load_state_dict(torch.load(encoder_ckp_path))
    model.classifier.load_state_dict(torch.load(classifier_ckp_path))
    model.test_loss.data.zero_()
    model.test_accuracy.data.zero_()
    model_probabilities = []
    gt_labels = []
    cnt = 0
    total_time = 0.
    total_batches = 0.
    for i, data in enumerate(testloader):
        input_pc, input_sn, input_label, input_node, input_node_knn_I = data
        model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
        dur = model.test_model()

        # Eval and get duration
        if total_batches != 0:
             total_time += dur
        total_batches += 1
        if jetson:
            print ("BATCH:", total_batches, "TIME:", dur)

        model_probabilities.append(model.score.data)
        cnt += model.input_label.size()[0]
        gt_labels.append(model.input_label.cpu())
    model_probabilities = torch.cat(model_probabilities, 0).cpu().numpy()
    gt_labels = torch.cat(gt_labels, 0).numpy()
    model_probabilities_list.append(model_probabilities)
    if jetson:
        print (model_name, 'evaluated in', total_time / (total_batches-1), 'per batch!')
        exit()
print('Done')


# In[4]:


models_probabilities = np.stack(model_probabilities_list)
models_predictions = np.argmax(models_probabilities, -1)
correct_mask = np.equal(models_predictions, gt_labels).astype(np.float)
models_accuracy = np.mean(correct_mask, -1)
mean_accuracy = np.mean(models_accuracy)
max_accuracy = np.max(models_accuracy)
print('Mean model accuracy =', mean_accuracy)
print('Max model accuracy =', max_accuracy)

preditions_ens = np.squeeze(stats.mode(models_predictions)[0])
correct_mask_ens = np.equal(preditions_ens, gt_labels).astype(np.float)
accuracy_ens = np.mean(correct_mask_ens)
print('Ensembling (mode aggregation) accuracy =', accuracy_ens)

preditions_ens = np.argmax(np.mean(models_probabilities, 0), -1)
correct_mask_ens = np.equal(preditions_ens, gt_labels).astype(np.float)
accuracy_ens = np.mean(correct_mask_ens)
print('Ensembling (mean aggregation) accuracy =', accuracy_ens)


# In[5]:


models_probabilities_t = np.transpose(models_probabilities, axes=(1, 2, 0))
print(models_probabilities_t.shape, gt_labels.shape)


# In[6]:


np.save('probabilities.npy', models_probabilities_t)
np.save('true_labels.npy', gt_labels)


# In[ ]:




