import os
import numpy as np
import glob
from scipy.misc import *


from skimage.util import view_as_blocks, view_as_windows, montage
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import time
import os
import copy
import torch
from torch import tensor 
from torch.autograd import Variable
from torch import randn, matmul
import matplotlib.pyplot as plt
from progressbar import progressbar
from google.colab import files
import matplotlib.pyplot as plt
import time
from torchvision.models import *
from PIL import Image


def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x)
    ax.axis('off')
    fig.set_size_inches(8, 8)
    plt.show()
    
    

def read_ims(directory, imsz):


    main_dir = os.getcwd()
    os.chdir(directory)

    num_channels = 3  # remove

    num_ims = sum([len(files) for (r, d, files) in os.walk(directory)])


    imgs = np.zeros([num_ims, imsz, imsz, num_channels])

    im_num = 0
    class_num = 0



    for filename in sorted(os.listdir(os.getcwd())):
        print(filename)
        im = Image.open(filename)
        im = np.array(im.resize((imsz, imsz)))


        imgs[im_num, :, :, :] = im



        im_num += 1
    os.chdir(directory)
    os.chdir(main_dir)
    return (imgs)


