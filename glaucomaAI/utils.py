import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image



def plot(x):
    # Plots an image
    fig, ax = plt.subplots()
    im = ax.imshow(x)
    ax.axis('off')
    fig.set_size_inches(8, 8)
    plt.show()
    
    

def read_ims(directory, imsz):
    # Reads images from a given directory, and returns them as a numpy array

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


