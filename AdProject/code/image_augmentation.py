from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from PIL import Image
import os

def read_rgb(filename, height=270, width=480):
    f = open(filename, "r")
    arr = np.fromfile(f, dtype=np.uint8)
    num_frames=int(arr.shape[0]/(height*width*3)) #9000
    arr = arr.reshape(num_frames, (height * width * 3))
    frames=[]
    for frame_number in np.arange(num_frames):
        r=arr[frame_number][0:(width*height)]
        g=arr[frame_number][(width*height):(2*width*height)]
        b=arr[frame_number][(2*width*height):]
        rgb = np.array(list(zip(r, g, b)))
        rgb_frame = rgb.reshape((height, width, 3))
        frames.append(rgb_frame)
    return np.array(frames)


#augment logos
#    mcdonalds='./dataset2/Brand Images/Mcdonalds_logo.raw'

def augment_img(input_filename, output_dir,num_augs=None,prefix=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if num_augs is None:
        num_augs=20
    if prefix is None:
        prefix=(input_filename.split('/')[-1]).split('.')[0]

    img_arr = read_rgb(filename)
    #k=img_arr[0][0][0][0]
    k=255
    datagen = ImageDataGenerator(
        rotation_range=45,
        zca_epsilon=10**-1,
        zca_whitening=True,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=40,
        zoom_range=0,
        channel_shift_range=.5,
        horizontal_flip=False,
        rescale=0,
        fill_mode='constant',cval=k,)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `data/augmented` directory
    i = 0
    for batch in datagen.flow(
        img_arr,
        batch_size=1,
        save_to_dir=output_dir,
        save_prefix=prefix,
        save_format='jpeg'):
        i += 1
        if i > num_augs:
            break  # otherwise the generator would loop indefinitely
    print('Success! Created {} augmentations of {}'.format(num_augs, prefix))
    return

logoDir=['./dataset/Brand Images/subway_logo.rgb',
        './dataset/Brand Images/starbucks_logo.rgb',
        './dataset2/Brand Images/nfl_logo.rgb',
        './dataset2/Brand Images/Mcdonalds_logo.raw']

for filename in logoDir:
    augment_img(filename, './dataset2/augmented/')
#blur
import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
image = misc.ascent()
w = signal.gaussian(50, 10.0)
image_new = signal.sepfir2d(image, w, w)