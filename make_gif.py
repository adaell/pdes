"""
Uses the images in the images folder to create an animated gif

Requires imageio

"""

import imageio
import os

images = []
image_folder=os.path.join(os.getcwd(),'images')
savepath=os.path.join(os.getcwd(),'animation.gif')
images = []

filelist=os.listdir(image_folder)
filelist.sort(key=lambda x: os.path.getmtime(os.path.join(image_folder, x)))

for file_name in filelist:
    if file_name.endswith('.png'):
        file_path = os.path.join(image_folder, file_name)
        images.append(imageio.imread(file_path))


# for file_name in os.listdir(image_folder):
#     if file_name.endswith('.png'):
#         file_path = os.path.join(image_folder, file_name)
#         images.append(imageio.imread(file_path))

imageio.mimsave(savepath, images)