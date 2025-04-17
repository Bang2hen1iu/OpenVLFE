import os
import random

datasets = ['train', 'validation']

known_list = ['bicycle', 'bus', 'car', 'motorcycle', 'train', 'truck']
unknown_list = ['aeroplane', 'horse', 'knife', 'person', 'plant', 'skateboard']

source_list = '../txt/source_visda_obda.txt'
target_list = '../txt/target_visda_obda.txt'
subfolder_source = '../data/VisDA/Classification/train'
subfolder_target = '../data/VisDA/Classification/validation'

s_file = open(source_list, 'w')
t_file = open(target_list, 'w')

for idx, k in enumerate(known_list):
    s_imgs = os.listdir(os.path.join(subfolder_source, k))
    imgs_path = [os.path.join(subfolder_source, k, img) for img in s_imgs]
    write_item = [p+' {}\n'.format(idx) for p in imgs_path]
    for w in write_item: 
        s_file.write(w)

    t_imgs = os.listdir(os.path.join(subfolder_target, k))
    imgs_path = [os.path.join(subfolder_target, k, img) for img in t_imgs]
    write_item = [p+' {}\n'.format(idx) for p in imgs_path]
    for w in write_item: 
        t_file.write(w)

for idx, k in enumerate(unknown_list):
    imgs = os.listdir(os.path.join(subfolder_target, k))
    imgs_path = [os.path.join(subfolder_target, k, img) for img in imgs]

    write_item = [p+' {}\n'.format(len(known_list)) for p in imgs_path]
    for w in write_item: 
        t_file.write(w)

s_file.close()
t_file.close()

