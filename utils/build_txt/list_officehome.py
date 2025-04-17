import os
import random

datasets = ['Art', 'Clipart', 'Product', 'Real']

# known_num = 25

# for subset in datasets:
#     source_list = './txt/office31/source_{}.txt'.format(subset)
#     target_list = './txt/office31/target_{}.txt'.format(subset)
#     subfolder = './data/Office31/{}'.format(subset)
    
#     classes = sorted(os.listdir(subfolder))

#     known_cls = classes[:known_num]
#     known_pri_cls = classes[known_num:20]
#     unknown_cls = classes[20:]

#     s_file = open(source_list, 'w')
#     t_file = open(target_list, 'w')

#     for idx, k in enumerate(known_cls):
#         imgs = os.listdir(os.path.join(subfolder, k))
#         imgs_path = [os.path.join(subfolder, k, img) for img in imgs]

#         write_item = [p+' {}\n'.format(idx) for p in imgs_path]
#         for w in write_item: 
#             s_file.write(w)
#             t_file.write(w)

#     for idx, k in enumerate(unknown_cls):
#         imgs = os.listdir(os.path.join(subfolder, k))
#         imgs_path = [os.path.join(subfolder, k, img) for img in imgs]

#         write_item = [p+' {}\n'.format(known_num) for p in imgs_path]
#         for w in write_item: 
#             t_file.write(w)
#     t_file.close()


for subset in datasets:
    list = '../txt/Closeset/{}.txt'.format(subset)
    subfolder = './data/OfficeHome/{}'.format(subset)
    
    classes = sorted(os.listdir(subfolder))

    file = open(list, 'w')

    for idx, k in enumerate(classes):
        imgs = os.listdir(os.path.join(subfolder, k))
        imgs_path = [os.path.join(subfolder, k, img) for img in imgs]

        write_item = [p+' {}\n'.format(idx) for p in imgs_path]
        for w in write_item: 
            file.write(w)
    
    file.close()

