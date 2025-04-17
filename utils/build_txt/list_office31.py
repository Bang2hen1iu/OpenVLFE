import os
import random

datasets = ['amazon']

known = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 
        'laptop_computer', 'monitor', 'mouse', 'mug','projector']

for subset in datasets:
    source_list = './txt/Office31_ablation/source_{}_obda_21.txt'.format(subset)
    target_list = './txt/Office31_ablation/target_{}_obda_21.txt'.format(subset)
    subfolder = './data/Office31/{}'.format(subset)
    
    classes = sorted(os.listdir(subfolder))

    unknown_cls = [i for i in classes if i not in known]

    s_file = open(source_list, 'w')
    t_file = open(target_list, 'w')

    for idx, k in enumerate(known):
        imgs = os.listdir(os.path.join(subfolder, k))
        imgs_path = [os.path.join(subfolder, k, img) for img in imgs]

        write_item = [p+' {}\n'.format(idx) for p in imgs_path]
        for w in write_item: 
            # s_file.write(w)
            t_file.write(w)

    for idx, k in enumerate(unknown_cls):
        imgs = os.listdir(os.path.join(subfolder, k))
        imgs_path = [os.path.join(subfolder, k, img) for img in imgs]

        write_item = [p+' {}\n'.format(len(known)) for p in imgs_path]
        for w in write_item: 
            t_file.write(w)
    t_file.close()


# for subset in datasets:
#     list = '../txt/Closeset/{}.txt'.format(subset)
#     subfolder = './data/OfficeHome/{}'.format(subset)
    
#     classes = sorted(os.listdir(subfolder))

#     file = open(list, 'w')

#     for idx, k in enumerate(classes):
#         imgs = os.listdir(os.path.join(subfolder, k))
#         imgs_path = [os.path.join(subfolder, k, img) for img in imgs]

#         write_item = [p+' {}\n'.format(idx) for p in imgs_path]
#         for w in write_item: 
#             file.write(w)
    
#     file.close()

