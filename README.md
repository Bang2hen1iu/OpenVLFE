# Open-VLFE (Mixed Domain Adaptation via Visual-Linguistic Focal Evolving)
This code provides the evaluation phase for the paper.

## Environment
Python 3.9, Pytorch 1.8.1, Torch Vision 0.9.1, Pytorch-lightning 1.5.9. We used the pytorch-lightning library for memory efficient float16 training.

## Data Preparation

#### Datasets

[Office Dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/),
[OfficeHome Dataset](http://hemanthdv.org/OfficeHome-Dataset/), 
[VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

Prepare dataset in data directory.
```
./data/Office31/amazon/images/ ## Office31
./data/OfficeHome/Real ## OfficeHome
./data/VisDA/Classification/train ## VisDA synthetic images
./data/VisDA/Classification/validation ## VisDA real images
```

#### File split

The file list for the datasets above are provided in ./txt folder, part of the split follows [OVANet](https://github.com/VisionLearningGroup/OVANet), e.g.

```
./txt/Office31/source_amazon_obda.txt ## Office
./txt/OfficeHome/source_Real_obda.txt ## OfficeHome
```

## Evaluation

[checkpoints](https://drive.google.com/drive/folders/1LCgp0oTx028X2QTSlSGs6P-HvLTCzqfJ?usp=sharing)

Download checkpoints file and place in ./checkpoints . To evaluate the performance:
```
python eval.py --dataset 'Office31' --source './txt/Office31/source_dslr_obda.txt' --target './txt/Office31/target_amazon_obda.txt' --network 'resnet50' --num_class 10 --batch_size 64 --num_workers 4 --gpu 0
python eval.py --dataset 'OfficeHome' --source './txt/OfficeHome/source_Art_obda.txt' --target './txt/OfficeHome/target_Clipart_obda.txt' --network 'resnet50' --num_class 25 --batch_size 64 --num_workers 4 --gpu 0
```

