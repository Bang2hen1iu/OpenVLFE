import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import pytorch_lightning as pl


from trainers import Trainer
from data_loader.get_loader import TransferDataModule, TransferCLIPDataModule

parser = argparse.ArgumentParser(description='Pytorch Openset-VLEF',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# base settings
parser.add_argument('--expname', type=str, default='VLEF', help='')
parser.add_argument('--dataset', type=str, default='Office31', help='dataset for Office31|OfficeHome|VisDA')
parser.add_argument('--source', type=str, default='./txt/Office31/source_amazon_obda.txt', help='')
parser.add_argument('--target', type=str, default='./txt/Office31/target_dslr_obda.txt', help='')
parser.add_argument('--savedir', type=str, default= './record', help='path to save everything')
parser.add_argument('--log_interval', type=float, default=0.25, help='proportion of an epoch for validation step')
parser.add_argument('--gpu', type= int, nargs='+', help='manually defined gpu devices')
parser.add_argument('--seed', type=int, default=0, help='manuall defined random seed')

######=============== configurations ==============######
# 1. network paramters
parser.add_argument('--network', type=str, default='resnet50', help='backbone network type (VGG|resnetXX)')
parser.add_argument('--num_class', type=int, default=10, help='number of classes for classification task')
parser.add_argument('--hidden_dim', type=int, default=512, help='dimension of hidden embedding layer, only use when --use_btnk set true')

parser.add_argument('--temp', type=float, default=0.05, help='temprature factor')
parser.add_argument('--top', default=False, action='store_true', help='use top pretrainied classifier or not')
parser.add_argument('--norm', default=False, action='store_true', help='feature normalized before classification or not')
parser.add_argument('--use_btnk', default=False, action='store_true', help='use addtional feature embedding layer before classfication or not')

# 2. dataloader parameters
parser.add_argument('--batch_size', type=int, default=36, help='mini-batch size')
parser.add_argument('--num_workers', type=int, default=3, help='number of multi-thread workers')
parser.add_argument('--is_train', default=True, action='store_true', help='')

# 3. training parameters
parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
parser.add_argument('--min_epoch', type=int, default=10, help='number of warm up epochs for training')
parser.add_argument('--lr_f', type=float, default=0.1, help='learning rate for finetuning pretrained layers')
parser.add_argument('--lr_n', type=float, default=1, help='learning rate for traning new initial layer')
parser.add_argument('--ent_thres', type=float, default=0.5, help='separated threshold for known/unknown')
parser.add_argument('--clip_thres', type=float, default=0.5, help='separated threshold for known/unknown')

parser.add_argument('--w_mi', type=float, default=0.1, help='initial weight of mutual entropy minimization loss')
parser.add_argument('--w_ent', type=float, default=0.1, help='initial weight of entropy minimization loss')
parser.add_argument('--w_ctst', type=float, default=0.1, help='initial weight of text|image contrastive loss')
parser.add_argument('--w_consist', type=float, default=0.1, help='initial weight of source|target to clip image space consistency loss')
parser.add_argument('--consist_type', type=str, default='', help='raw|matrix loss')
parser.add_argument('--w_adv', type=float, default=0.1, help='initial weight of adversarial loss')
parser.add_argument('--ls_factor', type=float, default=0.05, help='label smoothing factor')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay of optimizer')
parser.add_argument('--sgd_momentum', type=float, default=0.9, help='momentum for SGD optimizer')
parser.add_argument('--pretrained', default=False, action='store_true', help='')
parser.add_argument('--no_adapt', default=False, action='store_true', help='')
parser.add_argument('--flag', default=False, action='store_true', help='')


args = parser.parse_args()
pl.seed_everything(args.seed)

# folder for a single complete experiment
expname = args.expname
subname = args.source.split('/')[-1].split('.')[0] + '2' + args.target.split('/')[-1].split('.')[0]

cur_dir = '/'.join([args.savedir, expname, subname])
pretrained_folder = '/'.join([args.savedir, 'pretrained_source_model'])
args.curdir = cur_dir
if not os.path.exists(cur_dir):
    os.makedirs(cur_dir)
record_log = os.path.join(args.savedir, expname, subname, 'log.txt')
global_record_log = os.path.join(args.savedir, expname, 'log.txt')

if args.pretrained:
    dm = TransferDataModule(args.source, args.target, args.target, args)
else:
    dm = TransferCLIPDataModule(args.source, args.target, args.target, args)

dm.setup('fit')

load_path = pretrained_folder
G_checkpoint = '/'.join([load_path, '{}_G.pkt'.format(args.source.split('/')[-1].split('_')[1])])
C_checkpoint = '/'.join([load_path, '{}_C.pkt'.format(args.source.split('/')[-1].split('_')[1])])
    
if args.pretrained:
    model = Trainer.SourcePreTrainModule(args)
else:
    G_ckp = torch.load(G_checkpoint, map_location=None)
    C_ckp = torch.load(C_checkpoint, map_location=None)

    model = Trainer.VLEFTrainModule(args)
    G_dict = model.G.state_dict()
    C_dict = model.C1.state_dict()
    model.G.load_state_dict(G_ckp)
    model.C1.load_state_dict(C_ckp)


trainer = pl.Trainer(
    default_root_dir=cur_dir, # root path for auto model parameter save
    precision=32,
    devices=1, # could be more for parallel training
    max_epochs=args.epoch, 
    enable_progress_bar=False,
    fast_dev_run=False, # set true for fast running of 1 train/val/test process and end (unit test)
    log_every_n_steps=30
    )
trainer.fit(model, datamodule=dm)
with open(record_log, 'w') as f:
    f.write(str(args)+'\n')
    f.write('best record ------ kno {:.4f} | unk {:.4f} | hos {:.4f} \n'.format(model.known_acc, model.unknown_acc, model.h_score))
    f.write('last record ------ kno {:.4f} | unk {:.4f} | hos {:.4f} \n\n'.format(model.last_kno, model.last_unk, model.last_hos))
with open(global_record_log, 'a') as f:
    f.write('================= ' + subname + ' ==================\n')
    f.write('best record ------ kno {:.4f} | unk {:.4f} | hos {:.4f} \n'.format(model.known_acc, model.unknown_acc, model.h_score))
    f.write('last record ------ kno {:.4f} | unk {:.4f} | hos {:.4f} \n\n'.format(model.last_kno, model.last_unk, model.last_hos))

if args.pretrained:
    if not os.path.exists(pretrained_folder):
        os.makedirs(pretrained_folder)
    torch.save(model.G.state_dict(), G_checkpoint)
    torch.save(model.C1.state_dict(), C_checkpoint)

        