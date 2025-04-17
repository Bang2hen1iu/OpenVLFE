
import os
import clip
import math
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import pytorch_lightning as pl

# customerize self-defined module
from models.basenet import ResBase, Classifier
from models.get_model import get_base_model
from utils.loss import CrossEntropyLoss, entropy
from utils.lr_schedule import inv_lr_scheduler


class SourcePreTrainModule(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False # for complex optimization process (multi optimizer)
        self.args = args

        # customerize model architecture
        self.G, dim = get_base_model(self.args.network, self.args.num_class)
        self.C1 = Classifier(self.args.num_class, norm=False, input_size=dim)

        # customerize loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # customerize intermedia parameters
        self.known_acc = 0.0
        self.unknown_acc = 0.0
        self.h_score = 0.0
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        # customerize computational graph in single forward pass (if part of model defined in this module)
        z = self(x)
        return z

    def training_step(self, batch, batch_idx):
        # customerize behaviors in single batch forward 
        g_opt, c_opt = self.optimizers()
        
        self.C1.weight_norm()

        img_s, label_s = batch['src']
        feat_s = self.G(img_s)
        out_s = self.C1(feat_s)

        loss_s = self.cross_entropy_loss(out_s, label_s)

        loss = loss_s

        pred_s = out_s.data.max(1)[1]
        acc_s = (pred_s == label_s).float().mean()
        self.log_dict({'source loss': loss_s, 'train_acc': acc_s, },on_step=True, prog_bar=True)
        
    
        g_opt.zero_grad()
        c_opt.zero_grad()
        self.manual_backward(loss)
        g_opt.step()
        c_opt.step()
        
        
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()
    
        return loss
        
    def training_step_end(self, batch_parts):
        # only used when multi-gpu training, collect each return result and perform aggregation
        # collect log information | visualization | 
        pass

    def on_train_epoch_end(self):
        # input: a list of all return results of all training step
        # manually save model | intermedia result
        pass
    
    def validation_step(self, batch, batch_idx):
        # customerize behaviors in single batch forward 

        img_t, label_t = batch
        feat = self.G(img_t)

        out_t = self.C1(feat)
        prob_t = F.softmax(out_t, dim=-1)
        pred_t = out_t.data.max(1)[1]
        ent = entropy(prob_t, prob=False, mean=False)
        self.validation_step_outputs.append({'pred': pred_t, 'label': label_t, 'ent': ent})

    def on_validation_epoch_end(self):
        y_kno = torch.zeros(0).cuda(self.args.gpu[0])
        y_unk = torch.zeros(0).cuda(self.args.gpu[0])
        per_class_num = torch.zeros((self.args.num_class + 1))
        per_class_correct = torch.zeros((self.args.num_class + 1))
        class_list = [i for i in range(self.args.num_class + 1)]
        for out in self.validation_step_outputs:
            for i, t in enumerate(class_list):
                t_ind = torch.where(out['label'] == t)[0]
                correct_ind = torch.where(out['pred'][t_ind] == t)[0]
                per_class_correct[i] += float(len(correct_ind))
                per_class_num[i] += float(len(t_ind))
                if t < self.args.num_class:
                    y_kno = torch.cat([y_kno, out['ent'][t_ind]], dim=0)
                else:
                    y_unk = torch.cat([y_unk, out['ent'][t_ind]], dim=0)
        
        per_class_acc = per_class_correct / per_class_num
        known_acc = 100.0 * per_class_acc[:-1].mean()
        unknown_acc = 100.0 * per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)

        self.last_kno = known_acc
        self.last_unk = unknown_acc
        self.last_hos = h_score

        if max(self.h_score, h_score) == h_score:
            self.known_acc = known_acc
            self.unknown_acc = unknown_acc
            self.h_score = h_score
        self.log_dict(
            {'known': known_acc, 
             'unknown': unknown_acc, 
             'hos': h_score,
             },
             on_epoch=True)
        self.log_dict(
            {'best known': self.known_acc, 
             'best unknown': self.unknown_acc, 
             'best hos': self.h_score,
             },
             on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        # customerize behaviors in single batch forward 
        img_t, label_t, _ = batch
        feat = self.G(img_t)

        out_t = self.C1(feat)
        prob_t = F.softmax(out_t, dim=-1)
        pred_t = out_t.data.max(1)[1]
        ent = entropy(prob_t, prob=False, mean=False)
        self.test_step_outputs.append({'pred': pred_t, 'label': label_t, 'ent': ent})

    def on_test_epoch_end(self):
        labels = torch.zeros(0).cuda(self.args.gpu[0])
        preds = torch.zeros(0).cuda(self.args.gpu[0])
        y_kno = torch.zeros(0).cuda(self.args.gpu[0])
        y_unk = torch.zeros(0).cuda(self.args.gpu[0])
        per_class_num = torch.zeros((self.args.num_class + 1))
        per_class_correct = torch.zeros((self.args.num_class + 1))
        class_list = [i for i in range(self.args.num_class + 1)]
        for out in self.test_step_outputs:
            for i, t in enumerate(class_list):
                t_ind = torch.where(out['label'] == t)[0]
                correct_ind = torch.where(out['pred'][t_ind] == t)[0]
                per_class_correct[i] += float(len(correct_ind))
                per_class_num[i] += float(len(t_ind))
                if t < self.args.num_class:
                    y_kno = torch.cat([y_kno, out['ent'][t_ind]], dim=0)
                else:
                    y_unk = torch.cat([y_unk, out['ent'][t_ind]], dim=0)
                preds = torch.cat([preds, out['pred'][t_ind]], dim=0)
                labels = torch.cat([labels, out['label'][t_ind]], dim=0)
        
        per_class_acc = per_class_correct / per_class_num
        known_acc = 100.0 * per_class_acc[:-1].mean()
        unknown_acc = 100.0 * per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)

        self.last_kno = known_acc
        self.last_unk = unknown_acc
        self.last_hos = h_score

        if max(self.h_score, h_score) == h_score:
            self.known_acc = known_acc
            self.unknown_acc = unknown_acc
            self.h_score = h_score
        self.log_dict(
            {'known': known_acc, 
             'unknown': unknown_acc, 
             'hos': h_score,
             },
             on_epoch=True)
        self.log_dict(
            {'best known': self.known_acc, 
             'best unknown': self.unknown_acc, 
             'best hos': self.h_score,
             },
             on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # customerize optimizers
        
        params = []
        for key, value in dict(self.G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': self.args.lr_f,
                            'weight_decay': self.args.weight_decay}]
            else:
                params += [{'params': [value], 'lr': self.args.lr_f,
                            'weight_decay': self.args.weight_decay}]
        optimizer_g = optim.SGD(params, momentum=self.args.sgd_momentum, nesterov=True)
        optimizer_c = optim.SGD(
            list(self.C1.parameters()), lr=self.args.lr_n, 
            weight_decay=self.args.weight_decay, momentum=self.args.sgd_momentum, nesterov=True)

        scheLR_g = optim.lr_scheduler.LambdaLR(optimizer_g, lambda x:inv_lr_scheduler(x, self.args.epoch))
        scheLR_c = optim.lr_scheduler.LambdaLR(optimizer_c, lambda x:inv_lr_scheduler(x, self.args.epoch))

        return (
        {"optimizer": optimizer_g, "lr_scheduler": scheLR_g,},
        {"optimizer": optimizer_c, "lr_scheduler": scheLR_c},
        )

        
class VLEFTrainModule(pl.LightningModule): 
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False # for complex optimization process (multi optimizer)
        self.args = args
        self.build_text(args.dataset)
        self.init_clip_model()

        # customerize model architecture
        self.G, dim = get_base_model(self.args.network, self.args.num_class)
        self.C1 = Classifier(self.args.num_class, norm=False, input_size=dim)
        self.temp_G = copy.deepcopy(self.G)
        self.temp_C = copy.deepcopy(self.C1)

        # customerize loss function
        self.cross_entropy_loss = CrossEntropyLoss(self.args.num_class)

        # customerize intermedia parameters
        self.known_acc = 0.0
        self.unknown_acc = 0.0
        self.h_score = 0.0
        self.last_kno = 0.0
        self.last_unk = 0.0
        self.last_hos = 0.0

        # customerize hyper parameters
        self.alpha = 1.0

        if self.args.dataset == 'OfficeHome':
            self.min_epoch = 15
        elif self.args.dataset == 'Office31':
            self.min_epoch = 10
        elif self.args.dataset == 'VisDA':
            self.min_epoch = 5

        self.endure = 0.2
        self.ent_thres = 0.2
        self.start_thres = 0.0
        self.eval_thres = math.log(args.num_class)/2
        self.lower_bound = self.ent_thres - self.endure
        self.upper_bound = self.ent_thres + self.endure
        self.clip_thres = self.args.clip_thres
        
        self.validation_step_outputs=[]


    def forward(self, x):
        # customerize computational graph in single forward pass (if part of model defined in this module)
        z = self(x)
        return z

    def training_step(self, batch, batch_idx):
        # customerize behaviors in single batch forward 
        loss = 0
        
        img_s, label_s = batch['src']
        img_t, label_t = batch['tgt']
        feat_s = self.G(img_s[:,:3,:,:])
        out_s = self.C1(feat_s)

        loss_s = self.cross_entropy_loss(out_s, label_s)
        loss += loss_s

        
        # if self.current_epoch+1 < self.min_epoch+5:
        feat_t = self.G(img_t[:,:3,:,:])
        out_t = self.C1(feat_t)
        ent = entropy(out_t, prob=True, mean=False)
        
        with torch.no_grad():
            feat_clip_t = self.clip_model.encode_image(img_t[:,3:,:,:])
            tgt_similarity = self.get_cosine_similarity(feat_clip_t, self.text_features).detach()
            tgt_entropy = entropy(tgt_similarity, prob=True, mean=False)
        
            clip_kno_idx = torch.where(tgt_entropy < self.clip_thres)[0]
            clip_pseudo_label = tgt_similarity.softmax(dim=-1).max(-1)[1]
            pseudo_kno_idx = torch.where(ent < self.lower_bound)[0]
            pseudo_unk_idx = torch.where(ent > self.upper_bound)[0]

        if len(clip_kno_idx) > 0:
            loss_t = self.cross_entropy_loss(out_t[clip_kno_idx], clip_pseudo_label[clip_kno_idx])
        else: 
            loss_t = 0 
        if len(pseudo_kno_idx) > 0:
            loss_reg_kno = entropy(out_t[pseudo_kno_idx])
        else: 
            loss_reg_kno = 0 

        if len(pseudo_unk_idx) > 0:
            loss_reg_unk = - entropy(out_t[pseudo_unk_idx])
        else: loss_reg_unk = 0 
    
        if self.current_epoch+1 < self.min_epoch:
            loss += loss_t
        elif self.current_epoch+1 >= self.min_epoch:
            loss += self.args.w_ent * loss_t
            loss += self.args.w_ent * (loss_reg_kno + loss_reg_unk)

        # if batch_idx % 10 == 0:
        #     print('source loss: {:.6f}, target loss: {:.6f}, regularization: {:.6f},{:.6f}'.format(float(loss_s),float(loss_t),float(loss_reg_kno),float(loss_reg_unk)))
        # print("=============== training time: {} ==================".format(end - start))
        g_opt, c_opt = self.optimizers()
        g_opt.zero_grad()
        c_opt.zero_grad()
        self.manual_backward(loss)
        g_opt.step()
        c_opt.step()
        
        
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()
        
        return loss
        
    def training_step_end(self, batch_parts):
        # only used when multi-gpu training, collect each return result and perform aggregation
        # collect log information | visualization |
        pass

    def on_train_epoch_end(self):
        # input: a list of all return results of all training step
        # manually save model | intermedia result
        if self.current_epoch+1 >= self.min_epoch:
            self.ent_thres = self.start_thres + (self.eval_thres - self.start_thres) * \
                            (self.current_epoch + 1 - self.min_epoch)/(self.args.epoch - self.min_epoch) # linear releasing of hard threshold
            self.lower_bound = self.ent_thres - self.endure if (self.ent_thres - self.endure) > 0 else 0.0
            self.upper_bound = self.ent_thres + self.endure

    
    def validation_step(self, batch, batch_idx):
        # customerize behaviors in single batch forward 
        img_t, label_t = batch
        feat = self.G(img_t)

        out_t = self.C1(feat)
        prob_t = F.softmax(out_t, dim=-1)
        pred_t = out_t.data.max(1)[1]

        ent = entropy(prob_t, prob=False, mean=False)
        ind_unk = torch.where(ent > self.eval_thres)[0] # use entropy of classifier output as unknown detector
        pred_t[ind_unk] = self.args.num_class
        self.validation_step_outputs.append({'feat':feat, 'pred': pred_t, 'label': label_t, 'ent': ent})

    def on_validation_epoch_end(self): 
        feats = torch.zeros(0).cuda(self.args.gpu[0])
        ents = torch.zeros(0).cuda(self.args.gpu[0])
        labels = torch.zeros(0).cuda(self.args.gpu[0])

        per_class_num = torch.zeros((self.args.num_class + 1))
        per_class_correct = torch.zeros((self.args.num_class + 1))
        class_list = [i for i in range(self.args.num_class + 1)]

        for out in self.validation_step_outputs:
            for i, t in enumerate(class_list):
                t_ind = torch.where(out['label'] == t)[0]
                correct_ind = torch.where(out['pred'][t_ind] == t)[0]
                per_class_correct[i] += float(len(correct_ind))
                per_class_num[i] += float(len(t_ind))

                feats = torch.cat([feats, out['feat'][t_ind]], dim=0)
                ents = torch.cat([ents, out['ent'][t_ind]], dim=0)
                labels = torch.cat([labels, out['label'][t_ind]], dim=0)

        per_class_acc = per_class_correct / per_class_num
        known_acc = 100.0 * per_class_acc[:-1].mean()
        unknown_acc = 100.0 * per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
        print(per_class_acc.mean()*100, known_acc, unknown_acc, h_score)
        self.last_kno = known_acc
        self.last_unk = unknown_acc
        self.last_hos = h_score
        

        if max(self.h_score, h_score) == h_score:
            self.known_acc = known_acc
            self.unknown_acc = unknown_acc
            self.h_score = h_score
            
        self.log_dict(
            {'known': known_acc, 
             'unknown': unknown_acc, 
             'hos': h_score,
             },
             on_epoch=True)
        self.log_dict(
            {'best known': self.known_acc, 
             'best unknown': self.unknown_acc, 
             'best hos': self.h_score,
             },
             on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
             
    def configure_optimizers(self):
        # customerize optimizers
        
        params = []
        for key, value in dict(self.G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': self.args.lr_f,
                            'weight_decay': self.args.weight_decay}]
            else:
                params += [{'params': [value], 'lr': self.args.lr_f,
                            'weight_decay': self.args.weight_decay}]
        optimizer_g = optim.SGD(params, momentum=self.args.sgd_momentum, nesterov=True)
        optimizer_c = optim.SGD(
            list(self.C1.parameters()), lr=self.args.lr_n, 
            weight_decay=self.args.weight_decay, momentum=self.args.sgd_momentum, nesterov=True)

        scheLR_g = optim.lr_scheduler.LambdaLR(optimizer_g, lambda x:inv_lr_scheduler(x, self.args.epoch))
        scheLR_c = optim.lr_scheduler.LambdaLR(optimizer_c, lambda x:inv_lr_scheduler(x, self.args.epoch))

        return (
        {"optimizer": optimizer_g, "lr_scheduler": scheLR_g,},
        {"optimizer": optimizer_c, "lr_scheduler": scheLR_c},
        )

    def build_text(self, datasets):
        if datasets == 'Office31':
            text = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug','projector'] # 10 
        elif datasets == 'OfficeHome':
            text = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
                        'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
                        'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork'] # 25
        elif datasets == 'VisDA':
            text = ['bicycle', 'bus', 'car', 'motorcycle', 'train', 'truck']

        self.text = ['a photo of ' + word for word in text] # with "the" as the prefix
    
    def init_clip_model(self):
        self.clip_model, _ = clip.load("ViT-B/32") # pretrain clip model
        self.clip_model = self.clip_model.cuda(self.args.gpu[0])
        text_token = clip.tokenize(self.text).cuda(self.args.gpu[0])
        self.text_features = self.clip_model.encode_text(text_token) # freeze text feature for better computational cost
        self.prior = torch.zeros(self.args.num_class).cuda(self.args.gpu[0])
        self.temp_prior = torch.zeros(self.args.num_class).cuda(self.args.gpu[0])

    def get_cosine_similarity(self, image_feat, text_feat): # from clip forward part
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_feat @ text_feat.t()

        return logits_per_image

    def get_hist_figure(self, y_kno, y_unk, range=(0,1), bins=40, fig_name=''):
        src_set = self.args.source.split('/')[-1].split('_')[1]
        tgt_set = self.args.target.split('/')[-1].split('_')[1]
        plt.figure(fig_name)
        plt.grid(True)
        plt.hist(y_kno, bins=bins, alpha=0.5, label='a', range=range)
        plt.hist(y_unk, bins=bins, alpha=0.5, label='b', range=range)
        plt.axvline(x=self.ent_thres,ls="-",c="green")
        plt.title('entropy hist map: {}-{}\nkno:{}, unk:{}, hos:{}'.format(src_set, tgt_set, self.last_kno, self.last_unk, self.last_hos))
        plt.savefig(os.path.join(self.args.curdir, '{}2{}-{}-{}.png'.format(src_set, tgt_set, fig_name, self.current_epoch+1)))
        plt.cla()

class VLEFTestModule(pl.LightningModule): # add clip as supervised, unknown entmax with dynamic threshold, start from source initialized network
    def __init__(self, args) -> None:
        super().__init__()
        # self.automatic_optimization = False # for complex optimization process (multi optimizer)
        self.args = args
        self.build_text(args.dataset)
        self.init_clip_model()

        # customerize model architecture
        self.G, dim = get_base_model(self.args.network, self.args.num_class)
        self.C1 = Classifier(self.args.num_class, norm=False, input_size=dim)
        self.temp_G = copy.deepcopy(self.G)
        self.temp_C = copy.deepcopy(self.C1)

        # customerize loss function
        self.cross_entropy_loss = CrossEntropyLoss(self.args.num_class)
        # self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.Contrast_loss = DomainContrastiveLoss(self.args.gpu[0])
        # self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        # customerize intermedia parameters
        self.known_acc = 0.0
        self.unknown_acc = 0.0
        self.h_score = 0.0
        self.last_kno = 0.0
        self.last_unk = 0.0
        self.last_hos = 0.0

        # customerize hyper parameters
        self.alpha = 1.0

        if self.args.dataset == 'OfficeHome':
            self.min_epoch = 15
            # self.min_epoch = 2
        elif self.args.dataset == 'Office31':
            self.min_epoch = 15
        elif self.args.dataset == 'ImageCLEF':
            self.min_epoch = 15
        elif self.args.dataset == 'VisDA':
            self.min_epoch = 5
        self.endure = 0.2
        self.ent_thres = 0.2
        self.start_thres = 0.2
        self.eval_thres = math.log(args.num_class)/2
        self.lower_bound = self.ent_thres - self.endure
        self.upper_bound = self.ent_thres + self.endure
        self.clip_thres = self.args.clip_thres
        


    def forward(self, x):
        # customerize computational graph in single forward pass (if part of model defined in this module)
        z = self(x)
        return z

    def training_step(self, batch, batch_idx, optimizer_idx):
        # customerize behaviors in single batch forward 
        # print(self.global_step)
        loss = torch.Tensor([0])
        
        return loss
        
    def training_step_end(self, batch_parts):
        # only used when multi-gpu training, collect each return result and perform aggregation
        # collect log information | visualization |
        # self.alpha = (1 + 100 * min(1.0, (epoch_num / max_epoch)*())) ** (-power)
        pass

    def training_epoch_end(self, training_step_outputs):
        # input: a list of all return results of all training step
        # manually save model | intermedia result
        # self.step_flag = 0
        # self.alpha = 1.0 - ((self.current_epoch + 1)/self.args.epoch)
        # copy the last network as a baseline estimator for delta entropy calculation
        pass
    
    def validation_step(self, batch, batch_idx):
        # customerize behaviors in single batch forward 
        img_t, label_t = batch
        feat = self.G(img_t)

        out_t = self.C1(feat)
        prob_t = F.softmax(out_t, dim=-1)
        pred_t = out_t.data.max(1)[1]

        # pre_out_t = self.temp_C(self.temp_G(img_t))
        # pre_ent = entropy(pre_out_t, prob=True, mean=False)
        ent = entropy(prob_t, prob=False, mean=False)
        # delta_ent = ent - pre_ent
        # delta_ent[torch.where(ent <= self.eval_thres)[0]] = -3
        # print(ent)
        ind_unk = torch.where(ent > self.eval_thres)[0] # use entropy of classifier output as unknown detector
        pred_t[ind_unk] = self.args.num_class

        return {'feat':feat, 'pred': pred_t, 'label': label_t, 'ent': ent} #, 'delta': delta_ent}

    def validation_epoch_end(self, validation_step_outputs): 
        feat = torch.zeros(0)
        pre_label = torch.zeros(0)
        # y_kno = torch.zeros(0).cuda(self.args.gpu[0])
        # y_unk = torch.zeros(0).cuda(self.args.gpu[0])
        # y_kno_delta = torch.zeros(0).cuda(self.args.gpu[0])
        # y_unk_delta = torch.zeros(0).cuda(self.args.gpu[0])
        per_class_num = torch.zeros((self.args.num_class + 1))
        per_class_correct = torch.zeros((self.args.num_class + 1))
        class_list = [i for i in range(self.args.num_class + 1)]
        for out in validation_step_outputs:
            feat = torch.cat([feat, out['feat']], dim=0)
            pre_label = torch.cat([pre_label, out['pred']], dim=0)
            for i, t in enumerate(class_list):
                t_ind = torch.where(out['label'] == t)[0]
                correct_ind = torch.where(out['pred'][t_ind] == t)[0]
                per_class_correct[i] += float(len(correct_ind))
                per_class_num[i] += float(len(t_ind))
                
                # if t < self.args.num_class:
                #     y_kno = torch.cat([y_kno, out['ent'][t_ind]], dim=0)
                #     # y_kno_delta = torch.cat([y_kno_delta, out['delta'][t_ind]], dim=0)
                # else:
                #     y_unk = torch.cat([y_unk, out['ent'][t_ind]], dim=0)
                #     # y_unk_delta = torch.cat([y_unk_delta, out['delta'][t_ind]], dim=0)
        
        torch.save(feat, 'A-D-features.pkt')
        torch.save(pre_label, 'A-D-pred.pkt')
        per_class_acc = per_class_correct / per_class_num
        known_acc = 100.0 * per_class_acc[:-1].mean()
        unknown_acc = 100.0 * per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
        print(known_acc, unknown_acc, h_score)
        self.last_kno = known_acc
        self.last_unk = unknown_acc
        self.last_hos = h_score
        

        if max(self.h_score, h_score) == h_score:
            self.known_acc = known_acc
            self.unknown_acc = unknown_acc
            self.h_score = h_score
        # self.get_hist_figure(y_kno.data.cpu().numpy(), y_unk.data.cpu().numpy(), range=(0, np.log(self.args.num_class)), fig_name='entropy')
        # self.get_hist_figure(y_kno_delta.data.cpu().numpy(), y_unk_delta.data.cpu().numpy(), range=(-1, 1), fig_name='delta')
        self.log_dict(
            {'known': known_acc, 
             'unknown': unknown_acc, 
             'hos': h_score,
             },
             on_epoch=True)
        self.log_dict(
            {'best known': self.known_acc, 
             'best unknown': self.unknown_acc, 
             'best hos': self.h_score,
             },
             on_epoch=True, prog_bar=True)
             
    def configure_optimizers(self):
        # customerize optimizers
        
        params = []
        for key, value in dict(self.G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': self.args.lr_f,
                            'weight_decay': self.args.weight_decay}]
            else:
                params += [{'params': [value], 'lr': self.args.lr_f,
                            'weight_decay': self.args.weight_decay}]
        optimizer_g = optim.SGD(params, momentum=self.args.sgd_momentum, nesterov=True)
        optimizer_c = optim.SGD(
            list(self.C1.parameters()), lr=self.args.lr_n, 
            weight_decay=self.args.weight_decay, momentum=self.args.sgd_momentum, nesterov=True)

        scheLR_g = optim.lr_scheduler.LambdaLR(optimizer_g, lambda x:inv_lr_scheduler(x, self.args.epoch))
        scheLR_c = optim.lr_scheduler.LambdaLR(optimizer_c, lambda x:inv_lr_scheduler(x, self.args.epoch))

        return [optimizer_g, optimizer_c], [scheLR_g, scheLR_c]

    def build_text(self, datasets):
        if datasets == 'Office31':
            text = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug','projector']
        elif datasets == 'OfficeHome':
            text = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
                        'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
                        'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork'] 
        elif datasets == 'VisDA':
            text = ['bicycle', 'bus', 'car', 'motorcycle', 'train', 'truck']
        elif datasets == 'ImageCLEF':
            text = ['airplanes', 'car side', 'computer monitor', 'dog', 'horse', 'hummingbird']

        self.text = ['a photo of ' + word for word in text] # with "the" as the prefix
    
    def init_clip_model(self):
        self.clip_model, _ = clip.load("ViT-B/32") # pretrain clip model
        self.clip_model = self.clip_model.cuda(self.args.gpu[0])
        text_token = clip.tokenize(self.text).cuda(self.args.gpu[0])
        self.text_features = self.clip_model.encode_text(text_token) # freeze text feature for better computational cost
        self.prior = torch.zeros(self.args.num_class).cuda(self.args.gpu[0])
        self.temp_prior = torch.zeros(self.args.num_class).cuda(self.args.gpu[0])

    def get_cosine_similarity(self, image_feat, text_feat): # from clip forward part
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_feat @ text_feat.t()
        # logits_per_text = logits_per_image.t()

        return logits_per_image

    def get_hist_figure(self, y_kno, y_unk, range=(0,1), bins=40, fig_name=''):
        src_set = self.args.source.split('/')[-1].split('_')[1]
        tgt_set = self.args.target.split('/')[-1].split('_')[1]
        plt.figure(fig_name)
        plt.grid(True)
        plt.hist(y_kno, bins=bins, alpha=0.5, label='a', range=range)
        plt.hist(y_unk, bins=bins, alpha=0.5, label='b', range=range)
        plt.axvline(x=self.ent_thres,ls="-",c="green")
        plt.title('entropy hist map: {}-{}\nkno:{}, unk:{}, hos:{}'.format(src_set, tgt_set, self.last_kno, self.last_unk, self.last_hos))
        plt.savefig(os.path.join(self.args.curdir, '{}2{}-{}-{}.png'.format(src_set, tgt_set, fig_name, self.current_epoch+1)))
        plt.cla()
