import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers import (
    get_scheduler
)
from sklearn.utils.class_weight import compute_class_weight
from accelerate import Accelerator
import argparse
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from lightning_fabric.utilities.seed import seed_everything
from model import MultiTaskModel
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from dataloader import MultiTaskDataCollator, MultiTaskDataset
from transformers import Wav2Vec2FeatureExtractor
from optims import LinearWarmupCosineLRScheduler

def get_loaders(args, train_path, valid_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/lqf/workspace/wavlm-multi/wavlm-large", return_attention_mask=True)
    data_collator = MultiTaskDataCollator(feature_extractor=feature_extractor, padding=True, max_length=args.max_length, max_length_labels=32)

    train_dataset = MultiTaskDataset(train_path)
    class_weight = train_dataset.class_weight_k()
    valid_dataset = MultiTaskDataset(valid_path)
    # test_dataset = MERDataset(test_path)
    sampler = WeightedRandomSampler(weights=class_weight, num_samples=train_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, sampler=sampler, collate_fn=data_collator, num_workers=10)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator, num_workers=10)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)

    return train_dataloader, valid_dataloader, class_weight

def unweightedacc(y_true, y_pred):
    ua = 0.0
    cm = confusion_matrix(y_true, y_pred)

    for i in range(len(cm)):
        tmp = cm[i]
        ua += (tmp[i] / np.sum(tmp))
    return (ua / len(cm))

def train_model(accelerator, model, cls_loss, dataloader, lr_scheduler=None, optimizer=None, epoch=0):

    batch_losses = 0.0
    batch_losses_avc = 0.0
    batch_losses_itm = 0.0
    batch_losses_cls = 0.0
    emo_probs, emo_labs = [], []

    assert optimizer!=None
    
    model.train()

    for i, data in enumerate(dataloader):
        ## analyze dataloader
        input_values, attention_mask, vinput_values, vattention_mask, emo_labels = data["input_values"], data["attention_mask"], data["vinput_values"], data["vattention_mask"], data['emo_labels']
        
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            cls_output, loss_avc, loss_itm = model(input_values=input_values, attention_mask=attention_mask, vinput_values=vinput_values, vattention_mask=vattention_mask, emo_label=emo_labels)
            # c_loss = cls_loss(cls_output, emo_labels)
            loss = c_loss + loss_avc + loss_itm
            # loss = c_loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step(epoch, i)
        if accelerator.state.num_processes > 1:
            loss = accelerator.gather_for_metrics(loss.detach()).sum()
            loss_avc = accelerator.gather_for_metrics(loss_avc.detach()).sum()
            loss_itm = accelerator.gather_for_metrics(loss_itm.detach()).sum()
            c_loss   = accelerator.gather_for_metrics(c_loss.detach()).sum()
        
        batch_losses += loss.item()
        batch_losses_avc += loss_avc.item()
        batch_losses_itm += loss_itm.item()
        batch_losses_cls += c_loss.item()

        all_emos_out, all_emos = accelerator.gather_for_metrics((cls_output, emo_labels))
        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labs.append(all_emos.data.cpu().numpy())

    emo_probs  = np.concatenate(emo_probs)
    emo_labs = np.concatenate(emo_labs)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labs, emo_preds)
    emo_ua = unweightedacc(emo_labs, emo_preds)

    save_results = {}
    save_results['Train_loss'] = batch_losses / len(dataloader)
    save_results['Train_cls'] = batch_losses_cls / len(dataloader)
    save_results['Train_avc'] = batch_losses_avc / len(dataloader)
    save_results['Train_avm'] = batch_losses_itm / len(dataloader)
    save_results['Train-UA'] = emo_accuracy
    save_results['Train-WA'] = emo_ua
    

    return save_results

def eval_model(accelerator, model, cls_loss, eval_loader):  
    model.eval()

    batch_losses = 0.0
    batch_losses_avc = 0.0
    batch_losses_itm = 0.0
    batch_losses_cls = 0.0
    emo_probs, emo_labs = [], []


    for data in eval_loader:
        ## analyze dataloader
        input_values, attention_mask, vinput_values, vattention_mask, emo_labels = data["input_values"], data["attention_mask"], data["vinput_values"], data["vattention_mask"], data['emo_labels']
        with accelerator.autocast():
            with torch.no_grad():
                cls_output, loss_avc, loss_itm = model(input_values=input_values, attention_mask=attention_mask, vinput_values=vinput_values, vattention_mask=vattention_mask, emo_label=emo_labels)
                # c_loss = cls_loss(cls_output, emo_labels)
                loss = c_loss + loss_avc + loss_itm
                # loss = c_loss

            if accelerator.state.num_processes > 1:
                loss = accelerator.gather_for_metrics(loss.detach()).sum()
                loss_avc = accelerator.gather_for_metrics(loss_avc.detach()).sum()
                loss_itm = accelerator.gather_for_metrics(loss_itm.detach()).sum()
                c_loss   = accelerator.gather_for_metrics(c_loss.detach()).sum()
        
        batch_losses += loss.item()
        batch_losses_avc += loss_avc.item()
        batch_losses_itm += loss_itm.item()
        batch_losses_cls += c_loss.item()

        all_emos_out, all_emos = accelerator.gather_for_metrics((cls_output, emo_labels))
        emo_probs.append(all_emos_out.data.cpu().numpy())
        emo_labs.append(all_emos.data.cpu().numpy())

    emo_probs  = np.concatenate(emo_probs)
    emo_labs = np.concatenate(emo_labs)
    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labs, emo_preds)
    emo_ua = unweightedacc(emo_labs, emo_preds)

    save_results = {}
    save_results['Valid_loss'] = batch_losses / len(eval_loader)
    save_results['Valid_cls'] = batch_losses_cls / len(eval_loader)
    save_results['Valid_avc'] = batch_losses_avc / len(eval_loader)
    save_results['Valid_avm'] = batch_losses_itm / len(eval_loader)
    save_results['Valid-UA'] = emo_accuracy
    save_results['Valid-WA'] = emo_ua

    return save_results

def train(args, accelerator, model, train_loader, dev_loader, test_loader=None, lr_scheduler=None, optimizer=None):
    
    cls_loss = torch.nn.CrossEntropyLoss()
    best_ua, best_wa = 0, 0

    for epoch in range(args.epochs):
        train_logs = train_model(accelerator, model, cls_loss, train_loader, lr_scheduler=lr_scheduler, optimizer=optimizer, epoch=epoch)
        dev_logs = eval_model(accelerator, model, cls_loss, dev_loader)

        if accelerator.is_main_process:
            log_str = "Epoch: " + str(epoch + 1) + " " 
            for k, v in train_logs.items():
                log_str += "| {}: {:.3}".format(k, v)
            for k, v in dev_logs.items():
                log_str += "| {}: {:.3}".format(k, v)
            print(log_str)
        
        if best_ua < dev_logs['Valid-UA']:
            best_ua = dev_logs['Valid-UA']
            best_wa = dev_logs['Valid-WA']
    
    accelerator.print("Best-UA: {}, Best-WA: {}".format(best_ua, best_wa))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--save_root', type=str, default='./session5', help='save prediction results and models')

    ## Params for model
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes [defined by args.label_path]')
    parser.add_argument('--pooling_model', type=str, default="mean", help="method for aggregating frame-level into utterence-level")

    ## Params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=1234, help='make split manner is same with same seed')
    parser.add_argument('--fp16', type=bool, default=True, help='whether to use fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--max_length', type=int, default=14, help='max length of audio')

    parser.add_argument('--train_src', type=str, default="/home/lqf/workspace/icassp2023/session5_train.scp", help='the path of train_src')
    parser.add_argument('--valid_src', type=str, default="/home/lqf/workspace/icassp2023/session5_valid.scp", help='the path of valid_src')
    #88
    
    args = parser.parse_args()

    seed_everything(seed=args.seed)

    train_src_path = args.train_src
    valid_src_path = args.valid_src

    model_path = args.save_root
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision = 'fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
        )
    max_eval_metric = -100

    accelerator.print (f'====== Reading Data =======')
    train_loader, eval_loader, class_weight = get_loaders(args, train_src_path, valid_src_path)  
    
    if accelerator.is_main_process:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    accelerator.print (f'====== Training and Evaluation =======')

    accelerator.print (f'Step1: build model (each folder has its own model)')

    model = MultiTaskModel.from_pretrained("/home/lqf/workspace/wavlm-multi/wavlm-large")
    # model.freeze_feature_extractor()
    num_step = len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = LinearWarmupCosineLRScheduler(optimizer=optimizer, max_epoch=args.epochs, min_lr=1e-5, init_lr=1e-4, warmup_start_lr=1e-6, warmup_steps=num_step * int(0.1 * args.epochs))
    model, optimizer, train_loader, eval_loader,lr_scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, lr_scheduler)
    
    min_eval_metric = 1000

    accelerator.print (f'Step2: training (multiple epoches)')

    train(args, accelerator, model, train_loader, eval_loader, None, None, optimizer)



    