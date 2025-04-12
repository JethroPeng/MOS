import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import network
from data.augmentations import get_transform
from data.get_datasets  import get_datasets, get_class_splits
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from util.loss import info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cub, scars, fgvc_aricraft')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_root', type=str, default='dev_outputs')
    parser.add_argument('--exp_name', default='mos', type=str)
    parser.add_argument("--dataset_dir"  , type=str, default='cub', help="dataset root path")
    parser.add_argument("--osr_split_dir", type=str, default='data/ssb_splits', help="osr")
    parser.add_argument("--pretrain_path", type=str, default='pretrain_weight/dino_vitbase16_pretrain.pth', help="pretrain_path")
    args                       = parser.parse_args()
    args                       = get_class_splits(args)
    args.num_labeled_classes   = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.interpolation         = 3
    args.crop_pct              = 0.875
    args.image_size            = 224
    args.feat_dim              = 768
    args.num_mlp_layers        = 3
    args.mlp_out_dim           = args.num_labeled_classes + args.num_unlabeled_classes
    return args
    
def get_model(args,device):
    backbone   = network.__dict__['vit_base']()
    state_dict = torch.load(args.pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict, strict=False)
    for m in backbone.parameters(): m.requires_grad = False
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block: m.requires_grad = True
    for name, param in backbone.named_parameters():
        if param.requires_grad: print(f'{name} - {param.shape}')
    projector = network.SAHead(in_dim=2*args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model     = nn.Sequential(backbone, projector).to(device)
    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        warmup_dict = torch.load(args.warmup_model_dir, map_location='cpu')
        model.load_state_dict(warmup_dict['model'], strict=False)
    return model

def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups     = network.get_params_groups(student)
    optimizer         = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler       = torch.cuda.amp.GradScaler() if args.fp16 else None
    exp_lr_scheduler  = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
    cluster_criterion = DistillLoss(args.warmup_teacher_temp_epochs,args.epochs,args.n_views,args.warmup_teacher_temp,args.teacher_temp)

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            (images_masks_view1, images_masks_view2), class_labels, uq_idxs, mask_lab = batch
            images_view1, masks_view1 = images_masks_view1
            images_view2, masks_view2 = images_masks_view2
            images                    = torch.cat([images_view1, images_view2], dim=0).cuda(non_blocking=True) 
            images_masked             = torch.cat([masks_view1, masks_view2], dim=0).cuda(non_blocking=True)        
            mask_lab               = mask_lab[:, 0]
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out,sup_next = student((images,None))
                teacher_out = student_out.detach()
                student_proj_masked, student_out_masked,_ = student((images_masked,sup_next))
                teacher_out_masked = student_out_masked.detach()

                # clustering, sup
                sup_logits         = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_logits_masked  = torch.cat([f[mask_lab] for f in (student_out_masked / 0.1).chunk(2)], dim=0)
                sup_labels         = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss           = nn.CrossEntropyLoss()(sup_logits, sup_labels) + nn.CrossEntropyLoss()(sup_logits_masked, sup_labels)     

                # clustering, unsup
                avg_probs          = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss        = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss       = cluster_criterion(student_out, teacher_out, epoch) + args.memax_weight * me_max_loss
                avg_probs_masked   = (student_out_masked / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss_masked = - torch.sum(torch.log(avg_probs_masked**(-avg_probs_masked))) + math.log(float(len(avg_probs_masked)))
                cluster_loss      += cluster_criterion(student_out_masked, teacher_out_masked, epoch) + args.memax_weight * me_max_loss_masked


                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss   = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
                contrastive_logits_masked, contrastive_labels_masked = info_nce_logits(features=student_proj_masked)
                contrastive_loss  += torch.nn.CrossEntropyLoss()(contrastive_logits_masked, contrastive_labels_masked)

                # representation learning, sup
                student_proj       = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj       = torch.nn.functional.normalize(student_proj, dim=-1)
                student_proj_masked= torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj_masked.chunk(2)], dim=1)
                student_proj_masked= torch.nn.functional.normalize(student_proj_masked, dim=-1)
                sup_con_labels     = class_labels[mask_lab]
                sup_con_loss       = SupConLoss()(student_proj, labels=sup_con_labels) + SupConLoss()(student_proj_masked, labels=sup_con_labels)

                pstr  = f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = (1 - args.sup_weight) * (cluster_loss+ contrastive_loss) + args.sup_weight * (cls_loss +  sup_con_loss)
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'.format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

def test(model, test_loader, epoch, save_name, args):
    model.eval()
    preds, targets,mask = [], [],np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        (images, images_masked) = images
        images = images.to(device, non_blocking=True)
        images_masked = images_masked.to(device, non_blocking=True)
        with torch.no_grad():
            _,_,add_info = model((images,None))
            _,logits,_ = model((images_masked,add_info))
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    return all_acc, old_acc, new_acc

if __name__ == "__main__":
    args                            = parse_args()
    init_experiment(args,runner_name=[args.exp_name])
    device                          = torch.device(args.device)
    torch.backends.cudnn.benchmark  = True
    model                           = get_model(args,device)
    
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform                 = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,train_transform,test_transform,args)
    
    label_len,unlabelled_len        = len(train_dataset.labelled_dataset),len(train_dataset.unlabelled_dataset)
    sample_weights                  = torch.DoubleTensor([1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))])
    sampler                         = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
    
    train_loader                    = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled          = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,batch_size=256, shuffle=True, pin_memory=False)
    
    train(model, train_loader, None, test_loader_unlabelled, args)