import os
import torch
import random
import numpy as np
import sys
sys.path.append("./")
import torch.nn as nn
from core import enbisenetv2
import torch.cuda.amp as amp
import torch.utils.data as data
# from utils.opts import get_argparser
# from utils.loss import JointEdgeSegLoss
# from utils.stream_metrics import StreamSegMetrics
# from utils.scheduler import PolyLR, WarmupPolyLrScheduler
# from utils.common import get_dataset, get_params, mkdir, validate
from utils import get_argparser, JointEdgeSegLoss, StreamSegMetrics, PolyLR, \
                  WarmupPolyLrScheduler, get_dataset, get_params, mkdir, validate

def main():
    args = get_argparser().parse_args()
    if args.dataset.lower() == 'voc':
        args.num_classes = 21
    elif args.dataset.lower() == "shelves":
        args.num_classes = 4
    else:
        raise ValueError("dataset type: " + args.dataset + " is unknown")
    print("num_classes:", args.num_classes)

    os.environ["CUDA_VISIBLE_DEVICES"]=  args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # 构造模型
    model = enbisenetv2(args.num_classes)

    # 构造 metrics
    metrics = StreamSegMetrics(args.num_classes)

    ## 加载训练和验证集
    train_dst, val_dst = get_dataset(args)
    train_loader = data.DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=args.val_batch_size, shuffle=True, pin_memory=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
                    (args.dataset, len(train_dst), len(val_dst)))

    # 参与训练的模型权重
    backbone_params, heads_params = get_params(model)

    optimizer = torch.optim.SGD(params=[
            {'params': backbone_params, 'lr' : args.lr}, 
            {'params': heads_params, 'lr' : args.lr}
    ], momentum=0.9, weight_decay=args.weight_decay)

    if args.lr_policy == 'poly':
        scheduler = PolyLR(optimizer, args.total_itrs, power=0.9)
    elif args.lr_policy == 'warmup':
        scheduler = WarmupPolyLrScheduler(optimizer, power=0.9, 
                            max_iter=args.total_itrs, warmup_iter=1000, 
                            warmup_ratio=0.1, warmup='exp', last_epoch=-1)
    else:
        raise ValueError("scheduler type: " + args.lr_policy + " is Unkown")

    # 损失函数
    if args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif args.loss_type == 'JointEdgeSegLoss':
        criterion = JointEdgeSegLoss(classes=args.num_classes, upper_bound=1.0)
    else:
        raise ValueError("loss funciotn type: " + args.loss_type + " is Unkown")

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(), 
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if args.ckpt is not None and os.path.exists(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location=torch.device("cpu"))
        state_dicts = checkpoint["model_state"]
        pretrained_dict = {k:v for k, v in state_dicts.items() if k in model.state_dict()}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)

        model = nn.DataParallel(model)
        model.to(device)
        if args.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % args.ckpt)
        print("Model restored from %s" % args.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    scaler = amp.GradScaler()
    interval_loss = 0

    while True: 
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels, edgemask) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32) 
            labels = labels.to(device, dtype=torch.long)
            edgemask = edgemask.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                outputs, aux_out0, aux_out1 = model(images)
                # 加权交叉熵函数与边缘注意力损失函数
                loss = criterion(outputs, (labels, edgemask)) 
                loss += args.aux_alpha * criterion(aux_out0, (labels, edgemask))
                loss += args.aux_alpha * criterion(aux_out1, (labels, edgemask))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                vis_lr = scheduler.get_lr()
                vis_lr = sum(vis_lr) / len(vis_lr)
                print("Epoch %d, Itrs %d/%d, lr=%f, Loss=%f" %
                      (cur_epochs, cur_itrs, args.total_itrs, vis_lr, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % args.val_interval == 0:
                save_ckpt('checkpoints/latest_%s.pth' % (args.dataset))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=args, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=None)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s.pth' % (args.dataset))

                model.train()
            scheduler.step()

            if cur_itrs >=  args.total_itrs:
                print("best_score:", best_score)
                return

if __name__ == '__main__':
    main()