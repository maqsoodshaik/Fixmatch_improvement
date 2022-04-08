import argparse
import math
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

import models.wideresnet as models
from dataset.cifar import get_cifar10, get_cifar100
from utils.misc import accuracy
from vat import VATLoss


def save_checkpoint(state, is_best, checkpoint, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main():
    parser = argparse.ArgumentParser(description="FixMatch Training")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--dataset",
        default="cifar100",
        type=str,
        choices=["cifar10", "cifar100"],
        help="dataset name",
    )
    parser.add_argument(
        "--num-labeled", type=int, default=2500, help="number of labeled data"
    )
    parser.add_argument(
        "--expand-labels", action="store_true", help="expand labels to fit eval steps"
    )
    parser.add_argument(
        "--total-steps", default=2, type=int, help="number of total steps to run"
    )
    parser.add_argument(
        "--eval-step", default=1, type=int, help="number of eval steps to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument("--batch-size", default=64, type=int, help="train batchsize")
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.03,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--warmup", default=1, type=float, help="warmup epochs (unlabeled data based)"
    )
    parser.add_argument("--wdecay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--nesterov", action="store_true", default=True, help="use nesterov momentum"
    )
    parser.add_argument(
        "--use-ema", action="store_true", default=True, help="use EMA model"
    )
    parser.add_argument("--ema-decay", default=0.999, type=float, help="EMA decay rate")
    parser.add_argument(
        "--mu", default=7, type=int, help="coefficient of unlabeled batch size"
    )
    parser.add_argument(
        "--lambda-u", default=1, type=float, help="coefficient of unlabeled loss"
    )
    parser.add_argument("--T", default=1, type=float, help="pseudo label temperature")
    parser.add_argument(
        "--threshold", default=0.95, type=float, help="pseudo label threshold"
    )
    parser.add_argument(
        "--out", default="result", help="directory to output the result"
    )
    parser.add_argument(
        "--bestmodel", default="./result/model_best.pth.tar", help="directory to output the result"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,  # ./result/past.pth.tar
        help="path to latest model (default: none)",
    )
    parser.add_argument(
        "--checkpoint",
        default="./result/checkpoint.pth.tar",
        type=str,  # ./result/past.pth.tar
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--seed", default=256, type=int, help="random seed")
    parser.add_argument("--vat-xi", default=0.5, type=float, help="VAT xi parameter")
    parser.add_argument(
        "--vat-eps", default=2.0, type=float, help="VAT epsilon parameter"
    )
    parser.add_argument(
        "--vat-iter", default=1, type=int, help="VAT iteration parameter"
    )
    parser.add_argument(
        "--model-depth", type=int, default=28, help="model depth for wide resnet"
    )
    parser.add_argument(
        "--model-width", type=int, default=2, help="model width for wide resnet"
    )
    parser.add_argument(
        "--train-mode", type=int, default=0, help="flag to indicate training"
    )

    args = parser.parse_args()

    def create_model(args):
        model = models.build_wideresnet(
            depth=args.model_depth,
            widen_factor=args.model_width,
            dropout=0,
            num_classes=args.num_classes,
        )
        return model

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, "./data")
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, "./data")
    if args.resume:
        model = torch.load(args.resume)
    else:
        model = create_model(args)
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model.to(args.device)

    bias_array = ["bias", "bn"]  # setting weight decay except for batchnormalization as it causes fluctuations during training
    grouped_parameters = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(bias in name for bias in bias_array)
            ],
            "weight_decay": args.wdecay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(bias in name for bias in bias_array)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.SGD(
        grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov
    )

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps
    )

    if args.use_ema:
        from models.ema import ModelEMA

        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0
    if args.train_mode == 1:
        if args.resume:
            assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
            args.out = os.path.dirname(args.checkpoint)
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint["epoch"]
            if args.use_ema:
                ema_model.ema.load_state_dict(checkpoint["ema_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

        model.zero_grad()
        args.test_dataset = test_dataset
        train(
            args,
            labeled_trainloader,
            unlabeled_trainloader,
            test_loader,
            model,
            optimizer,
            ema_model,
            scheduler,
        )
    else:
        # test
        args.dict = 0
        test_logits = test(
            args, test_dataset, filepath=args.bestmodel
        )
        test_accuracy_fn(test_logits, test_loader)


def train(
    args,
    labeled_trainloader,
    unlabeled_trainloader,
    test_loader,
    model,
    optimizer,
    ema_model,
    scheduler,
):
    test_accs = []
    best_acc = 0
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    vatloss = VATLoss(args)
    model.train()
    for epoch in range(args.start_epoch, args.epochs):

        losses = 0
        losses_x = 0
        losses_u = 0
        mask_probs = 0
        mask_probs_non_r = 0
        sim_track = 0
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s, inputs_u_w_non_r), _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, inputs_u_w_non_r), _ = unlabeled_iter.next()

            batch_size = inputs_x.shape[0]
            optimizer.zero_grad()
            inputs_u_w_non_r = inputs_u_w_non_r.to(args.device)

            vtloss = vatloss(model, inputs_u_w_non_r)

            vtloss = vtloss.to(args.device)
            inputs_u_w = inputs_u_w.to(args.device) + vtloss.to(args.device)
            inputs = torch.cat(
                (
                    inputs_x.to(args.device),
                    inputs_u_w.to(args.device),
                    inputs_u_s.to(args.device),
                    inputs_u_w_non_r.to(args.device),
                )
            ).to(args.device)
            targets_x = targets_x.to(args.device)
            optimizer.zero_grad()
            logits = model(inputs)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s, logits_u_w_non_r = logits[batch_size:].chunk(3)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            pseudo_label_non_r = torch.softmax(
                logits_u_w_non_r.detach() / args.T, dim=-1
            )
            #####sim imp
            logits_u_s_sorted, _ = torch.sort(logits_u_s)
            sim = torch.abs(
                logits_u_s_sorted[:, None, :] - logits_u_s_sorted[None, :, :]
            ).sum(axis=-1)
            sim_ind = (sim <= 0.3).nonzero()[
                ((sim <= 0.3).nonzero()[:, 0] - (sim <= 0.3).nonzero()[:, 1]).nonzero()
            ]
            sim_ind = sim_ind.squeeze()
            sim_track += sim_ind.shape[0]
            mask_gt = (
                logits_u_s_sorted[sim_ind[:, 0], 9]
                > logits_u_s_sorted[sim_ind[:, 1], 9]
            )
            mask_ls = ~(mask_gt)
            pseudo_label[sim_ind[mask_gt][:, 1]] = pseudo_label[sim_ind[mask_gt][:, 0]]
            pseudo_label[sim_ind[mask_ls][:, 0]] = pseudo_label[sim_ind[mask_ls][:, 1]]
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            max_probs_non_r, targets_u_non_r = torch.max(pseudo_label_non_r, dim=-1)
            mask_non_r = max_probs_non_r.ge(args.threshold).float()

            Lu = (
                F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask
            ).mean()

            Lu_r = (
                F.cross_entropy(logits_u_s, targets_u_non_r, reduction="none")
                * mask_non_r
            ).mean()
            Lu = Lu + Lu_r

            loss = Lx + args.lambda_u * Lu

            loss.backward()

            losses += loss.item()
            losses_x += Lx.item()
            losses_u += Lu.item()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            optimizer.zero_grad()
            mask_probs += mask.mean().item()
            mask_probs_non_r += mask_non_r.mean().item()

        model_to_save = model.module if hasattr(model, "module") else model
        if args.use_ema:
            ema_to_save = (
                ema_model.ema.module
                if hasattr(ema_model.ema, "module")
                else ema_model.ema
            )

        torch.save(model, "./result/past.pth.tar")
        args.dict = 1
        test_logits = test(
            args, args.test_dataset, filepath=os.path.join(args.out, "past.pth.tar")
        )
        test_acc = test_accuracy_fn(test_logits, test_loader)
        args.writer.add_scalar("train/1.train_loss", losses / args.eval_step, epoch)
        args.writer.add_scalar("train/2.train_loss_x", losses_x / args.eval_step, epoch)
        args.writer.add_scalar("train/3.train_loss_u", losses_u / args.eval_step, epoch)
        args.writer.add_scalar("train/4.mask", mask_probs / args.eval_step, epoch)
        args.writer.add_scalar(
            "train/4.mask_r", mask_probs_non_r / args.eval_step, epoch
        )
        args.writer.add_scalar("test/1.test_acc", test_acc, epoch)
        args.writer.add_scalar("sim_parameter", sim_track / args.eval_step, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model_to_save.state_dict(),
                "ema_state_dict": ema_to_save.state_dict() if args.use_ema else None,
                "acc": test_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            args.out,
        )
        test_accs.append(test_acc)

    args.writer.close()


def test(args, testdataset, filepath="./path/to/model.pth.tar"):
    test_loader = DataLoader(
        testdataset,
        sampler=SequentialSampler(testdataset),
        batch_size=64,
        num_workers=4,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dict == 0:
        checkpoint = torch.load(filepath, map_location=device)
        model = models.build_wideresnet(
            depth=args.model_depth,
            widen_factor=args.model_width,
            dropout=0,
            num_classes=args.num_classes,
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = torch.load(filepath, map_location=device)

    model.eval()
    predicted_arr = torch.tensor([])
    predicted_arr = predicted_arr.to(device)
    labels_arr = torch.tensor([])
    labels_arr = labels_arr.to(device)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_arr = torch.cat((predicted_arr, outputs), dim=0)
    return predicted_arr


def test_accuracy_fn(test_logits, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_arr = torch.tensor([])
    labels_arr = labels_arr.to(device)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        labels_arr = torch.cat((labels_arr, labels), dim=0)
    accuracy_val = accuracy(test_logits, labels_arr)[0]
    print("test accuracy: %.3f" % accuracy_val)
    return accuracy_val


if __name__ == "__main__":
    main()
