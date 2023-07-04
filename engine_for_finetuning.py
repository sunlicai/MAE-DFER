import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import pickle

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    outputs, targets = [], []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        output, target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        outputs.append(output)
        targets.append(target)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # me: cal total metrics across the val set
    preds, labels = np.concatenate(outputs), np.concatenate(targets)
    preds = np.argmax(preds, axis=1)
    from sklearn.metrics import confusion_matrix, f1_score
    conf_mat = confusion_matrix(y_pred=preds, y_true=labels)
    class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    uar = np.mean(class_acc)
    war = conf_mat.trace() / conf_mat.sum()
    weighted_f1 = f1_score(y_pred=preds, y_true=labels, average='weighted')
    micro_f1 = f1_score(y_pred=preds, y_true=labels, average='micro')
    macro_f1 = f1_score(y_pred=preds, y_true=labels, average='macro')
    metric_logger.meters['uar'].update(uar, n=len(preds))
    metric_logger.meters['war'].update(war, n=len(preds))
    metric_logger.meters['weighted_f1'].update(weighted_f1, n=len(preds))
    metric_logger.meters['micro_f1'].update(micro_f1, n=len(preds))
    metric_logger.meters['macro_f1'].update(macro_f1, n=len(preds))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    print('* WAR {war.global_avg:.4f} UAR {uar.global_avg:.4f} weighted_f1 {weighted_f1.global_avg:.4f} micro_f1 {micro_f1.global_avg:.4f} macro_f1 {macro_f1.global_avg:.4f}'
          .format(war=metric_logger.war, uar=metric_logger.uar, weighted_f1=metric_logger.weighted_f1, micro_f1=metric_logger.micro_f1, macro_f1=metric_logger.macro_f1))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, save_feature=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    # me: for saving feature in the last layer
    saved_features = {}

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # me: for saving feature in the last layer
            if save_feature:
                output, saved_feature = model(videos, save_feature=save_feature)
            else:
                output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

            # me: for saving feature in the last layer
            if save_feature:
                if ids[i] not in saved_features:
                    saved_features[ids[i]] = {'chunk_id': [], 'split_id': [],
                                              'label': int(target[i].cpu().numpy()),
                                              'feature': [], 'logit': []}
                saved_features[ids[i]]['chunk_id'].append(int(chunk_nb[i].cpu().numpy()))
                saved_features[ids[i]]['split_id'].append(int(split_nb[i].cpu().numpy()))
                saved_features[ids[i]]['feature'].append(saved_feature.data[i].cpu().numpy().tolist())
                saved_features[ids[i]]['logit'].append(output.data[i].cpu().numpy().tolist())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)

    # me: for saving feature in the last layer
    if save_feature:
        feature_file = file.replace(file[-4:], '_feature.pkl')
        pickle.dump(saved_features, open(feature_file, 'wb'))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, args, best=False):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    # me: for saving feature in the last layer
    overall_saved_features = {}

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt') if not best else os.path.join(eval_path, str(x) + '_best.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label

        # me: for saving feature in the last layer
        if args.save_feature:
            feature_file = file.replace(file[-4:], '_feature.pkl')
            saved_features = pickle.load(open(feature_file, 'rb'))
            for sample_id in saved_features.keys():
                if sample_id not in overall_saved_features:
                    overall_saved_features[sample_id] = {
                        'chunk_split_id': [], # the only identifier for each view
                        'label': saved_features[sample_id]['label'],
                        'feature': [], 'prob': []}
                chunk_ids = saved_features[sample_id]['chunk_id']
                split_ids = saved_features[sample_id]['split_id']
                for idx, (chunk_id, split_id) in enumerate(zip(chunk_ids, split_ids)):
                    chunk_split_id = f"{chunk_id}_{split_id}"
                    # avoid repetition
                    if chunk_split_id not in overall_saved_features[sample_id]['chunk_split_id']:
                        overall_saved_features[sample_id]['chunk_split_id'].append(chunk_split_id)
                        overall_saved_features[sample_id]['feature'].append(saved_features[sample_id]['feature'][idx])
                        # NOTE: do softmax, logit -> prob
                        overall_saved_features[sample_id]['prob'].append(softmax(saved_features[sample_id]['logit'][idx]))


    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    # me: more metrics and save preds
    pred_dict = {'id': [], 'label': [], 'pred': []}
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
        pred = int(np.argmax(np.mean(dict_feats[item], axis=0)))
        label = int(dict_label[item])
        pred_dict['pred'].append(pred)
        pred_dict['label'].append(label)
        pred_dict['id'].append(item.strip())
    # from multiprocessing import Pool
    # p = Pool(4)
    # ans = p.map(compute_video, input_lst)
    # me: disable multi-process because it often gets stuck
    ans = [compute_video(lst) for lst in input_lst]
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)

    # me: for saving feature in the last layer
    if args.save_feature:
        # get avg feature and pred
        for sample_id in overall_saved_features.keys():
            overall_saved_features[sample_id]['feature'] = np.mean(overall_saved_features[sample_id]['feature'], axis=0)
            overall_saved_features[sample_id]['pred'] = int(np.argmax(np.mean(overall_saved_features[sample_id]['prob'], axis=0)))
        feature_file = os.path.join(eval_path, 'overall_feature.pkl') if not best else os.path.join(eval_path, 'overall_feature_best.pkl')
        pickle.dump(overall_saved_features, open(feature_file, 'wb'))

    return final_top1*100 ,final_top5*100, pred_dict

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
