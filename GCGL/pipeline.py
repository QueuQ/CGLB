import random
import datetime
import numpy as np
import torch
import importlib
from dgllife.model import load_pretrained
from dgllife.utils import EarlyStopping, Meter
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import copy
from utils import collate_molgraphs, load_model, GraphLevelDataset
import os
import errno
import pickle
joint_alias = ['joint', 'Joint', 'joint_replay_all', 'jointtrain', 'jointreplay']
def assign_hyp_param(args, params):
    if args['method']=='lwf':
        args['lwf_args'] = params
    if args['method'] == 'bare':
        args['bare_args'] = params
    if args['method'] == 'gem':
        args['gem_args'] = params
    if args['method'] == 'ewc':
        args['ewc_args'] = params
    if args['method'] == 'mas':
        args['mas_args'] = params
    if args['method'] == 'twp':
        args['twp_args'] = params
    if args['method'] in joint_alias:
        args['joint_args'] = params


def str2dict(s):
    # accepts a str like " 'k1':v1; ...; 'km':vm ", values (v1,...,vm) can be single values or lists (for hyperparameter tuning)
    output = dict()
    kv_pairs = s.replace(' ','').replace("'",'').split(';')
    for kv in kv_pairs:
        key = kv.split(':')[0]
        v_ = kv.split(':')[1]
        if '[' in v_:
            # transform list of values
            v_list = v_.replace('[','').replace(']','').split(',')
            vs=[]
            for v__ in v_list:
                try:
                    # if the parameter is float
                    vs.append(float(v__))
                except:
                    # if the parameter is str
                    vs.append(str(v__))
            output.update({key:vs})
        else:
            try:
                output.update({key: float(v_)})
            except:
                output.update({key: str(v_)})
    return output

def compose_hyper_params(hyp_params):
    hyp_param_list = [{}]
    for hk in hyp_params:
        hyp_param_list_ = []
        hyp_p_current = hyp_params[hk] if isinstance(hyp_params[hk],list) else [hyp_params[hk]]
        for v in hyp_p_current:
            for hk_ in hyp_param_list:
                hk__ = copy.deepcopy(hk_)
                hk__.update({hk: v})
                hyp_param_list_.append(hk__)
        hyp_param_list = hyp_param_list_
    return hyp_param_list


def remove_illegal_characters(name, replacement='_'):
    # replace any potential illegal characters with 'replacement'
    for c in ['-', '[' ,']' ,'{', '}', "'", ',', ':', ' ']:
        name = name.replace(c,replacement)
    return name

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
def predict(args, model, bg, task_i=None):
    node_feats = bg.ndata.pop(args['node_data_field']).cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).cuda()
        if args['backbone'] in ['GCN', 'GAT', 'Weave']:
            return model(bg.to(f"cuda:{args['gpu']}"), node_feats, edge_feats)
        else:
            return model(bg.to(f"cuda:{args['gpu']}"), node_feats, edge_feats, task_i)
    else:
        if args['backbone'] in ['GCN', 'GAT', 'Weave']:
            return model(bg.to(f"cuda:{args['gpu']}"), node_feats)
        else:
            return model(bg.to(f"cuda:{args['gpu']}"), node_feats, task_i)



def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer, task_i):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.cuda(), masks.cuda()
        logits = predict(args, model, bg, task_i)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Mask non-existing labels
        loss = loss_criterion(logits, labels) * (masks != 0).float()
        loss = loss[:,task_i].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)


def eval_single_task_multi_label(args, model, data_loader, task_i):
    # only returns the performance of the given task
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.cuda()
            logits = predict(args, model, bg, task_i)
            if isinstance(logits, tuple):
                logits = logits[0]
            eval_meter.update(logits, labels, masks)

    return eval_meter.compute_metric(args['metric_name'])[task_i]


def eval_all_learnt_task_multi_label(args, model, data_loader, task_i):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        if args['classifier_increase']:
            t_end = task_i + 1
        else:
            t_end = args['n_tasks']
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.cuda()
            logits = predict(args, model, bg)
            if isinstance(logits, tuple):
                logits = logits[0]

            eval_meter.update(logits[:,0:t_end], labels[:,0:t_end], masks)

        test_score =  eval_meter.compute_metric(args['metric_name'])
        score_mean = round(np.mean(test_score),4)

        for t in range(t_end):
            score = test_score[t]
            print(f"T{t:02d} {score:.4f}|", end="")

        print(f"score_mean: {score_mean}", end="")
        print()

    return test_score


def eval_single_task_multiclass_tskIL(args, model, data_loader, task_i):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            logits = predict(args, model, bg, task_i)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, args['tasks'][task_i]]
            y_pred.append(logits.detach().cpu())
            y_true.append(labels.detach().cpu())
        y_pred = torch.cat(y_pred, dim=0)
        y_pred = y_pred.argmax(-1)
        y_true = torch.cat(y_true, dim=0)
        for i, c in enumerate(args['tasks'][task_i]):
            y_true[y_true == c] = i
        ids_per_cls = [(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()]
        acc_per_cls = [(y_pred[ids] == y_true[ids]).sum() / len(ids) for ids in ids_per_cls]
    return sum(acc_per_cls).item() / len(acc_per_cls)

def eval_all_learnt_task_multiclass_tskIL(args, model, data_loaders, task_i):
    model.eval()
    acc_learnt_tsk = np.zeros(args['n_tasks'])
    with torch.no_grad():
        t_end = task_i + 1
        #t_end = args['n_tasks']
        for tid, data_loader in enumerate(data_loaders):
            y_pred = []
            y_true = []
            for batch_id, batch_data in enumerate(data_loader):
                smiles, bg, labels, masks = batch_data
                logits = predict(args, model, bg, task_i)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits[:, args['tasks'][tid]]
                y_pred.append(logits.detach().cpu())
                y_true.append(labels.detach().cpu())
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = y_pred.argmax(-1)
            y_true = torch.cat(y_true, dim=0)
            for i, c in enumerate(args['tasks'][tid]):
                y_true[y_true == c] = i
            ids_per_cls = [(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()]
            acc_per_cls = [(y_pred[ids] == y_true[ids]).sum() / len(ids) for ids in ids_per_cls]
            acc_learnt_tsk[tid] = sum(acc_per_cls).item() / len(acc_per_cls)

        score_mean = round(np.mean(acc_learnt_tsk[0:t_end]), 4)

        for t in range(t_end):
            score = acc_learnt_tsk[t]
            print(f"T{t:02d} {score:.4f}|", end="")

        print(f"score_mean: {score_mean}", end="")
        print()

    return acc_learnt_tsk

def eval_single_task_multiclass_clsIL(args, model, data_loader, task_i):
    # calculate the AP of tasks learnt so far
    model.eval()
    y_pred = []
    y_true = []
    learnt_cls = []
    for tid in range(task_i+1):
        learnt_cls.extend(args['tasks'][tid])
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            logits = predict(args, model, bg, task_i)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, learnt_cls]
            y_pred.append(logits.detach().cpu())
            y_true.append(labels.detach().cpu())
        y_pred = torch.cat(y_pred, dim=0)
        y_pred = y_pred.argmax(-1)
        y_true = torch.cat(y_true, dim=0)
        for i, c in enumerate(learnt_cls):
            y_true[y_true == c] = i
        ids_per_cls = [(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()]
        acc_per_cls = [(y_pred[ids] == y_true[ids]).sum() / len(ids) for ids in ids_per_cls]
    return sum(acc_per_cls).item() / len(acc_per_cls)

def eval_all_learnt_task_multiclass_clsIL(args, model, data_loaders, task_i):
    model.eval()
    acc_learnt_tsk = np.zeros(args['n_tasks'])
    learnt_cls,allclss = [],[]
    for tid in range(task_i + 1):
        learnt_cls.extend(args['tasks'][tid])
    for tid in args['tasks']:
        allclss.extend(tid)
    with torch.no_grad():
        #selected_clss = allclss if args['method'] is 'jointtrain' else learnt_cls
        #t_end = task_i + 1 if args['method'] is not 'jointtrain' else args['n_tasks']
        selected_clss = learnt_cls
        t_end = task_i + 1
        for tid, data_loader in enumerate(data_loaders[0:t_end]):
            y_pred,y_true = [],[]
            for batch_id, batch_data in enumerate(data_loader):
                smiles, bg, labels, masks = batch_data
                logits = predict(args, model, bg, task_i)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits[:, selected_clss]
                y_pred.append(logits.detach().cpu())
                y_true.append(labels.detach().cpu())
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = y_pred.argmax(-1)
            y_true = torch.cat(y_true, dim=0)
            for i, c in enumerate(selected_clss):
                y_true[y_true == c] = i
            ids_per_cls = {i:(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()}
            acc_per_cls = [(y_pred[ids_per_cls[ids]] == y_true[ids_per_cls[ids]]).sum().item() / len(ids_per_cls[ids]) for ids in ids_per_cls]
            acc_learnt_tsk[tid] = sum(acc_per_cls) / len(acc_per_cls)

        score_mean = round(np.mean(acc_learnt_tsk[0:t_end]), 4)

        for t in range(t_end):
            score = acc_learnt_tsk[t]
            print(f"T{t:02d} {score:.4f}|", end="")

        print(f"score_mean: {score_mean}", end="")
        print()

    return acc_learnt_tsk


def get_pipeline(args):
    if args['dataset'] in ['SIDER-tIL', 'Tox21-tIL']:
        # for multi-label datasets
        return pipeline_multi_label
    else:
        return pipeline_multi_class


def pipeline_multi_label(args, valid=False):
    '''

    :param args: arguments specified in train.py
    :valid: whether to use validation set or testing set to evaluate the model. If True, the model is in training mode. If false, the model is in testing mode, and the training epochs will be 0

    '''
    epochs = args['num_epochs'] if valid else 0
    torch.cuda.set_device(args['gpu'])
    # set_random_seed(args['random_seed'])
    G = GraphLevelDataset(args)
    dataset, train_set, val_set, test_set = G.dataset, G.train_set, G.val_set, G.test_set
    args['n_cls'] = dataset.labels.shape[1]

    args['tasks'] = [list(range(i, i + args['n_cls_per_task'])) for i in
                     range(0, args['n_cls'], args['n_cls_per_task'])]
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                               collate_fn=collate_molgraphs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    test_loader_ = DataLoader(test_set, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
    args['d_data'] = dataset.graphs[0].ndata['h'].shape[1]
    test_loader = val_loader if valid else test_loader_

    if args['pre_trained']:
        args['num_epochs'] = 0
        model = load_pretrained(args['exp'])
    else:
        args['n_tasks'] = len(args['tasks'])
        args['n_outheads'] = args['n_tasks'] # for multi-label tasks, #output heads equals #tasks
        model = load_model(args)
        life_model = importlib.import_module(f"Baselines.{args['method']}_model")
        life_model_ins = life_model.NET(model, args) if valid else None
        data_loader = DataLoader(train_set, batch_size=len(train_set),collate_fn=collate_molgraphs, shuffle=True)
        if life_model_ins is not None:
            life_model_ins.data_loader = data_loader
        loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights(train_set.indices).cuda(),
                                           reduction='none')

    model.cuda(args['gpu'])
    #score_mean = []
    score_matrix = np.zeros([args['n_tasks'], args['n_tasks']])

    prev_model = None
    val_func, test_func = eval_single_task_multi_label, eval_all_learnt_task_multi_label
    for tid, task_i in enumerate(args['tasks']):
        if args['method'] == 'jointtrain' and life_model_ins is not None:
            # reset the model for joint train
            dic = load_model(args).state_dict()
            life_model_ins.net.load_state_dict(dic)
        name, ite = args['current_model_save_path']
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_i}'
        save_model_path = f"{args['result_path']}/{subfolder_c}val_models/{save_model_name}.pkl"

        print('\n********' + str([tid, task_i]))
        dt = datetime.datetime.now()
        mkdir_if_missing(f"{args['result_path']}/early_stop")
        stopper = EarlyStopping(patience=args['patience'],filename='{}/early_stop/{}_{:02d}-{:02d}-{:02d}-{}.pth'.format(args['result_path'],
                dt.date(), dt.hour, dt.minute, dt.second,str(random.randint(100,999))))
        for epoch in range(epochs):
            # Train
            if args['method'] == 'lwf':
                life_model_ins.observe(train_loader, loss_criterion, tid, args, prev_model)
            else:
                life_model_ins.observe(train_loader, loss_criterion, tid, args)

            # Validation and early stop
            val_score = val_func(args, model, val_loader, tid)
            early_stop = stopper.step(val_score, model)

            if early_stop:
                print(epoch)
                break

        if not args['pre_trained'] and valid:
            stopper.load_checkpoint(model)
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args['gpu'])
        score_matrix[tid] = test_func(args, model, test_loader, tid)
        if valid:
            mkdir_if_missing(f"{args['result_path']}/{subfolder_c}/val_models")
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        if args['method'] == 'lwf':
            prev_model = copy.deepcopy(life_model_ins).cuda(args['gpu']) if valid else None

    AP = round(np.mean(score_matrix[-1, :]), 4)
    print('AP: ', round(np.mean(score_matrix[-1, :]), 4))
    backward = []
    for t in range(args['n_tasks'] - 1):
        b = score_matrix[args['n_tasks'] - 1][t] - score_matrix[t][t]
        backward.append(round(b, 4))
    mean_backward = round(np.mean(backward), 4)
    print('AF: ', mean_backward)
    return AP, mean_backward, score_matrix

def pipeline_multi_class(args, valid=False):
    epochs = args['num_epochs'] if valid else 0
    torch.cuda.set_device(args['gpu'])
    G = GraphLevelDataset(args)
    dataset, train_set, val_set, test_set = G.dataset, G.train_set, G.val_set, G.test_set
    coll_f = collate_molgraphs
    args['n_cls'] = dataset.labels.max().int().item() + 1
    args['n_outheads'] = args['n_cls']

    train_loader = [DataLoader(s, batch_size=args['batch_size'], collate_fn=coll_f, shuffle=True) for s in train_set] # train_set, a list of data for each task
    val_loader = [DataLoader(s, batch_size=args['batch_size'], collate_fn=coll_f) for s in val_set]
    test_loader_ = [DataLoader(s, batch_size=args['batch_size'],collate_fn=coll_f) for s in test_set]
    test_loader = val_loader if valid else test_loader_
    args['d_data'] = dataset.graphs[0].ndata['h'].shape[1]

    if args['pre_trained']:
        args['num_epochs'] = 0
        model = load_pretrained(args['exp'])
    else:
        args['n_tasks'] = len(args['tasks'])  # dataset.n_tasks
        model = load_model(args)
        life_model = importlib.import_module(f"Baselines.{args['method']}_model")
        life_model_ins = life_model.NET(model, args) if valid else None

        dataloader_shuffle = False if args['method'] is 'gem' else True # to ensure indices stored by gem always refer to same data in the dataloader
        data_loader = [DataLoader(s, batch_size=len(s), collate_fn=coll_f, shuffle=dataloader_shuffle) for s in train_set]
        if life_model_ins is not None:
            life_model_ins.data_loader = data_loader

        loss_criterion = torch.nn.functional.cross_entropy

    model.cuda(args['gpu'])
    score_matrix = np.zeros([args['n_tasks'], args['n_tasks']])

    prev_model = None
    if args['clsIL']:
        train_func = life_model_ins.observe_clsIL if valid else None
        val_func, test_func = eval_single_task_multiclass_clsIL, eval_all_learnt_task_multiclass_clsIL
    elif not args['clsIL']:
        train_func = life_model_ins.observe_tskIL_multicls if valid else None
        val_func, test_func = eval_single_task_multiclass_tskIL, eval_all_learnt_task_multiclass_tskIL

    for tid, task_i in enumerate(args['tasks']):
        if args['method'] == 'jointtrain' and life_model_ins is not None:
            # reset the model for joint train
            dic = load_model(args).state_dict()
            life_model_ins.net.load_state_dict(dic)
        name, ite = args['current_model_save_path']
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_i}'
        save_model_path = f"{args['result_path']}/{subfolder_c}val_models/{save_model_name}.pkl"
        print('\n********' + str([tid, task_i]),{i:args['n_per_cls'][i] for i in task_i})

        for epoch in range(epochs):
            # Train
            if args['method'] == 'lwf':
                train_func(train_loader, loss_criterion, tid, args, prev_model)
            else:
                train_func(train_loader, loss_criterion, tid, args)

        if not valid:
            # if testing, load the trained model
            model = pickle.load(open(save_model_path,'rb')).cuda(args['gpu'])
        score_matrix[tid] = test_func(args, model, test_loader, tid)
        if valid:
            mkdir_if_missing(f"{args['result_path']}/{subfolder_c}/val_models")
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        if args['method'] == 'lwf':
            prev_model = copy.deepcopy(life_model_ins).cuda(args['gpu']) if valid else None

    AP = round(np.mean(score_matrix[-1, :]), 4)
    print('AP: ', AP)
    backward = []
    for t in range(args['n_tasks'] - 1):
        b = score_matrix[args['n_tasks'] - 1][t] - score_matrix[t][t]
        backward.append(round(b, 4))
    mean_backward = round(np.mean(backward), 4)
    print('AF: ', mean_backward)
    return AP, mean_backward, score_matrix

