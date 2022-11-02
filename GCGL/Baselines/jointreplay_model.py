# modified from jointtrain, to replay all previous tasks during each new tasl

import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping, Meter
import numpy as np

def predict(args, model, bg, task_id):
    # bg = bg.cuda()
    node_feats = bg.ndata.pop(args['node_data_field']).cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).cuda()
        if args['backbone'] in ['GCN', 'GAT', 'Weave']:
            return model(bg, node_feats, edge_feats)
        else:
            return model(bg, node_feats, edge_feats, task_id)
    else:
        if args['backbone'] in ['GCN', 'GAT', 'Weave']:
            return model(bg, node_feats)
        else:
            return model(bg, node_feats, task_id)


class NET(torch.nn.Module):
    """
            Jointly trained model baseline for GCGL tasks

            :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
            :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

            """

    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.optimizer = Adam(model.parameters(), lr=args['lr'])

        self.data_loader = None

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, data_loader, loss_criterion, task_i, args):
        """
                        The method for learning the given tasks under the task-IL setting with multi-label classification datasets.

                        :param data_loader: The data loader for mini-batch training.
                        :param loss_criterion: The loss function.
                        :param task_i: Index of the current task.
                        :param args: Same as the args in __init__().

                        """
        self.net.train()
        train_meter = Meter()
        offset1, offset2 = task_i, task_i + 1
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg.to(f"cuda:{args['gpu']}"), task_i)

            # Mask non-existing labels
            if args['classifier_increase']:
                loss_all = loss_criterion(logits, labels)* (masks != 0).float()
                loss = loss_all[:, offset1:offset2].mean()
            else:
                loss_all = loss_criterion(logits, labels) * (masks != 0).float()
                loss = loss_all[:, task_i].mean()
            #loss_all = loss_criterion(logits, labels) * (masks != 0).float()
            #loss = loss_all[:,task_i].mean()

            for old_t in range(task_i):
                loss = loss + loss_all[:,old_t].mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)
        
        train_score = np.mean(train_meter.compute_metric(args['metric_name']))

    def observe_tskIL_multicls(self, data_loader, loss_criterion, task_i, args):
        """
                                The method for learning the given tasks under the task-IL setting with multi-class classification datasets.

                                :param data_loader: The data loader for mini-batch training.
                                :param loss_criterion: The loss function.
                                :param task_i: Index of the current task.
                                :param args: Same as the args in __init__().

                                """
        # not real cls-IL, just task Il under multi-class setting
        self.net.train()
        #train_meter = Meter()
        loss=0
        for oldt_id, old_t in enumerate(args['tasks']):  # range(task_i):
            for batch_id, batch_data in enumerate(data_loader[oldt_id]):
                smiles, bg, labels, masks = batch_data
                labels, masks = labels.cuda(), masks.cuda()
                logits = predict(args, self.net, bg.to(f"cuda:{args['gpu']}"), oldt_id)

                # Mask non-existing labels
                n_per_cls = [(labels == j).sum() for j in old_t]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                # labels= labels.long()
                for i, c in enumerate(old_t):
                    labels[labels == c] = i
                loss_aux = loss_criterion(logits[:, old_t], labels.long(), weight=loss_w_).float()
                loss = loss + loss_aux

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #train_meter.update(logits, labels, masks)

    def observe_clsIL(self, data_loader, loss_criterion, task_i, args):
        """
                                        The method for learning the given tasks under the class-IL setting with multi-class classification datasets.

                                        :param data_loader: The data loader for mini-batch training.
                                        :param loss_criterion: The loss function.
                                        :param task_i: Index of the current task.
                                        :param args: Same as the args in __init__().

                                        """
        # not real cls-IL, just task Il under multi-class setting
        self.net.train()
        #train_meter = Meter()

        #loss=0
        for oldt_id, old_t in enumerate(args['tasks'][task_i:task_i+1]): #[0:task_i+1]):  # range(task_i):
            clss = []
            for tid in args['tasks'][0:task_i+1]:
                clss.extend(tid)
            for batch_id, batch_data in enumerate(data_loader[task_i]):
                smiles, bg, labels, masks = batch_data
                labels, masks = labels.cuda(), masks.cuda()
                logits = predict(args, self.net, bg.to(f"cuda:{args['gpu']}"), oldt_id)

                # Mask non-existing labels
                n_per_cls = [(labels == j).sum() for j in clss]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                # labels= labels.long()
                for i, c in enumerate(clss):
                    labels[labels == c] = i
                #loss_aux = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_)
                #loss = loss + loss_aux

                y_pred=logits[:, clss].argmax(-1)
                ids_per_cls = {i: (labels == i).nonzero().view(-1).tolist() for i in labels.int().unique().tolist()}
                acc_per_cls = [round((y_pred[ids_per_cls[ids]] == labels[ids_per_cls[ids]]).sum().item() / len(ids_per_cls[ids]),3) for
                               ids in ids_per_cls]
                #print(acc_per_cls)

                loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        #self.optimizer.zero_grad()
        #loss.backward()
        #self.optimizer.step()
