import torch


class NET(torch.nn.Module):
    """
        MAS baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()
        self.reg = args.mas_args['memory_strength']

        self.task_manager = task_manager

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = 0
        self.optpar = []
        self.fisher = []
        self.n_seen_examples = 0
        self.mem_mask = None
        self.epochs = 0

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks under the class-IL setting.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                :param dataset: The entire dataset (not in use in the current baseline).

                """
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        n_new_examples = len(train_ids)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output = self.net(g, features)
        if isinstance(output,tuple):
            output = output[0]
        output_labels = labels[train_ids]
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        if args.classifier_increase:
            loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], output_labels, weight=loss_w_)

        if t > 0:
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()

        if last_epoch==0:
            self.optpar = []
            new_fisher = []
            output = self.net(g, features)
            if isinstance(output, tuple):
                output = output[0]
            if args.classifier_increase:
                output = output[train_ids, offset1:offset2]
            else:
                output = output[train_ids, :]

            output.pow_(2)
            loss = output.mean()
            self.net.zero_grad()
            loss.backward()

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                new_fisher.append(pg)
                self.optpar.append(pd)

            if len(self.fisher) != 0:
                for i, f in enumerate(new_fisher):
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i]*n_new_examples) / (
                                self.n_seen_examples + n_new_examples)
                self.n_seen_examples += n_new_examples
            else:
                for i, f in enumerate(new_fisher):
                    self.fisher.append(new_fisher[i])
                self.n_seen_examples = n_new_examples

            self.current_task = t

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                        :param dataset: The entire dataset (not in use in the current baseline).

                        """
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        n_new_examples = len(train_ids)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output = self.net(g, features)
        if isinstance(output,tuple):
            output = output[0]
        output_labels = labels[train_ids]
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])

        if t > 0:
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()

        if last_epoch==0:
            self.optpar = []
            new_fisher = []
            output = self.net(g, features)
            if isinstance(output, tuple):
                output = output[0]
            output = output[train_ids, offset1:offset2]

            output.pow_(2)
            loss = output.mean()
            self.net.zero_grad()
            loss.backward()

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                new_fisher.append(pg)
                self.optpar.append(pd)

            if len(self.fisher) != 0:
                for i, f in enumerate(new_fisher):
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i]*n_new_examples) / (self.n_seen_examples + n_new_examples)
                self.n_seen_examples += n_new_examples
            else:
                for i, f in enumerate(new_fisher):
                    self.fisher.append(new_fisher[i])
                self.n_seen_examples = n_new_examples

            self.current_task = t

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param dataloader: The data loader for mini-batch training
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                        :param dataset: The entire dataset (currently not in use).

                        """
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        n_new_examples = len(train_ids)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_labels = output_labels - offset1

            output_predictions = self.net.forward_batch(blocks, input_features)
            if isinstance(output_predictions, tuple):
                output_predictions = output_predictions[0]
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)

            if t > 0:
                for i, p in enumerate(self.net.parameters()):
                    l = self.reg * self.fisher[i]
                    l = l * (p - self.optpar[i]).pow(2)
                    loss += l.sum()
            loss.backward()
            self.opt.step()

        if last_epoch==0:
            self.optpar = []
            new_fisher = []
            pgss = []

            #output = torch.tensor([]).cuda(args.gpu)
            for input_nodes, output_nodes, blocks in dataloader:
                pgs = []
                blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                #output_labels = blocks[-1].dstdata['label'].squeeze()
                output_predictions = self.net.forward_batch(blocks, input_features)
                if isinstance(output_predictions, tuple):
                    output_predictions = output_predictions[0]
                output_predictions.pow_(2)
                loss = output_predictions.mean()
                self.net.zero_grad()
                loss.backward()
                for p in self.net.parameters():
                    pg = p.grad.data.clone().pow(2)
                    pgs.append(pg)
                pgss.append(pgs)

            for i,p in enumerate(self.net.parameters()):
                pg_ = []
                for pgs_ in pgss:
                    pg_.append(pgs_[i])
                new_fisher.append(sum(pg_) / len(pg_))
                pd = p.data.clone()
                self.optpar.append(pd)

            if len(self.fisher) != 0:
                for i, f in enumerate(new_fisher):
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i]*n_new_examples) / (
                                self.n_seen_examples + n_new_examples)
                self.n_seen_examples += n_new_examples
            else:
                for i, f in enumerate(new_fisher):
                    self.fisher.append(new_fisher[i])
                self.n_seen_examples = n_new_examples

            self.current_task = t

    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                                The method for learning the given tasks under the class-IL setting with mini-batch training.

                                :param args: Same as the args in __init__().
                                :param g: The graph of the current task.
                                :param dataloader: The data loader for mini-batch training
                                :param features: Node features of the current task.
                                :param labels: Labels of the nodes in the current task.
                                :param t: Index of the current task.
                                :param train_ids: The indices of the nodes participating in the training.
                                :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                                :param dataset: The entire dataset (currently not in use).

                                """
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        n_new_examples = len(train_ids)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_predictions = self.net.forward_batch(blocks, input_features)
            if isinstance(output_predictions, tuple):
                output_predictions = output_predictions[0]
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)

            if t > 0:
                for i, p in enumerate(self.net.parameters()):
                    l = self.reg * self.fisher[i]
                    l = l * (p - self.optpar[i]).pow(2)
                    loss += l.sum()
            loss.backward()
            self.opt.step()

        if last_epoch==0:
            self.optpar = []
            new_fisher = []
            pgss = []
            for input_nodes, output_nodes, blocks in dataloader:
                pgs = []
                blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label'].squeeze()
                output_predictions = self.net.forward_batch(blocks, input_features)
                if isinstance(output_predictions, tuple):
                    output_predictions = output_predictions[0]
                output_predictions.pow_(2)
                loss = output_predictions.mean()
                self.net.zero_grad()
                loss.backward()
                for p in self.net.parameters():
                    pg = p.grad.data.clone().pow(2)
                    pgs.append(pg)
                pgss.append(pgs)

            for i,p in enumerate(self.net.parameters()):
                pg_ = []
                for pgs_ in pgss:
                    pg_.append(pgs_[i])
                new_fisher.append(sum(pg_) / len(pg_))
                pd = p.data.clone()
                self.optpar.append(pd)

            if len(self.fisher) != 0:
                for i, f in enumerate(new_fisher):
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i]*n_new_examples) / (
                                self.n_seen_examples + n_new_examples)
                self.n_seen_examples += n_new_examples
            else:
                for i, f in enumerate(new_fisher):
                    self.fisher.append(new_fisher[i])
                self.n_seen_examples = n_new_examples

            self.current_task = t
