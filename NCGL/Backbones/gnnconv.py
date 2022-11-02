"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from dgl.nn.pytorch import edge_softmax
from dgl.utils import expand_as_pair
from dgl.base import DGLError
import math
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
import torch.autograd as autograd
class CustomSGC_Agg(nn.Module):
    # only the neighborhood aggregation of SGC
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat


class GINConv(nn.Module):
    r"""Graph Isomorphism Network layer from paper `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    """
    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))
        self.learn_eps = learn_eps
        self.init_eps = init_eps

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        self.eps.fill_(self.init_eps)
        self.apply_func.reset_parameters()

    def forward(self, graph, feat):
        r"""Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        elist = []
        graph = graph.local_var().to('cuda:{}'.format(feat.get_device()))
        feat_src, feat_dst = expand_as_pair(feat)
        graph.srcdata['h'] = feat_src
        graph.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
        if self.apply_func is not None:
            rst = self.apply_func(rst)
        #graph.apply_edges(fn.u_sub_v('h', 'h', 'e'))
        graph.apply_edges(lambda edges: {'e': th.sum((th.mul(edges.src['h'], th.tanh(edges.dst['h']))), 1)})
        e = graph.edata.pop('e')

        e_soft = edge_softmax(graph, e)
        elist.append(e_soft)
        return rst, elist

    def forward_batch(self, block, feat):
        r"""Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        elist = []
        block = block.local_var().to('cuda:{}'.format(feat.get_device()))
        feat_src, feat_dst = expand_as_pair(feat, block)

        block.srcdata['h'] = feat_src
        block.dstdata['h'] = feat_dst

        block.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        rst = (1 + self.eps) * feat_dst + block.dstdata['neigh']
        if self.apply_func is not None:
            rst = self.apply_func(rst)
        #graph.apply_edges(fn.u_sub_v('h', 'h', 'e'))
        block.apply_edges(lambda edges: {'e': th.sum((th.mul(edges.src['h'], th.tanh(edges.dst['h']))), 1)})
        e = block.edata.pop('e')

        e_soft = edge_softmax(block, e)
        elist.append(e_soft)
        return rst, elist


def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores

def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, negative_slope=0.2):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph, feat):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        elist = []
        graph = graph.local_var().to('cuda:{}'.format(feat.get_device()))
        h = self.linear(feat)
        graph.ndata['h'] = h
        graph.update_all(gcn_msg, gcn_reduce)
        h = graph.ndata['h']

        graph.apply_edges(lambda edges: {'e': th.sum((th.mul(edges.src['h'], th.tanh(edges.dst['h']))), 1)})
        e = self.leaky_relu(graph.edata.pop('e'))

        e_soft = edge_softmax(graph, e)
        elist.append(e_soft)
        return h, elist

    def forward_batch(self, block, feat):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        elist = []
        block = block.local_var().to('cuda:{}'.format(feat.get_device()))
        feat_src, feat_dst = expand_as_pair(feat)
        h = self.linear(feat_src)
        block.srcdata['h'] = h
        block.update_all(gcn_msg, gcn_reduce)
        h = block.dstdata['h']

        block.apply_edges(lambda edges: {'e': th.sum((th.mul(edges.src['h'], th.tanh(edges.dst['h']))), 1)})
        e = self.leaky_relu(block.edata.pop('e'))

        e_soft = edge_softmax(block, e)
        elist.append(e_soft)
        return h, elist

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)


class GATConv(nn.Module):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.
    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:
    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 k = 1):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
            
        self.attn_l1 = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r1 = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            #else:
             #   self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.lrelu = nn.Sigmoid()#nn.LeakyReLU()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l1, gain=gain)
        nn.init.xavier_normal_(self.attn_r1, gain=gain)        
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

        
    def forward(self, graph, feat):
        r"""Compute graph attention network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        elist = []
        graph = graph.local_var().to('cuda:{}'.format(feat.get_device()))
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l1).sum(dim=-1).unsqueeze(-1) 
        er = (feat * self.attn_r1).sum(dim=-1).unsqueeze(-1) 
        graph.ndata.update({'ft': feat, 'el': el, 'er': er}) 
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))      
        e = self.leaky_relu(graph.edata.pop('e'))  
        e_soft = edge_softmax(graph, e)
        elist.append(e_soft)
        graph.edata['a'] = self.attn_drop(e_soft)       
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft')) 
        rst = graph.ndata['ft'] 
        if self.activation:
            rst = self.activation(rst)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        return rst, elist

    def forward_batch(self, block, feat):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        elist = []
        block = block.local_var().to('cuda:{}'.format(feat.get_device()))
        h_src = h_dst = self.feat_drop(feat)
        feat_src = self.fc(h_src).view(
            -1, self._num_heads, self._out_feats)
        feat_dst = feat_src[:block.number_of_dst_nodes()] # the first few nodes are dst nodes, as explained in https://docs.dgl.ai/tutorials/large/L1_large_node_classification.html
        el = (feat_src * self.attn_l1).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r1).sum(dim=-1).unsqueeze(-1)
        #block.srcdata.update({'ft': feat, 'el': el, 'er': er})
        block.srcdata.update({'ft': feat_src, 'el': el})
        block.dstdata.update({'er': er})
        block.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(block.edata.pop('e'))
        e_soft = edge_softmax(block, e)
        elist.append(e_soft)
        block.edata['a'] = self.attn_drop(e_soft)
        block.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = block.dstdata['ft']
        if self.activation:
            rst = self.activation(rst)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        return rst, elist
