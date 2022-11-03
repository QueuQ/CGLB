# graph construction
from dgllife.utils import smiles_to_bigraph
# node featurization
from dgllife.utils import CanonicalAtomFeaturizer

config = {
    'random_seed': 2,
    'lr': 1e-3,
    'node_data_field': 'h',
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'predictor_hidden_feats': 64,
    'patience': 10,
    'smiles_to_graph': smiles_to_bigraph,
    'node_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc_score'
}

def get_exp_configure(exp_name):
    return config