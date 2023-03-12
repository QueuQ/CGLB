import traceback
import pickle
import datetime
from distutils.util import strtobool
from pipeline import *
import sys
import os
dir_home = os.getcwd()
sys.path.append(os.path.join(dir_home,'continual_graph_learning/CGLB')) # for hpc usage
sys.path.append(os.path.join(dir_home,'.local/lib/python3.7/site-packages')) # for hpc usage
from NCGL.visualize import *


if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='G-CGL')
    parser.add_argument('--backbone', type=str, default='GCN', choices=['CusGCN','GCN', 'GAT', 'Weave', 'HPNs'],
                        help='Model to use')
    parser.add_argument('--method', type=str, choices=['bare', 'lwf', 'gem', 'ewc', 'mas', 'twp', 'jointtrain'],
                        default='lwf', help='Method to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['SIDER-tIL','Tox21-tIL','Aromaticity-CL'], default='SIDER-tIL',
                        help='Dataset to use')
    parser.add_argument('--clsIL', type=strtobool, default=False)
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help="which GPU to use.")

    # parameters for different methods
    parser.add_argument('--lwf_args', type=str2dict, default={'lambda_dist': [0.1], 'T': [0.2]})
    parser.add_argument('--twp_args', type=str2dict, default={'lambda_l': 10000., 'lambda_t': 100., 'beta': 0.1})
    parser.add_argument('--ewc_args', type=str2dict, default={'memory_strength': [10000.,1000000.]})
    parser.add_argument('--mas_args', type=str2dict, default={'memory_strength': 10000.})
    parser.add_argument('--gem_args', type=str2dict, default={'memory_strength': 0.5, 'n_memories': 100})
    parser.add_argument('--bare_args', type=str2dict, default={'Na': None})
    parser.add_argument('--joint_args', type=str2dict, default={'Na': None, 'reset_param':False})

    parser.add_argument('-s', '--random_seed', type=int, default=0,
                        help="seed for exp")
    parser.add_argument('--alpha_dis', type=float, default=0.1)
    parser.add_argument('--classifier_increase',default=False,help='this is deprecated, no effect at all')
    #parser.add_argument('--n_cls_per_task', default=1)
    parser.add_argument('--num_epochs',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=10000)
    parser.add_argument('--threshold_pubchem', default=20)
    parser.add_argument('--frac_train', default=0.8)
    parser.add_argument('--frac_val', default=0.1)
    parser.add_argument('--frac_test', default=0.1)
    parser.add_argument('--patience', default=100)
    parser.add_argument('--repeats',type=int, default=3)
    parser.add_argument('--early_stop', type=strtobool, default=False)
    parser.add_argument('--replace_illegal_char', type=strtobool, default=False)
    parser.add_argument('--result_path', type=str, default='./results', help='the path for saving results')
    parser.add_argument('--overwrite_result', type=strtobool, default=True,
                        help='whether to overwrite existing results')

    args = parser.parse_args().__dict__
    args['exp'] = 'config'
    args.update(get_exp_configure(args['exp']))

    method_args = {'lwf': args['lwf_args'], 'twp': args['twp_args'],'jointtrain':args['joint_args'],'jointreplay':args['joint_args'],
                   'ewc': args['ewc_args'], 'bare': args['bare_args'], 'gem': args['gem_args'], 'mas': args['mas_args']}
    hyp_param_list = compose_hyper_params(method_args[args['method']])
    AP_best, name_best = 0, None
    #AP_best, name_best, model_best, hyp_best = 0, None, None, None
    model_dict = {str(hyp_params).replace("'", '').replace(' ', '').replace(',', '_').replace(':', '_'): [] for
                  hyp_params in hyp_param_list}
    AP_dict = {str(hyp_params).replace("'", '').replace(' ', '').replace(',', '_').replace(':', '_'): [] for hyp_params
               in hyp_param_list}
    AF_dict = {str(hyp_params).replace("'", '').replace(' ', '').replace(',', '_').replace(':', '_'): [] for hyp_params
               in hyp_param_list}
    PM_dict = {str(hyp_params).replace("'", '').replace(' ', '').replace(',', '_').replace(':', '_'): [] for hyp_params
               in hyp_param_list}
    for hyp_params in hyp_param_list:
        hyp_params_str = str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_')
        assign_hyp_param(args, hyp_params)
        main = get_pipeline(args)
        if args['dataset'] in ['Aromaticity-CL']:
            args['n_cls_per_task'] = 2
            if args['clsIL']:
                subfolder = f"clsIL/{args['frac_train']}/"
            else:
                subfolder = f'tskIL/{args["frac_train"]}/'
        else:
            args['n_cls_per_task'] = 1
            subfolder = f'tskIL/{args["frac_train"]}/'

        dt = datetime.datetime.now()
        formatted_time = f'{dt.date()}{dt.hour}{dt.minute}'.replace('-', '')
        name = f"{subfolder}val_{args['dataset']}_{args['n_cls_per_task']}_{args['method']}_{list(hyp_params.values())}_{args['backbone']}_{args['gcn_hidden_feats']}_{args['classifier_increase']}_es{args['early_stop']}_p{args['patience']}_bs{args['batch_size']}_{args['num_epochs']}_{args['repeats']}_{formatted_time}"
        args['name'] = name
        mkdir_if_missing(f"{args['result_path']}/" + subfolder)
        if args['replace_illegal_char']:
            name = remove_illegal_characters(name)
        if os.path.isfile(f"{args['result_path']}/{name}.pkl") and not args['overwrite_result']:
            print('the following results exist and will be used \n', f"{args['result_path']}/{name}.pkl")
            AP_AF = show_final_APAF(f"{args['result_path']}/{name}.pkl", GCGL=True) if args.dataset=='Aromaticity-CL' else show_final_APAF_f1(f"{args['result_path']}/{name}.pkl")
            AP = float(AP_AF.split('$')[0])
            if AP > AP_best:
                AP_best = AP
                hyp_best_str = hyp_params_str
                name_best = name
                print(f'best params is {hyp_best_str}, best AP is {AP_best}')
            continue
        else:
            acc_matrices = []
            for ite in range(args['repeats']):
                args['current_model_save_path'] = [name, ite]
                print(name, ite)
                try:
                    AP, AF, acc_matrix = main(args,valid=True)
                    acc_matrices.append(acc_matrix)
                    AP_dict[hyp_params_str].append(AP)
                    torch.cuda.empty_cache()
                    if ite == 0:
                        with open(
                                f"{args['result_path']}/log.txt", 'a') as f:
                            f.write(name)
                            f.write('\nAP:{},AF:{}\n'.format(AP, AF))
                except Exception as e:
                    mkdir_if_missing(f"{args['result_path']}/errors/" + subfolder)
                    if ite > 0:
                        name_ = f"{subfolder}val_{args['dataset']}_{args['n_cls_per_task']}_{args['method']}_{list(hyp_params.values())}_{args['backbone']}_{args['gcn_hidden_feats']}_{args['classifier_increase']}_bs{args['batch_size']}_{args['num_epochs']}_{ite}"
                        with open(
                                f"{args['result_path']}/{name_}.pkl", 'wb') as f:
                            pickle.dump(acc_matrices, f)
                    print('error', e)
                    name = 'errors/{}'.format(name)
                    acc_matrices = traceback.format_exc()
                    print(acc_matrices)
                    print('error happens on \n', name)
                    torch.cuda.empty_cache()
                    break
            if np.mean(AP_dict[hyp_params_str]) > AP_best:
                AP_best = np.mean(AP_dict[hyp_params_str])
                hyp_best_str = hyp_params_str
                name_best = name
            print(f'best params is {hyp_best_str}, best AP is {AP_best}')
            with open(f"{args['result_path']}/{name}.pkl", 'wb') as f:
                pickle.dump(acc_matrices, f)

    # save the models
    config_name = name_best.split('/')[-1]
    subfolder_c = name_best.split(config_name)[-2]
    print('----------Now in testing--------')
    acc_matrices = []
    for r in range(args['repeats']):
        args['current_model_save_path'] = [name_best, r]
        AP_test, AF_test, acc_matrix_test = main(args, valid=False)
        acc_matrices.append(acc_matrix_test)
    with open(f"{args['result_path']}/{name_best}.pkl".replace('val', 'te'), 'wb') as f:
        pickle.dump(acc_matrices, f)


