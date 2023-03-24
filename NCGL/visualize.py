import pickle
import matplotlib.pyplot as plt
import numpy as np

def AP_err(performance_matrices):
    # given a list of performence matrices, return the APs the errors (range), and std
    n_tasks = performance_matrices[0].shape[0]
    performance_means_all_repeats = np.stack([[np.mean(m[i,0:i+1]) for i in range(n_tasks)] for m in performance_matrices])
    performance_means = np.mean(performance_means_all_repeats, axis=0)
    err_all = performance_means_all_repeats - performance_means
    std = performance_means_all_repeats.std(0)
    err_plus = err_all.max(axis=0)
    err_minus = err_all.min(axis=0).__abs__()
    err = np.stack([err_minus, err_plus])
    return performance_means, err, std

def AF(acc_matrix):
    # given a acc matrix, return AF
    n_tasks = acc_matrix.shape[0]
    backward = []
    for t in range(n_tasks - 1):
        b = acc_matrix[n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(b)
    return np.mean(backward)

def AF_err(performance_matrices):
    # given a list of acc matrices, return the AMs and the errors
    AF_all_repeats = []
    for m in performance_matrices:
        AF_all_repeats.append(AF(m))
    AF_mean = np.mean(AF_all_repeats)
    AF_all_repeats = np.stack(AF_all_repeats)
    err_all = AF_all_repeats - AF_mean
    std = AF_all_repeats.std(0)
    err_plus = err_all.max(axis=0)
    err_minus = err_all.min(axis=0).__abs__()
    err = np.stack([err_minus, err_plus])
    return AF_mean, err, std

def show_performance_matrices(result_path, save_fig_name=None, multiplier=1.0):
    """
    The function to visualize the performance matrix.

    :param result_path: The path to the experimental result
    :param save_fig_name: If specified, the generated visualization will be stored with the specified name under the directory "./results/figures"
    """
    # visualize the acc matrices
    print(result_path)
    fig, ax = plt.subplots()
    performance_matrices = pickle.load(open(result_path, 'rb'))
    acc_matrix_mean = np.mean(performance_matrices, axis=0)
    mask = np.tri(acc_matrix_mean.shape[0], k=-1).T
    acc_matrix_mean = np.ma.array(acc_matrix_mean, mask=mask) * multiplier
    im = plt.imshow(acc_matrix_mean)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel('$\mathrm{Tasks}$')
    plt.ylabel('$\mathrm{Tasks}$')
    plt.clim(vmin=0, vmax=100)
    cbar = fig.colorbar(im, ticks=[0, 50, 100])  # , fontsize = 15)
    cbar.ax.tick_params()

    if save_fig_name is not None:
        plt.savefig(f'./results/figures/{save_fig_name}_performance_matrix', bbox_inches='tight')
    plt.show()

def show_learning_curve(result_path, save_fig_name=None):
    """
        The function to visualize the dynamics of AP.

        :param result_path: The path to the experimental result
        :param save_fig_name: If specified, the generated visualization will be stored with the specified name under the directory "./results/figures"
        """
    #to draw AP against buffer task with different methods
    print(result_path)
    performance_matrices = pickle.load(open(result_path, 'rb'))
    performance_mean, err, _ = AP_err(performance_matrices)
    x = list(range(len(performance_mean)))
    plt.errorbar(x, performance_mean)
    if save_fig_name is not None:
        plt.savefig(
            f'./results/figures/{save_fig_name}_learning_curve', bbox_inches='tight')
    plt.show()

def show_final_APAF(result_path, GCGL=False):
    """
        The function to show the final AP and AF. Output are orgnized in a LaTex firendly way.

        :param result_path: The path to the experimental result
        """
    #show the final AP and AF
    performance_matrices = pickle.load(open(result_path, 'rb'))
    performance_mean, err, std_am = AP_err(performance_matrices)

    # AF
    AF_mean, err_AF, std_AF = AF_err(performance_matrices)

    if not GCGL:
        output_str=r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF)
        print(r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF))
    else:
        output_str=r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1]*100, std_am[-1]*100, AF_mean*100, std_AF*100)
        print(r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1]*100, std_am[-1]*100, AF_mean*100, std_AF*100)) # convert GCGL results to percentages
    return output_str


def show_final_APAF_f1(result_path):
    #show the final AP and AF for results in the form of f1 score
    performance_matrices = pickle.load(open(result_path, 'rb'))
    performance_mean, err, std_am = AP_err(performance_matrices)

    # AF
    AF_mean, err_AF, std_AF = AF_err(performance_matrices)

    output_str = r'{:.3f}$\pm${:.3f}&{:.3f}$\pm${:.3f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF)
    print(r'{:.3f}$\pm${:.3f}&{:.3f}$\pm${:.3f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF))
    return output_str