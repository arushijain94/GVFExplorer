import os 
import argparse
import numpy as np; np.random.seed(1)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# def tsplot(ax, data_x, data_y, x_lim=None, y_lim=None, label=None, **kw):
#     plt.clf()
#     from scipy.stats import sem
#     est = np.mean(data_y, axis=0)
#     sd = sem(data_y, axis=0)
#     cis = (est - 1.96 * sd, est + 1.96 * sd)
#     # print (data_x.shape, cis[0].shape, cis[1].shape)
#     ax.fill_between(data_x,cis[0],cis[1],alpha=0.1, **kw)
#     ax.plot(data_x,est, label=label, **kw)
#     ax.margins(x=0)
#     if x_lim:
#         ax.set_xlim(x_lim)
    
#     if y_lim:
#         ax.set_ylim(y_lim)

def plot_single_run(data, tl, el):
    data = data[','.join([tl, el])]
    for i, plt_type in enumerate(['Entropy', 'Coverage', 'Length']):
        tsplot([x[0] for x in data], [x[1 + i] for x in data], plt_type + '.png')

def plot_aggregated_metrics(data, tl, el):
    data = data[[tl, el]]
    for i, plt_type in enumerate(['Entropy', 'Coverage', 'Length']):
        tsplot([x[0] for x in data], [x[1 + i] for x in data], plt_type + '.png')

def parse_log_file_behavior(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            row = line.split(',')
            row = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])]
            data.append(row)
    return np.array(data)

def read_seeds(base_dir):
    data = []
    for s in range(1, 24):
        path = os.path.join(base_dir, 'seed_' + str(s) + '/log.txt') 
        # if data_type == 'target':
        #     data.append(parse_log_file_target(path))
        # elif data_type == 'behavior':
        data.append(parse_log_file_behavior(path))
    return data

def tsplot(ax, x_data, y_data, x_lim=None, y_lim=None, label=None, **kw):
    seeds = np.arange(len(x_data))
    import ipdb
    ipdb.set_trace()
    # for x, y in zip(x_data, y_data):
        # print ('here')
        # ax.plot(x, y)
    # # xa.legend([f"Seed {seed}" for seed in seeds])
    combined_x = np.unique(np.concatenate(x_data))  # Combine and sort x values
    interp_y_data = []

    for x, y in zip(x_data, y_data):
        interp_func = interp1d(x, y, fill_value="extrapolate")  # Interpolate y values on the first set of x values
        interp_y = interp_func(combined_x)
        interp_y_data.append(interp_y)
    
    combined_y = np.array(interp_y_data)
    y_std = np.std(combined_y, axis=0)
    y_mean = np.mean(combined_y, axis=0)
    ax.plot(combined_x, y_mean, label=label)
    ax.fill_between(combined_x, y_mean - y_std, y_mean + y_std, alpha=0.3, label=label)
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Combined Plots with Different X-axis')
    # plt.show()

def getcol(data, ind):
    return [[x[ind] for x in y] for y in data]

if __name__ == '__main__':
    plt.style.use('seaborn-darkgrid')

    plt.rcParams.update({'font.size': 12})

    # col_ids = 
    fig, axs = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
    # Data -> ## [Value Beh, Variances, Biases, MSE]
    
    label = 'Target (MC)'
    base_dir = 'logs/'
    data = np.array(read_seeds(os.path.join(base_dir, 'target')))
    
    for i in range(3):
        tsplot(axs[0,i], getcol(data, 0), getcol(data, 2*(i + 1)), label=label) 
        tsplot(axs[1,i], getcol(data, 0), getcol(data, 2*(i + 1) + 1), label=label) 
        
    label = 'BPI'
    data = np.array(read_seeds(os.path.join(base_dir, 'bpi')))
    for i in range(3):
        tsplot(axs[0,i], getcol(data, 0), getcol(data, 2*(i + 1)), label=label) 
        tsplot(axs[1,i], getcol(data, 0), getcol(data, 2*(i + 1) + 1), label=label) 
    
    label = 'BPG'
    data = np.array(read_seeds(os.path.join(base_dir, 'bpg')))
    # print (data[0][0], getcol(data, 0)[:5])
    
    for i in range(3):
        tsplot(axs[0,i], getcol(data, 0), getcol(data, 2*(i + 1)), label=label) 
        tsplot(axs[1,i], getcol(data, 0), getcol(data, 2*(i + 1) + 1), label=label) 
    
    order = [0,1,2]
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend([handles[i] for i in order], [labels[i] for i in order], bbox_to_anchor=(0.5, -0.1), mode="expand", loc='lower center', ncol=5, fontsize=17, bbox_transform = plt.gcf().transFigure)

    axs[0, 0].set_ylabel("Variance", fontsize=14)
    axs[0, 1].set_ylabel("Bias", fontsize=14)
    axs[0, 2].set_ylabel("MSE", fontsize=14)
    axs[1, 0].set_ylabel("Variance", fontsize=14)
    axs[1, 1].set_ylabel("Bias", fontsize=14)
    axs[1, 2].set_ylabel("MSE", fontsize=14)

    axs[1, 0].set_xlabel("Steps", fontsize=14)
    axs[1, 1].set_xlabel("Steps", fontsize=14)
    axs[1, 2].set_xlabel("Steps", fontsize=14)
    
    axs[0, 0].set_yscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 2].set_yscale('log')
    axs[0, 0].set_xscale('log')
    axs[0, 1].set_xscale('log')
    axs[0, 2].set_xscale('log')

    axs[1, 0].set_yscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 2].set_yscale('log')
    axs[1, 0].set_xscale('log')
    axs[1, 1].set_xscale('log')
    axs[1, 2].set_xscale('log')
    # axs[0, 1].set_yscale('log')
    
    # axs[0].set_ylim(0, 1000)
    # axs[1].set_ylim(0, 1000)
    # axs[2].set_ylim(-5, 15)
    
    axs[0, 0].set_xlim(0, 50000)
    axs[0, 1].set_xlim(0, 50000)
    axs[0, 2].set_xlim(0, 50000)
    axs[1, 0].set_xlim(0, 50000)
    axs[1, 1].set_xlim(0, 50000)
    axs[1, 2].set_xlim(0, 50000)
    # xlabels = [str(int(x)) + 'K' for x in axs[0].get_xticks()/1000]
    # xlabels[0] = '0'
    # axs[0].set_xticklabels(xlabels)

    # axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fig.savefig('results_grid.png', dpi=300, bbox_inches="tight")
