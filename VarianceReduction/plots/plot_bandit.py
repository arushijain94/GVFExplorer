import os 
import argparse
import numpy as np; np.random.seed(1)
import matplotlib.pyplot as plt

def tsplot(ax, data_x, data_y, x_lim=None, y_lim=None, label=None, **kw):
    # plt.clf()
    from scipy.stats import sem
    est = np.mean(data_y, axis=0)
    sd = sem(data_y, axis=0)
    cis = (est - 1.96 * sd, est + 1.96 * sd)
    # print (data_x.shape, cis[0].shape, cis[1].shape)
    ax.fill_between(data_x,cis[0],cis[1],alpha=0.1, **kw)
    ax.plot(data_x,est, label=label, **kw)
    ax.margins(x=0)
    if x_lim:
        ax.set_xlim(x_lim)
    
    if y_lim:
        ax.set_ylim(y_lim)

def plot_single_run(data, tl, el):
    data = data[','.join([tl, el])]
    for i, plt_type in enumerate(['Entropy', 'Coverage', 'Length']):
        tsplot([x[0] for x in data], [x[1 + i] for x in data], plt_type + '.png')

def plot_aggregated_metrics(data, tl, el):
    data = data[[tl, el]]
    for i, plt_type in enumerate(['Entropy', 'Coverage', 'Length']):
        tsplot([x[0] for x in data], [x[1 + i] for x in data], plt_type + '.png')

def parse_log_file_target(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.replace('\n', '')
            row = line.split(',')
            row = [int(row[0]), float(row[1]), float(row[2]), float(row[3])]
            data.append(row)
    return np.array(data)

def parse_log_file_behavior(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.replace('\n', '')
            row = line.split(',')
            row = [int(row[0]), float(row[1]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])]
            data.append(row)
    return np.array(data)

def read_seeds(base_dir, data_type='target'):
    data = []
    for s in range(1, 6):
        path = os.path.join(base_dir, 'seed_' + str(s) + '/log.csv') 
        if data_type == 'target':
            data.append(parse_log_file_target(path))
        elif data_type == 'behavior':
            data.append(parse_log_file_behavior(path))
    return data

if __name__ == '__main__':
    # plt.style.use('seaborn-darkgrid')
    plot_envs = [#('chain', '_start_2', 6, '20,20', '20,20', 1000, 100, 'ChainMDP'),
                 ('schain', '_start_2', 6, '50,50', '50,50', 1000, 500, 'RiverSwim'), #,
                #  ('grid', '', 5, '50,50', '50,50', 1000, 200, '5x5 GridWorld'), #] #,
                 ('tworooms', '', 9, '100,100', '100,100', 1000, 1000, 'TwoRooms'),
                 ('fourrooms', '', 11, '200,200', '200,200', 1250, 1000, 'FourRooms')]

    plt.rcParams.update({'font.size': 12})

    fig, axs = plt.subplots(1, 4, figsize=(16, 3), constrained_layout=True)
    
    label = 'Target (MC)'
    base_dir = 'logs/complex-two-arms'
    data = np.array(read_seeds(os.path.join(base_dir, 'target'), 'target'))
    for i in range(3):
        tsplot(axs[i], data[0,:,0], data[:,:,i + 1], label=label) 
        
    label = 'BPI'
    data = np.array(read_seeds(os.path.join(base_dir, 'behavior'), 'behavior'))
    for i in range(3):
        tsplot(axs[i], data[0,:,0], data[:,:,i + 1], label=label) 
    tsplot(axs[3], data[0,:,0], data[:,:,-1], label=label) 
    
    order = [0,1]
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend([handles[i] for i in order], [labels[i] for i in order], bbox_to_anchor=(0.5, -0.15), mode="expand", loc='lower center', ncol=2, fontsize=17, bbox_transform = plt.gcf().transFigure)

    axs[0].set_ylabel("MSE", fontsize=14)
    axs[1].set_ylabel("Variance", fontsize=14)
    axs[2].set_ylabel("Bias", fontsize=14)
    axs[3].set_ylabel("Change in Variance", fontsize=14)

    axs[0].set_xlabel("Steps", fontsize=14)
    axs[1].set_xlabel("Steps", fontsize=14)
    axs[2].set_xlabel("Steps", fontsize=14)
    axs[3].set_xlabel("Steps", fontsize=14)

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    
    axs[0].set_ylim(0, 1000)
    axs[1].set_ylim(0, 1000)
    axs[2].set_ylim(-5, 15)
    
    axs[0].set_xlim(0, 1000)
    axs[1].set_xlim(0, 1000)
    axs[2].set_xlim(0, 1000)
    axs[3].set_xlim(0, 1000)
    
    # xlabels = [str(int(x)) + 'K' for x in axs[0].get_xticks()/1000]
    # xlabels[0] = '0'
    # axs[0].set_xticklabels(xlabels)

    # axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fig.savefig('results_bandits.png', dpi=300, bbox_inches="tight")


    plt.cla()
    plt.clf()
    plt.figure(figsize=(4, 3))

    tsplot(plt, data[0,:,0], data[:,:,-2], label='$\mu(a_1)$', color='red')
    tsplot(plt, data[0,:,0], data[:,:,-3], label='$\mu(a_2)$', color='green')

    tsplot(plt, data[0,:,0], np.ones_like(data[:,:,-3]) * 0.1, label='$\pi(a_1)$', color='green', linestyle='dashed')
    tsplot(plt, data[0,:,0], np.ones_like(data[:,:,-2]) * 0.9, label='$\pi(a_2)$', color='red', linestyle='dashed')
    plt.xlim(0, 1000)

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.36), ncol=4, fontsize=10)
    # # print (labels)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Probability of Arm", fontsize=10)
    # fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.15), mode="expand", loc='lower center', ncol=2, fontsize=17, bbox_transform = plt.gcf().transFigure)

    plt.savefig('results_bandits_arms.png', dpi=300, bbox_inches="tight")
