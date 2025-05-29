import numpy as np
import matplotlib.pyplot as plt

def plot_rt_distribution(RT_sim_corr, RT_sim_incorr, RT_dice_corr, RT_dice_incorr, save_path=None, show_plot=True, figsize=(8, 6), dpi=300, colors=None):
    plt.style.use('seaborn-v0_8-paper') 
    
    colors_dict = {
        'sim_correct': '#2166AC',
        'sim_incorrect': '#B2182B',
        'dice_correct': '#92C5DE',
        'dice_incorrect': '#EF8A62' 
    }
    if colors is not None:
        colors_dict.update(colors)

    RT_sim_corr_sec = RT_sim_corr / 1000
    RT_sim_incorr_sec = RT_sim_incorr / 1000

    quantiles = np.linspace(0, 100, 100)
    q_corr = np.percentile(RT_sim_corr_sec , quantiles)  
    q_incorr = np.percentile(RT_sim_incorr_sec, quantiles)

    q_pos = np.percentile(RT_dice_corr, quantiles)
    q_neg = np.percentile(RT_dice_incorr, quantiles)
    
    def compute_ecdf(data):
        data = data[~np.isnan(data)]
        sorted_data = np.sort(data)
        n = sorted_data.size
        ecdf = np.arange(1, n + 1) / n
        return sorted_data, ecdf
    
    pos_x_sim, pos_ecdf_sim = compute_ecdf(RT_sim_corr_sec)
    neg_x_sim, neg_ecdf_sim = compute_ecdf(RT_sim_incorr_sec)
    pos_x_dice, pos_ecdf_dice = compute_ecdf(RT_dice_corr)
    neg_x_dice, neg_ecdf_dice = compute_ecdf(RT_dice_incorr)
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=dpi)
    
    all_rts = np.concatenate([
        RT_sim_corr_sec[~np.isnan(RT_sim_corr_sec)],
        RT_dice_corr[~np.isnan(RT_dice_corr)],
        RT_sim_incorr_sec[~np.isnan(RT_sim_incorr_sec)],
        RT_dice_incorr[~np.isnan(RT_dice_incorr)]
    ])
    min_rt = np.min(all_rts)
    max_rt = np.max(all_rts)
    num_bins = 25  
    common_bins = np.linspace(min_rt, max_rt, num_bins + 1)
    
    hist_params = {
        'bins': common_bins,
        'density': True,
        'alpha': 0.7,
        'edgecolor': 'black',
        'linewidth': 0.5
    }
    
    ax = axs[0, 0]
    ax.hist(RT_sim_corr_sec, color=colors_dict['sim_correct'], label='Neural circuit model', **hist_params)
    ax.hist(RT_dice_corr, color=colors_dict['dice_correct'], label='Discrete simulation', **hist_params)
    ax.set_xlabel('Reaction Time (s)', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.set_title('Correct Responses', fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = axs[0, 1]
    ax.step(pos_x_sim, pos_ecdf_sim, where='post', color=colors_dict['sim_correct'], 
            label='Simulation', linewidth=2)
    ax.step(pos_x_dice, pos_ecdf_dice, where='post', color=colors_dict['dice_correct'], 
            label='Dice', linewidth=2)
    ax.set_xlabel('Reaction Time (s)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title('Correct Responses CDF', fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    axs[0, 2].plot(q_pos, q_corr, 'o', color=colors_dict['sim_correct'])
    axs[0, 2].plot([min(q_pos.min(), q_corr.min()), max(q_pos.max(), q_corr.max())],
                [min(q_pos.min(), q_corr.min()), max(q_pos.max(), q_corr.max())],
                'r--')
    axs[0, 2].set_title("QQ plot of correct trials", fontsize=12, pad=10) 
    axs[0, 2].set_xlabel("Quantiles of RT_dice")
    axs[0, 2].set_ylabel("Quantiles of RT_sim")
    axs[0, 2].grid(False)
    
    ax = axs[1, 0]
    ax.hist(RT_sim_incorr_sec, color=colors_dict['sim_incorrect'], label='Neural circuit model', **hist_params)
    ax.hist(RT_dice_incorr, color=colors_dict['dice_incorrect'], label='Discrete simulation', **hist_params)
    ax.set_xlabel('Reaction Time (s)', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.set_title('Incorrect Responses', fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = axs[1, 1]
    ax.step(neg_x_sim, neg_ecdf_sim, where='post', color=colors_dict['sim_incorrect'], 
            label='Neural circuit model', linewidth=2)
    ax.step(neg_x_dice, neg_ecdf_dice, where='post', color=colors_dict['dice_incorrect'], 
            label='Discrete simulation', linewidth=2)
    ax.set_xlabel('Reaction Time (s)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title('Incorrect Responses CDF', fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    axs[1, 2].plot(q_neg, q_incorr, 'o', color=colors_dict['sim_correct'])
    axs[1, 2].plot([min(q_neg.min(), q_incorr.min()), max(q_neg.max(), q_incorr.max())],
                [min(q_neg.min(), q_incorr.min()), max(q_neg.max(), q_incorr.max())],
                'r--')
    axs[1, 2].set_title("QQ plot of incorrect trials", fontsize=12, pad=10) 
    axs[1, 2].set_xlabel("Quantiles of RT_dice")
    axs[1, 2].set_ylabel("Quantiles of RT_sim")
    axs[1, 2].grid(False)
    
    plt.tight_layout(pad=2.0)
    
    for i, ax_item in enumerate(axs.flat):
        ax_item.text(-0.2, 1.1, chr(65 + i), transform=ax_item.transAxes, 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
    
    if show_plot:
        plt.show()
    else:
        plt.close() 