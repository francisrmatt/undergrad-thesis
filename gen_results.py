import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import constants
import xarray as xr
import numpy as np
import matplotlib.transforms as transforms


which_to_name = {
    'new_regular2' : 'Naive Training',
    'new_freqnoise' : 'Noisy Training',
    'new_synthetic' : 'General Purpose Training',
    'new_const_scale_more' : 'Single-scale Training',
    'tmp_noisy_test' : 'Noisy training with Cosine Onecycle',
    'final_noise' : 'Noisy training with Adam',
    'final_naive' : 'Naive training with Adam',
    'final_naive_cos' : 'Naive training with Cosine Onecycle',
    'final_noise_cos' : 'Noisy training with Cosine Onecycle',
    'rpe_noisy_cos' : 'Noisy training with RoPE Embedding',
    'stream_noise_rpe_big' : 'final model',
    'stream_noise_rpe_small' : 'final model (small)',
}

def gen_single_scale_over_snr(which : str,
         scale_set : float):

    da = xr.open_dataarray(f'params/{which}/results.nc')

    da.name = 'name'
    df = da.to_dataframe()
    df['name'].loc[:, scale_set]

    dfs = df['name'].loc[:, scale_set]
    foo = pd.DataFrame(dfs)
    ax = sns.boxplot(data = foo, x = 'snr',y = 'name', showfliers=False, showmeans=True, 
                    meanprops={'markersize': 10,"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", 'label':'Transformer Mean'},
                    boxprops=dict(facecolor='white'))
    nxt = constants.SNR_SET
    nxt[-1] = r'$\infty$'
    ax.set_xticklabels(nxt)

    # From  https://stackoverflow.com/questions/43126064/how-do-i-shift-categorical-scatter-markers-to-left-and-right-above-xticks-multi
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    ax.set_title(f'Compression rate for {which_to_name.get(which, 'Model')} at scale {scale_set}', fontsize = 16)
    ax.set_xlabel('SNR [dB]', fontsize = 14)
    ax.set_ylabel('Compression Rate', fontsize = 14)
    trans = plt.gca().transData

    # Add gzip means
    ax.scatter([0,1,2,3,4], constants.GZIP_RESULTS.loc[:, scale_set], marker = 's', edgecolors = 'red', facecolors='white', zorder = 5, label = 'Gzip Mean',
                 transform = trans+offset(-5))
    ax.scatter([0,1,2,3,4], constants.AC_RESULTS.loc[:, scale_set], marker = 's', edgecolors = 'orange', facecolors='white', zorder = 5, label = 'AC Mean',
               transform = trans+offset(5))
    #ax.set_ylim(constants.GZIP_RESULTS.loc[:, scale_set].min()-0.02, None)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-3:], labels[-3:])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals = 0))

    plt.savefig(f'params/{which}/results_over_noise_scale{scale_set}.png')
    plt.savefig(f'params/{which}/results_over_noise_scale{scale_set}.eps', format = 'eps', dpi = 1200)
    plt.close()

def compare_dists(d1, d2, noise, scale):
    da1 = xr.open_dataarray(f'params/{d1}/results.nc')

    da1.name = 'name'
    df1 = da1.to_dataframe()
    dfs1 = df1['name'].loc[noise, scale]
    dd1 = pd.DataFrame(dfs1)

    da2 = xr.open_dataarray(f'params/{d2}/results.nc')
    da2.name = 'name'
    df2 = da2.to_dataframe()
    dfs2 = df2['name'].loc[noise, scale]
    dd2 = pd.DataFrame(dfs2)




def gen_single_noise_over_scale(
        which : str,
        noise_set : int):

    da = xr.open_dataarray(f'params/{which}/results.nc')

    da.name = 'name'
    df = da.to_dataframe()
    dfs = df['name'].loc[noise_set, :]
    foo = pd.DataFrame(dfs)
    ax = sns.boxplot(data = foo, x = 'scale',y = 'name', showfliers=False, showmeans=True, 
                    meanprops={'markersize': 10,"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", 'label':'Transformer Mean'},
                    boxprops=dict(facecolor='white'))
    #nxt = constants.SNR_SET
    #nxt[-1] = r'$\infty$'
    #ax.set_xticklabels(nxt)

    # From  https://stackoverflow.com/questions/43126064/how-do-i-shift-categorical-scatter-markers-to-left-and-right-above-xticks-multi
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    nsname = r'$\infty$' if noise_set == 999 else str(noise_set)
    ax.set_title(f'Compression rate for {which_to_name[which]} at SNR {nsname} dB', fontsize = 16)
    ax.set_xlabel('Scale', fontsize = 14)
    ax.set_ylabel('Compression Rate', fontsize = 14)
    trans = plt.gca().transData

    # Add gzip means
    ax.scatter([0,1,2,3,4], constants.GZIP_RESULTS.loc[noise_set, ::-1], marker = 's', edgecolors = 'red', facecolors='white', zorder = 5, label = 'Gzip Mean',
                 transform = trans+offset(-5))
    ax.scatter([0,1,2,3,4], constants.AC_RESULTS.loc[noise_set, ::-1], marker = 's', edgecolors = 'orange', facecolors='white', zorder = 5, label = 'AC Mean',
               transform = trans+offset(5))

    ax.set_ylim(foo['name'].min(), constants.GZIP_RESULTS.loc[noise_set, :].max() + 0.08)
    ax.relim()
    ax.autoscale_view()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-3:], labels[-3:])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals = 0))

    plt.savefig(f'params/{which}/results_over_scale_noise{noise_set}.png')
    plt.savefig(f'params/{which}/results_over_scale_noise{noise_set}.eps', format = 'eps', dpi = 1200)
    plt.close()


def gen_single_violin(which : str,
                      snr : int,
                      scale : float,
                      overlay : bool):

    da = xr.open_dataarray(f'params/{which}/results.nc')
    da.name = 'name'
    df = da.to_dataframe()

    snrstr = r'$\infty$' if snr == 999 else str(snr)
    dfs = df['name'].loc[snr, scale]
    foo = pd.DataFrame(dfs)

    ax = sns.violinplot(data = foo, y= 'name',  inner = None)
    ax.set_title(f'Compression Rate Distribution for {which_to_name[which]} at Scale {scale}')
    ax.set_xlabel(f'SNR {snrstr} dB')
    ax.set_ylabel('Compression Rate')
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals = 0))

    if overlay:
        ax.scatter([0], constants.GZIP_RESULTS.loc[snr, scale], marker = 's', edgecolors = 'red', facecolors='white', zorder = 5, label = 'Gzip Mean')
        ax.scatter([0], constants.AC_RESULTS.loc[snr, scale], marker = 's', edgecolors = 'orange', facecolors='white', zorder = 5, label = 'AC Mean')
        ax.scatter([0], foo.mean(), marker = 's', s = 100,edgecolors = 'blue', facecolors='white', zorder = 5, label = 'Transformer Mean')
        ax.legend()


    plt.savefig(f'params/{which}/violin_snr{snr}_scale{scale}.png')
    plt.savefig(f'params/{which}/violin_snr{snr}_scale{scale}.eps', format = 'eps', dpi = 1200)
    plt.close()


gen_single_noise_over_scale(which = 'stream_noise_rpe_big',
                            noise_set = 30)

gen_single_scale_over_snr(which = 'stream_noise_rpe_big', 
                          scale_set = 1.0)

#gen_single_scale_over_snr(which = 'stream_noise_rpe_small', 
#                          scale_set = 1.0)

#gen_single_scale_over_snr(which = 'stream_noise_rpe_small', 

                          #scale_set = 1.0)

#gen_single_violin(which = 'final_naive',
                  #snr = 20,
                  #scale = 1.0,
                  #overlay = True)


#gen_single_noise_over_scale(which = 'tmp_noisy_test',
                            #noise_set = 40)


#gen_single_scale_over_snr(which = 'final_naive', 
#                          scale_set = 1.0)
#gen_single_scale_over_snr(which = 'final_naive_cos', 
#                          scale_set = 1.0)