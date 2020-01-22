from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from logging_formatter import Logger
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import six
logger = Logger()

marker_index = 0
markers      = ["*-", "o-", "^-", "s-","P-", "X-","D-"]

def plot_graph(lambdas, outDir, bpp, exps_metric, exp_labeles, xaxis, yaxis, title):
    global marker_index
    global markers
    #plt.cla()
    exps_names = ''
    add_lambdas = False
    plt.figure(figsize=(14, 8))
    draw_upper_bounds = False
    marker_index = 0

    for xs, ys,l,lmb in zip(bpp, exps_metric, exp_labeles, lambdas):
        if l == 'seg_com_reco_rgb_fixedlmbda2048':
            l = 'lambda_2048'
        if l == 'seg_com_reco_rgb_fixedlmbda512':
            l = 'lambda_512'
        if l == 'seg_com_reco_rgb_fixedlmbda8192':
            l = 'lambda_8192'
        if l == 'seg_com_reco_rgb_fixedlmbda128':
            l = 'lambda_128'
        if l == 'seg_com_reco_rgb_fixedlmbda16384':
            l = 'lambda_16384'
            
        if l == 'exp_synthiaSF_baseline_128':
            l = 'rgb'
        elif l == 'exp_synthiaSF_depth_128_concatAtStart':
            l = 'rgb_d_concat'
        elif l == 'exp_synthiaSF_depth_128_fusionAfterGDN1':
            l = 'rgb_d_afterGDN'
        elif l == 'exp_synthiaSF_depth_128_fusionAfterGDN1_allrelu':
            l = 'rgb_d_afterGDN_allrelu'
        elif l == 'exp_synthiaSF_depth_128_fusionAfterGDN1_relu':
            l = 'rgb_d_afterGDN_relu'
        elif l == 'exp_synthiaSF_depth_128_fusionAfterGDN1_SSMA_noBatchNorm':
            l = 'rgb_d_SSMA'
        elif l == 'exp_synthiaSF_depth_128_fusionAfterGDN1_f32_corr':
            l = 'rgb_d_afterGDN_32f'

        if l == 'seg_com_depth':
            l = 'w_comp_seg_depth'
            c = 'g'
        elif l == 'seg_com_rgb':
            l = 'lambda_0' #l = 'w_comp_seg_rgb'
            c = 'b'
            draw_upper_bounds = True
        elif l == 'seg_com_rgb_d':
            l = 'w_comp_seg_rgb_d'
            c = 'r'
            draw_upper_bounds = True
        elif l == 'seg_arch_for_compAndReconsrurction':
             l = 'beta_0'
        elif l == 'w_comp_seg_rgb_d_downby2':
            c = 'tab:purple'
        else:
            c = None
        marker = markers[marker_index%len(markers)]
        marker_index += 1
        if l == 'lambda_0':
            plt.plot(xs, ys, marker, label=l, color =c, linewidth=3)
        else:
            plt.plot(xs, ys, marker[0]+':', label=l, color=c)

        if add_lambdas:
            for x, y, t, in zip(xs,ys,lmb):
                plt.text(x, y, t, fontsize=9,verticalalignment='bottom',horizontalalignment='right')

        exps_names +='_'+l
    save_dir = os.path.join(outDir, 'plots'+exps_names)

    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    xmax = 2.1
    #plt.hlines(y=0.560, xmin=0, xmax=xmax, linestyles='dashdot', colors='brown', label='lambda_0')
    draw_upper_bounds = False
    if draw_upper_bounds:
        xmax = 2.1
        plt.hlines(y=0.312, xmin=0, xmax=xmax, linestyles = 'dotted', colors='g',label='wo_comp_seg_depth')
        plt.hlines(y=0.560, xmin=0, xmax=xmax, linestyles = 'dotted', colors='b',label='wo_comp_seg_rgb')
        plt.hlines(y=0.590, xmin=0, xmax=xmax, linestyles = 'dotted', colors='r',label='wo_comp_seg_rgb_d')

    plot_im_path = os.path.join(save_dir, title + '.png')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('')
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    #plt.savefig(plot_im_path)
    plt.show()

    return save_dir


def plot_graphs(expsDir,outDir):
    """
    Plot and save four graphs: bpp vs [PSNR [dB], MS-SSIM, MS-SSIM [dB], MSE]
    :param expsDir:
    :return:
    """
    global marker_index
    lambdas, bpp, MS_SSIM, MS_SSIM_db, PSNR, MSE, mIOU, exp_names = [], [], [], [], [], [], [], []

    logger.info('Plotting graphs:')

    for expDir in expsDir:
        lambda_dirs = os.listdir(expDir)

        lambda_files_data = []
        bpp.append([])
        MS_SSIM.append([])
        MS_SSIM_db.append([])
        PSNR.append([])
        MSE.append([])
        lambdas.append([])
        mIOU.append([])

        lambda_dirs.sort(key=lambda i: float(i.split('_')[1]))
        logger.info(expDir + str(lambda_dirs))

        for lambda_dir in lambda_dirs:
            #if lambda_dir == 'lambda_512000.0':
            #    continue
            lambda_file_path = os.path.join(expDir, lambda_dir, 'metrics_args.pkl')

            lmbda = float(lambda_dir.split('_')[1])

            try:
                with open(lambda_file_path, "rb") as fp:
                    data = pickle.load(fp)
                    lambda_files_data.append(data)
                    bpp[-1].append(data['exp_avg_metrics']['bpp'])
                    MS_SSIM[-1].append(data['exp_avg_metrics']['msssim'])
                    MS_SSIM_db[-1].append(data['exp_avg_metrics']['msssim_db'])
                    PSNR[-1].append(data['exp_avg_metrics']['psnr'])
                    MSE[-1].append(  data['exp_avg_metrics']['mse']  )
                    mIOU[-1].append( data['exp_avg_metrics']['mIOU'] )
                    lambdas[-1].append(str(lmbda))
                    #print('[',lmbda, data['exp_avg_metrics']['mse'])
            except Exception as e:
                logger.warn('Failed to load '+lambda_file_path)

        #exp_names.append(data['exp_args'].exp_name)
        name = os.path.split(os.path.split(expDir)[0])[1]
        print(expDir,'  -   ',name)
        exp_names.append(name)
    print('lambdas: ', lambdas)
    print('bpp: ', bpp)
    print('mIOU: ', mIOU)



    save_dir = plot_graph(lambdas, outDir, bpp, MSE, exp_names, 'bpp', 'MSE', 'bppVsMSE')
    save_dir = plot_graph(lambdas, outDir, bpp, MS_SSIM, exp_names, 'bpp', 'MS_SSIM', 'bppVsMSSSIM')
    save_dir = plot_graph(lambdas, outDir, bpp, MS_SSIM_db, exp_names, 'bpp', 'MS_SSIM [dB]', 'bppVsMSSSIMdB')
    save_dir = plot_graph(lambdas, outDir, bpp, PSNR, exp_names, 'bpp', 'PSNR [dB]', 'bppVsPSNRdB')
    save_dir = plot_graph(lambdas, outDir, bpp, mIOU, exp_names, 'bpp', 'mIOU', 'bppVsmiou')
    logger.info('Saving plots in ' + save_dir)



    # ================== results table
    cols = ['','Experiment', 'lambda','bpp', 'mIOU', 'MS_SSIM [dB]', 'MS_SSIM', 'PSNR', 'MSE']

    table = [cols]
    for i in range(len(exp_names)):
        for j in range(len(lambdas[i])):
            table.append( [i*len(exp_names)+j, exp_names[i] , lambdas[i][j],  bpp[i][j],mIOU[i][j],
                           MS_SSIM_db[i][j],MS_SSIM[i][j], PSNR[i][j], MSE[i][j] ] )

    for i in range(1,len(table)):
        for j in range(2,len(table[i])):
            table[i][j] = np.array(float(table[i][j])).round(decimals=3)

    data = np.array(table)

    table_df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
    print(table_df)
    _ , results_fig = render_mpl_table(table_df, header_columns=0, col_width=4)

    results_fig.savefig( os.path.join(save_dir, 'results.png'))
    table_df.to_csv(os.path.join(save_dir, 'results.csv'))

    return table_df


def render_mpl_table(data, col_width=1.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    logger.info(str(data.columns))

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns,
                         colWidths = [3.5,1.2,1.2,1.2,1.2,1.2,1.2,1.2], cellLoc='left', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)

        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell._loc = 'center'
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

        if k[1] == 0:
            cell._loc = 'left'

    return ax, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('expsDir', nargs='+')
    parser.add_argument("--outdir", type=str, default='experiments/',help="")
    args = parser.parse_args()
    plot_graphs(args.expsDir, args.outdir)
