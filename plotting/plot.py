import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Import plotly for making manhattan plots
import plotly.express as px
import plotly.graph_objects as go

import os
import warnings

def manhattan_plot(pval, 
                pos=None, 
                chrom=None,
                beta=None,
                cutoff='bonferroni',
                scale='log',
                ax=None,
                cmap=None,
                plotly=False,
                save_path=None):
    """
    Generate a Manhattan plot to visualize genome-wide association study (GWAS) results.

    Parameters:
    pval (array-like): Array of p-values for each SNP.
    pos (array-like, optional): Array of SNP positions. If not provided, the order of p-values will be used.
    chrom (array-like, optional): Array of chromosome numbers for each SNP. If not provided, it will be assumed that all SNPs are on the same chromosome.
    beta (array-like, optional): Array of effect sizes (beta values) for each SNP.
    cutoff (float or str, optional): Cutoff value for significance. If a number is provided, it will be used as the cutoff. If 'bonferroni' is provided, the Bonferroni correction will be applied. If 'gw' is provided, a genome-wide significance threshold of 5e-8 will be used. Default is 'bonferroni'.
    scale (str, optional): Scale of the y-axis. Can be 'log' for -log10(p-values) or 'linear' for raw p-values. Default is 'log'.
    ax (matplotlib Axes, optional): Axes object to plot on. If not provided, a new figure will be created.
    cmap (str or matplotlib colormap, optional): Colormap for coloring the SNPs by chromosome. Default is None, which uses the default seaborn color palette.
    plotly (bool, optional): If True, the plot will be generated using Plotly and returned as a Plotly figure object. If False, the plot will be generated using Matplotlib and returned as a tuple of the figure and axes objects. Default is False.

    Returns:
    If plotly=True, returns a Plotly figure object.
    If plotly=False, returns a tuple of the Matplotlib figure and axes objects.
    """

    # If trying to save and providing ax, warn user but continue
    if isinstance(save_path, str) and ax is not None:
        warnings.warn('Providing ax with save_path will only save the last plot', UserWarning)

    if scale == 'log':
        pval = -np.log10(np.clip(pval, 
                                a_min=1e-300, 
                                a_max=1))
    elif scale == 'linear' or scale is None:
        pass

    if chrom is None and pos is None:
        # Use order of pvals
        chrom = np.zeros_like(pval)
        pos = np.arange(len(pval))

    elif chrom is None and pos is not None:
        # Assume all on same chromosome
        chrom = np.zeros_like(pos)
    elif chrom is not None and pos is None:
        # Assume all on same chromosome
        pos = np.arange(len(chrom))

        # If chrom is not a np array, convert to int and then fill an array with it
        if not isinstance(chrom, np.ndarray):
            chrom = np.array(chrom).astype(int)

    # Manhatten plot adapted from https://stackoverflow.com/a/66062857
    results_df = pd.DataFrame(
        {
        'pos'  : pos,
        'pval' : pval,
        'chr' : chrom
        }
    )

    if beta is not None:
        results_df['beta'] = beta

    results_df = results_df.sort_values(['chr', 'pos'])
    results_df.reset_index(inplace=True, drop=True)
    results_df['i'] = results_df.index

    # If cutoff is a number, use that as the cutoff
    if cutoff is not None and isinstance(cutoff, (int, float)):
        alpha = cutoff
    else:
        if cutoff is None:
            alpha = 0.05
        elif cutoff == 'gw':
            alpha = 5e-8
        elif cutoff == 'bonferroni':
            alpha = np.clip(0.05/len(pval), a_min=1e-300, a_max=1)
        else:
            raise ValueError('Invalid value for cutoff')

        # If log scale, take -log10
        if scale == 'log':
            alpha = -np.log10(alpha)
        else:
            alpha = alpha

    if plotly:
        if cmap is None:
            # Use default plotly colors
            cmap = px.colors.qualitative.Plotly
        else:
            if isinstance(cmap, str):
                cmap = px.colors.qualitative[cmap]
            else:
                raise ValueError('Invalid value for cmap')

        chrom_df=results_df.groupby('chr')['i'].median()
        
        # Scatterplot of all SNPs as background
        # First layer should not have hover information
        fig = go.Figure()
        
        for chrom in results_df['chr'].unique():
            chrom_snps = results_df[results_df['chr'] == chrom]
            fig.add_trace(
                go.Scatter(x=chrom_snps['i'],
                            y=chrom_snps['pval'],
                            mode='markers',
                            marker=dict(color=cmap[chrom%len(cmap)],
                                        showscale=False,
                                        line=dict(width=0)),
                            hoverinfo='skip')
            )

        # Annotate SNPs that pass the cutoff
        if scale == 'log':
            sig_snps = results_df[results_df['pval'] >= alpha]
        else:
            sig_snps = results_df[results_df['pval'] <= alpha]

        # Then add layer of SNPs that pass the cutoff as interactive
        if beta is not None:
            if scale == 'log':
                # For each chromosome, add a trace with color matching that chromosome from first layer
                for chrom in sig_snps['chr'].unique():
                    chrom_snps = sig_snps[sig_snps['chr'] == chrom]
                    fig.add_trace(
                        go.Scatter(x=chrom_snps['i'],
                                    y=chrom_snps['pval'],
                                    mode='markers',
                                    marker=dict(color=cmap[chrom%len(cmap)],
                                                showscale=False,
                                                line=dict(width=0)),
                                    hoverinfo='text',
                                    hovertext=chrom_snps['chr'].astype(str) + ':' + chrom_snps['pos'].astype(str) + '<br>' + 'beta: ' + chrom_snps['beta'].apply("{:.2e}".format) + '<br>' + '-log10(p): ' + chrom_snps['pval'].apply("{:.2}".format)
                        )
                    )
            else:
                # For each chromosome, add a trace with color matching that chromosome from first layer
                for chrom in sig_snps['chr'].unique():
                    chrom_snps = sig_snps[sig_snps['chr'] == chrom]
                    fig.add_trace(
                        go.Scatter(x=chrom_snps['i'],
                                    y=chrom_snps['pval'],
                                    mode='markers',
                                    marker=dict(color=cmap[chrom%len(cmap)],
                                                showscale=False,
                                                line=dict(width=0)),
                                    hoverinfo='text',
                                    hovertext=chrom_snps['chr'].astype(str) + ':' + chrom_snps['pos'].astype(str) + '<br>' + 'beta: ' + chrom_snps['beta'].apply("{:.2e}".format) + '<br>' + 'pval: ' + chrom_snps['pval'].apply("{:.2e}".format)
                        )
                    )
        else:
            if scale == 'log':
                # For each chromosome, add a trace with color matching that chromosome from first layer
                for chrom in sig_snps['chr'].unique():
                    chrom_snps = sig_snps[sig_snps['chr'] == chrom]
                    fig.add_trace(
                        go.Scatter(x=chrom_snps['i'],
                                    y=chrom_snps['pval'],
                                    mode='markers',
                                    marker=dict(color=cmap[chrom%len(cmap)],
                                                showscale=False,
                                                line=dict(width=0)),
                                    hoverinfo='text',
                                    hovertext=chrom_snps['chr'].astype(str) + ':' + chrom_snps['pos'].astype(str) + '<br>' + '-log10(p): ' + chrom_snps['pval'].apply("{:.2}".format)
                        )
                    )
            else:
                # For each chromosome, add a trace with color matching that chromosome from first layer
                for chrom in sig_snps['chr'].unique():
                    chrom_snps = sig_snps[sig_snps['chr'] == chrom]
                    fig.add_trace(
                        go.Scatter(x=chrom_snps['i'],
                                    y=chrom_snps['pval'],
                                    mode='markers',
                                    marker=dict(color=cmap[chrom%len(cmap)],
                                                showscale=False,
                                                line=dict(width=0)),
                                    hoverinfo='text',
                                    hovertext=chrom_snps['chr'].astype(str) + ':' + chrom_snps['pos'].astype(str) + '<br>' + 'pval: ' + chrom_snps['pval'].apply("{:.2e}".format)
                        )
                    )

        if scale == 'log':
            fig.update_layout(
                xaxis_title='Chromosome',
                xaxis=dict(
                    tickmode='array',
                    tickvals=chrom_df,
                    ticktext=chrom_df.index
                ),
                yaxis_title=r'$-\log_{10}(p)$',
                showlegend=False,
                title='Manhattan Plot'
            )
        else:
            fig.update_layout(
                xaxis_title='Chromosome',
                xaxis=dict(
                    tickmode='array',
                    tickvals=chrom_df,
                    ticktext=chrom_df.index
                ),
                yaxis_title=r'$p$',
                showlegend=False,
                title='Manhattan Plot'
            )

        # Add horizontal line at alpha
        fig.add_hline(y=alpha, line_dash='dash', line_color='red')


        if isinstance(save_path, str):
            fig.write_html(save_path, 
                           include_mathjax = 'cdn')
        
        return fig
    else:
        # If ax is None, create a new figure
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if cmap is None:
            cmap = sns.color_palette()
    
        sns.scatterplot(x=results_df['i'], 
                        y=results_df['pval'], 
                        hue=results_df['chr'], 
                        linewidth=0, 
                        ax=ax, 
                        palette=cmap)

        # Add horizontal line at alpha
        ax.axhline(alpha, color='red', linestyle='--')

        chrom_df=results_df.groupby('chr')['i'].median()
        ax.set_xlabel('chr')
        ax.set_xticks(chrom_df, chrom_df.index, rotation=90, fontsize = 'xx-small')
        if scale == 'log':
            ax.set_ylabel(r'$-\log_{10}(p)$')
        else:
            ax.set_ylabel(r'$p$')

        ax.legend([],[], frameon=False)

        # Tight layout
        fig.tight_layout()

        if isinstance(save_path, str):
            plt.savefig(save_path, dpi=600)
        
        return fig, ax

def qq_plot(pvals,
            scale='log',
            ax=None,
            save_path=None):
    """
    Generate a QQ plot to visualize p-values.

    Parameters:
    - pvals (array-like): The p-values to be plotted.
    - scale (str, optional): The scale of the plot. Default is 'log'.
    - cmap (str or list-like, optional): The color map to be used for plotting. Default is None.
    - ax (matplotlib Axes, optional): The Axes object to plot on. If None, a new figure and Axes will be created.
    - save_path (str, optional): The file path to save the plot. If provided, the plot will be saved instead of displayed.

    Returns:
    - If save_path is provided, returns None.
    - If save_path is not provided, returns the matplotlib Figure and Axes objects.

    """

    # If trying to save and providing ax, warn user but continue
    if isinstance(save_path, str) and ax is not None:
        warnings.warn('Providing ax with save_path will only save the last plot', UserWarning)

    # If ax is None, create a new figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Sort p-values and generate theoretical quantiles
    theoretical = np.linspace(1/len(pvals),1.0,len(pvals))
    pvals = np.sort(pvals)
    
    # If log scale, take -log10
    if scale == 'log':
        pvals = -np.log10(np.clip(pvals, 
                                a_min=1e-300, 
                                a_max=1))
        theoretical = -np.log10(np.clip(theoretical, 
                                a_min=1e-300, 
                                a_max=1))


    sns.scatterplot(x=theoretical, 
                    y=pvals,
                    linewidth=0,
                    ax=ax)
    
    # Add line at 45 degree angle
    ax.axline((0, 0), slope=1, color='red', linestyle='--')
    
    # Set labels based on scale
    if scale == 'log':
        ax.set_xlabel(r'Theoretical: $-\log_{10}(p)$')
        ax.set_ylabel(r'Observed: $-\log_{10}(p)$')
    else:
        ax.set_xlabel('Theoretical: $p$')
        ax.set_ylabel('Observed: $p$')
    
    # Tight layout
    fig.tight_layout()

    if isinstance(save_path, str):
        plt.savefig(save_path, dpi=600)
    
    return fig, ax

    


