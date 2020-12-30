import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


def _two_group_histplot_num_var(df, numerical_cols, y_col, bins = 10):
    yvals = df[y_col].unique()
    total_figs = len(numerical_cols)
    fig_cols = 3
    fig_rows = int(total_figs/3)+1
    plt.figure(figsize=(15,fig_rows*3.8))
    fig = 1
    for col in numerical_cols:
        if col == y_col:
            continue
        tmp_df = df[[y_col,col]].copy()
 
        col_y_val_0 = df[df[y_col] == yvals[0]][col]
        col_y_val_1 = df[df[y_col] == yvals[1]][col]

        ax1 = plt.subplot(fig_rows,fig_cols,fig)    
        ax1.hist(col_y_val_0,bins=bins, histtype = 'step', color = 'r', weights=np.ones(len(col_y_val_0)) / len(col_y_val_0), label = 'y = '+str(yvals[0]));
        ax1.hist(col_y_val_1,bins=bins, histtype = 'step',color = 'b', weights=np.ones(len(col_y_val_1)) / len(col_y_val_1), label = 'y = '+str(yvals[1]));
        plt.title(col)
        plt.legend()
        plt.tight_layout()
        
        if fig % 3 == 1:
            plt.ylabel('Proportion')
            
        fig += 1
    plt.show()

def _two_group_histplot_cat_var(df, cat_cols, y_col, bins = 10):
    yvals = df[y_col].unique()
    total_figs = len(cat_cols)
    fig_cols = 3
    fig_rows = int(total_figs/3)+1
    plt.figure(figsize=(15,fig_rows*3.8))
    fig = 1
    for col in cat_cols:
        if col == y_col:
            continue
        hist_df = pd.DataFrame()
        tmp_df = df[[y_col,col]].copy()
        hist_df[col] = tmp_df[col].unique() 
 
        col_y_val_0 = df[df[y_col] == yvals[0]][[col, y_col]]
        col_y_val_1 = df[df[y_col] == yvals[1]][[col, y_col]]
        
        
        col_y_val_0_cnt = col_y_val_0.groupby(by = col,as_index = False).agg({y_col:'count'})
        col_y_val_0_cnt = col_y_val_0_cnt.rename(columns = {y_col: '0_prop'})
        
        col_y_val_1_cnt = col_y_val_1.groupby(by = col,as_index = False).agg({y_col:'count'})
        col_y_val_1_cnt = col_y_val_1_cnt.rename(columns = {y_col: '1_prop'})
        
        hist_df = hist_df.merge(col_y_val_0_cnt, how = 'left', on = col)
        hist_df = hist_df.merge(col_y_val_1_cnt, how = 'left', on = col)
        
        hist_df = hist_df.fillna(0)

        ax1 = plt.subplot(fig_rows,fig_cols,fig)   
        ax1.bar(np.array(range(len(hist_df[col]))),hist_df['0_prop'] / sum(hist_df['0_prop']), width = 0.2,edgecolor='r', color='None', label = 'y = '+str(yvals[0]))
        ax1.bar(np.array(range(len(hist_df[col]))) + 0.2,hist_df['1_prop']/ sum(hist_df['1_prop']), width = 0.2,edgecolor='b', color='None',label = 'y = '+str(yvals[1]))
        plt.xticks(np.array(range(len(hist_df[col]))) + 0.2, hist_df[col]);
        plt.xticks(rotation = 45)
        plt.title(col)
        plt.legend()
        plt.tight_layout()
        
        if fig % 3 == 1:
            plt.ylabel('Proportion')
            
        fig += 1
    plt.show()

def two_group_histplot(raw_data, y_col, bins = 10):
    """
    Histogram plots for each of two classes: y = 0 and y = 1
    
    Parameters:
        raw_data: pandas dataframe
        Including y column and x-features to plot
        
        y_col: string
        The column name of response y in raw_data
        
        bins: int, default = 10
        Number of bins in histogram plot
    
    Return:
        None
    """
    numerical_cols, char_cols = col_types(raw_data, y_col, to_print = False)
    if len(numerical_cols) > 0:
           _two_group_histplot_num_var(raw_data, numerical_cols, y_col, bins = bins) 
    
    if len(char_cols) > 0:
           _two_group_histplot_cat_var(raw_data, char_cols, y_col, bins = bins) 



def heatmap_corr(corr, figsize=(15, 12)):
    """
    Heatmap plot of pair-wise correlation coefficients
    
    Parameters:
        corr: rectangular dataset
        2D dataset that can be coerced into an ndarray. 
        If a Pandas DataFrame is provided, the index/column information will be used to label the columns and rows.
        
        figsize: tuple, default = (15,12)
        Figure size

    Return:
        None
    """
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            mask=mask,
            square=True
        );


def col_types(df, y_col,to_print = True):
    numerical_cols = list()
    categorical_cols = list()
    col_types = df.dtypes
    for k, v in col_types.items():
        if k == y_col:
            continue
        if v in ['int64','float64']:
            numerical_cols.append(k)
        else:
            categorical_cols.append(k)
    if to_print:
        print('Numerical Columns are:')
        display( numerical_cols )

        print('\n Categorical Columns are:')
        display( categorical_cols )
            
    return numerical_cols, categorical_cols


def iqr_outliers(df, col, q1 = None, q3 = None, alpha = 1.5):
    if q1 is None or q2 is None:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
    tmp_iqr = q3 - q1
    
    outliers = len(df[(df[col] < q1 - tmp_iqr*alpha) | (df[col] > q3 + tmp_iqr*alpha)].index)

    return outliers

def data_summary(df, cols = None):
    if cols is not None:
        data_describe = df[cols].describe().transpose()
    else:
        data_describe = df.describe().transpose()
    
    df_missing = pd.DataFrame(df.isna().sum(), columns=['missing'])
    data_describe  = data_describe.merge(df_missing, left_index = True, right_index = True )
    data_describe['missing %'] = data_describe['missing']/(data_describe['missing'] + data_describe['count'])*100
    data_describe['missing %'] = data_describe['missing %'].round(2)

    data_describe['1.5iqr outliers'] = data_describe.apply(lambda r: iqr_outliers(df, r.name), axis=1)
    data_describe['outliers %'] = data_describe['1.5iqr outliers'] / data_describe['count']*100
    data_describe['outliers %'] = data_describe['outliers %'].round(2)
    
    return data_describe

def missing_value_summary(df, y_col, cols = None):
    if cols is not None:
        tmp_df = df[cols].copy()
    else:
        tmp_df = df.copy()
    
    df_missing = pd.DataFrame(tmp_df.isna().sum(), columns=['missing'])
    df_missing['tot count'] = len(tmp_df.index)
    df_missing['missing %'] = df_missing['missing'] / df_missing['tot count'] *100
    df_missing['missing %'] = df_missing['missing %'].round(2)
    
    avg_by_missing = pd.DataFrame()
    for col in tmp_df.columns:
        tmp_df['missing_ind'] = tmp_df[col].isna()
        xx = tmp_df.groupby(by='missing_ind').agg({y_col: 'mean'}).transpose()
        xx.index = [col]
        xx.columns = [str(r) for r in xx.columns]
        avg_by_missing = avg_by_missing.append(xx)
       
    avg_by_missing = avg_by_missing.rename(columns={'False':'non-missing avg', 'True':'missing avg'})

    ans = df_missing[['tot count', 'missing', 'missing %']].merge(avg_by_missing, left_index = True, right_index = True)
    return ans


def scatter_plot(df,numerical_cols, y_col):
    
    if y_col in numerical_cols:
        numerical_cols.remove(y_col)
    total_figs = len(numerical_cols)
    fig_cols = 3
    fig_rows = int(total_figs/3)+1
    plt.figure(figsize=(15,fig_rows*3.8))
    fig = 1
    for col in numerical_cols:
        tmp_df = df[[y_col,col]].copy()
        plt.subplot(fig_rows,fig_cols,fig)
        plt.scatter(tmp_df[col],tmp_df[y_col])
        plt.title(col)
        #plt.xticks(rotation = 45)
        plt.tight_layout()
        fig += 1
    plt.show()
    plt.close()
    
    
def categorical_plot(df, char_cols, y_col):
    if y_col in char_cols:
        char_cols.remove(y_col)
    total_figs = len(char_cols)
    fig_cols = 3
    fig_rows = int(total_figs/3)+1
    plt.figure(figsize=(15,fig_rows*3.8))
    fig = 1
    for col in char_cols:
        tmp_df = df[[y_col,col]].copy()
        avg = tmp_df.groupby(by = col, as_index = False).agg({y_col: 'mean'})
        cnt = tmp_df.groupby(by = col, as_index = False).agg({y_col: 'count'})
        avg[col] = [str(i) for i in avg[col]]
        ax1 = plt.subplot(fig_rows,fig_cols,fig)    

        ax2 = ax1.twinx()
        ax1.bar(cnt[col],cnt[y_col], color = 'lightgray')
        ax2.plot(avg[col],avg[y_col], zorder = 10, linewidth = 2)
        ax1.yaxis.tick_right()
        ax2.yaxis.tick_left()
        plt.title(col)
        ax1.tick_params(axis='x', rotation = 45)
        plt.tight_layout()
            
        fig += 1
    plt.show()
    plt.close()
    
def bivariate_plot(df, y_col, cols = None, bins = 10, binary = False):
    """
    Bivariate plot of x(feature) and y(response) for each x in df. Numerical features use scatter plot is used.Categorical features use bin plot.  If y is binary, all features use bin plots.
    
    Parameters:
        df: pandas dataframe
        Including y column and x-features to plot
        
        cols: list
        List of column names for features to plot
        
        bins: int, default = 10
        Number of bins in histogram plot
        
        binary: boolean, default = False
        Indicates if the reponse is binary or not        
    
    Return:
        None
    """   
    
    col_types = df.dtypes
    numerical_cols = set()
    char_cols = set()

    for k, v in col_types.items():
        if v not in ['int64','float64']:
            char_cols.add(k)
        else:
            numerical_cols.add(k)
    
    if cols is not None:
        char_cols = char_cols & set(cols)
        numerical_cols = numerical_cols & set(cols)
    
    if binary == False:
        scatter_plot(df,list(numerical_cols), y_col)
    else:
        bin_plot(df, list(numerical_cols), y_col, bins = bins, quantile = False)
        
    categorical_plot(df, char_cols, y_col)
    
    
def bin_plot(df, numerical_cols, y_col, bins = 10, quantile = False):
    total_figs = len(numerical_cols)
    fig_cols = 3
    fig_rows = int(total_figs/3)+1
    plt.figure(figsize=(15,fig_rows*3.8))
    fig = 1
    for col in numerical_cols:
        if col == y_col:
            continue
        tmp_df = df[[y_col,col]].copy()
        if len(tmp_df[col].unique())<= bins:
            tmp_df['bin'] = tmp_df[col]
        else:
            if quantile == True:
                tmp_df['bin'] = pd.qcut(tmp_df[col], q=bins)
            else:
                tmp_df['bin'] = pd.cut(tmp_df[col], bins=bins)

        avg = tmp_df.groupby(by = 'bin', as_index = False).agg({y_col: 'mean'})
        cnt = tmp_df.groupby(by = 'bin', as_index = False).agg({y_col: 'count'})
        avg['bin'] = [str(i) for i in avg['bin']]

        ax1 = plt.subplot(fig_rows,fig_cols,fig)    
        ax2 = ax1.twinx()
        ax1.bar(avg['bin'],cnt[y_col], color = 'lightgray')

        ax2.plot(avg['bin'],avg[y_col], zorder = 10, linewidth = 2)

        ax1.yaxis.tick_right()
        ax2.yaxis.tick_left()
        plt.title(col)
        ax1.tick_params(axis='x', rotation = 45)
        plt.tight_layout()
            
        fig += 1

    plt.show()
    

def impute_missing(df, cols = None, numerical_impute = 'mean'):
    if cols is None:
        cols = df.columns
  
    tmp_df = df.copy()
    for k, v in tmp_df.dtypes.items():
        if k not in cols:
            continue
        if v in ['int64', 'float64']:
            if numerical_impute == 'mean':
                impute_v = tmp_df[k].mean()
            elif numerical_impute == 'median':
                impute_v = tmp_df[k].median()
            elif numerical_impute == 'mode':
                impute_v = tmp_df[k].mode()[0]
        else:
            impute_v = tmp_df[k].mode()[0]
        tmp_df[k] = tmp_df[k].fillna(impute_v)

    return tmp_df



def remove_outliers_iqr(df, cols = None, iqr_alpha = 1.5):
    df_copy = df.copy()
    col_types = df_copy.dtypes
    numerical_cols = set()
    
    for k, v in col_types.items():
        if v in ['int64','float64']:
            numerical_cols.add(k)
    if cols is not None:
        numerical_cols = numerical_cols & set(cols)
    
    for col in numerical_cols:
        q25 = df_copy[col].quantile(0.25)
        q75 = df_copy[col].quantile(0.75)
        
        iqr_v = q75 - q25
        
        low_lim = q25 - iqr_alpha * iqr_v
        up_lim = q75 + iqr_alpha * iqr_v
        
        df_copy = df_copy[(df_copy[col] >= low_lim) & (df_copy[col] <= up_lim)].copy()
    
    return df_copy