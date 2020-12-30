import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import f_classif, f_regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt
from statsmodels.api import qqplot
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def regression_metrics(estimator, X, y, show_qqplot = False):
    """
    Regression model evaulation metrics including:
        R-squared
        Adjusted R-squared
        RMSE
        MAE
        QQ plot
        Pred-actual scatter plot
    
    Parameters:
        estimator: sklearn estimator
        X: array-like of shape (n_samples, n_features)
        Test samples.

        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.

        show_qqplot: Boolean, default False
        
    Return:
        Evaluation matrix: pandas dataframe
    """
    
    X_copy = X.copy()
    X_copy['pred'] = estimator.predict(X)
    res_df = pd.DataFrame(columns=['metric','Score'])
    r_sq = estimator.score(X, y)
    res_df = res_df.append({'metric': 'R-Squared', 'Score': r_sq.round(4)}, ignore_index=True)
    
    adj_r_sq = 1- (1-r_sq)*(X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1)
    res_df = res_df.append({'metric': 'Adjusted R-Squared', 'Score': adj_r_sq.round(4)}, ignore_index=True)
    
    rmse = mean_squared_error(y, X_copy['pred'], squared = False)
    res_df = res_df.append({'metric': 'RMSE', 'Score': rmse.round(4)}, ignore_index=True)    
    
    mae = mean_absolute_error(y, X_copy['pred'])
    res_df = res_df.append({'metric': 'MAE', 'Score': mae.round(4)}, ignore_index=True)   
    
    display(res_df)
    
    plt.scatter(y, X_copy['pred'], label = 'model prediction')
    plt.plot(y, y, label = 'ideal')
    plt.legend()
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.title('Pred vs Actul')
    plt.show()
    plt.close()
    
    if show_qqplot:
        residuals = X_copy['pred'] - y
        qqplot(residuals, fit=True, line='45');
        plt.title('QQ Plot')
        plt.show()
        plt.close()
    return res_df


def clf_metrics(estimator, X, y):
    """
    Classification model evaulation metrics including:
        Acuracy
        AUC ROC
        Precision
        Recall
        F1 Score
        Pred-actual scatter plot
    
    Parameters:
        estimator: sklearn estimator
        X: array-like of shape (n_samples, n_features)
        Test samples.

        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.
        
    Return:
        Evaluation matrix: pandas dataframe
    """
    X_copy = X.copy()
    X_copy['pred_label'] = estimator.predict(X)
    X_copy['pred_score'] = estimator.predict_proba(X)[:,1]
    
    res_df = pd.DataFrame(columns=['metric','Score'])
    
    acuracy_score = estimator.score(X, y)
    res_df = res_df.append({'metric': 'Acuracy', 'Score': acuracy_score.round(4)}, ignore_index=True)
    
    auc = roc_auc_score(y,X_copy['pred_score'])
    res_df = res_df.append({'metric': 'AUC ROC', 'Score': auc.round(4)},ignore_index=True)
    
    precision_sc = precision_score(y,X_copy['pred_label'])
    res_df = res_df.append({'metric': 'Precision', 'Score': precision_sc.round(4)},ignore_index=True)
                           
    recall_sc = recall_score(y,X_copy['pred_label'])
    res_df = res_df.append({'metric': 'Recall', 'Score': recall_sc.round(4)},ignore_index=True )
                           
    recall_pre_f1 = f1_score(y,X_copy['pred_label'])
    res_df = res_df.append({'metric': 'F1 Score', 'Score': recall_pre_f1.round(4)},ignore_index=True)
    
    display (res_df)
    
    fpr, tpr, thr = roc_curve(y,X_copy['pred_score'])
    plt.plot(fpr, tpr, label = 'model')
    plt.plot(fpr, fpr, '--', label = 'random')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    plt.close()
    precision, recall, thre = precision_recall_curve(y,X_copy['pred_score'])
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall - Precision  Curve')
    plt.show()  
    
    return res_df



def logistic_reg_coef_df(model_res, var_list):
    coef_df = pd.DataFrame(data = list(model_res.intercept_)+ list(model_res.coef_[0]), index = ['Intercept'] + var_list, columns=['Estimate'])
    return coef_df

def coef_df(model_res, var_list):
    coef_df = pd.DataFrame(data = [model_res.intercept_] + list(model_res.coef_), index = ['Intercept'] + var_list, columns=['Estimate'])
    return coef_df

def var_importance(estimator_res, model_vars):
    var_imp = estimator_res.feature_importances_
    var_imp_df = pd.DataFrame(data = var_imp.transpose(), index = model_vars, columns = ['Importance'])
    var_imp_df = var_imp_df.sort_values(by='Importance', ascending=False)
    var_imp_df['cumsum'] = var_imp_df.cumsum().round(3)
    return var_imp_df



def vif(df, var_list):
    
    tmp_df = df[var_list].copy()
    tmp_df['const'] = 1
    vifs = [variance_inflation_factor(tmp_df.values, i) 
                          for i in range(len(var_list) + 1)] 
    vif_df = pd.DataFrame()
    vif_df['feature'] = list(tmp_df.columns)
    vif_df['VIF'] = vifs
    vif_df = vif_df[vif_df['feature'] != 'const'].sort_values(by = 'VIF', ascending = False)
    
    return vif_df



def drop_high_vif(df, cand_vars, vif_threshold = 5):
    tmp_vif_df = vif(df[cand_vars], cand_vars)
    if tmp_vif_df.iloc[0][1] < vif_threshold:
        return tmp_vif_df
    
    cand_vars_copy = cand_vars.copy()
    
    while tmp_vif_df.iloc[0][1] > vif_threshold:
        cand_vars_copy.remove(tmp_vif_df.iloc[0][0])
        tmp_vif_df = vif(df[cand_vars_copy], cand_vars_copy)
    
    display(tmp_vif_df)
    
    return tmp_vif_df


def univariate_f_classif(X,y):
    anova_df = pd.DataFrame(data = np.array(f_classif(X, y)).transpose(), 
                            index = X.columns, columns=['F-value', 'p-val']).round(4)
    
    res = anova_df.sort_values(by = 'F-value', ascending = False)
    
    display(res)
    
    return res



def univariate_f_regression(X,y):
    anova_df = pd.DataFrame(data = np.array(f_regression(X, y)).transpose(), 
                            index = X.columns, columns=['F-value', 'p-val']).round(4)
    
    res = anova_df.sort_values(by = 'F-value', ascending = False)
    display(res)
    
    return res