#!/usr/bin/env python
# coding: utf-8
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
#import warning
#warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

# In[ ]:
# data analysis and wrangling
import pandas as pd
import numpy as np
import os
import random

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

################################################

def check_missing_data(df, plot):
    """Checks the missing data at columns level and prints the % of missing data.
    Also prints the heat map for dataframe
    """
    missing_val = df.isnull().sum().sort_values(ascending=False)
    per_missing_val = round(df.isnull().sum()/df.isnull().count()*100,2).sort_values(ascending=False)
    per_missing_val = per_missing_val[per_missing_val.values > 0 ]
    print (per_missing_val.index)
    print (per_missing_val)
#   Plots the heatmap for missing values
    if (plot == 'y'):
        plt.subplots(figsize=(9,9))
        sns.heatmap(df.isnull())
        plt.show()
    return None


        
# Program for bi variate anaysis of categorical features.
def Analysis_CountPlot(df_cat, colToanalyze, hueColumn):  
    plt.subplots(figsize=(15,6))
    ax = sns.countplot(x= colToanalyze, hue=hueColumn, data=df_cat)
    ax.set_title('Bi-Variate Analysis for :' + colToanalyze)
    plt.show()

def Analysis_Crosstab(df_cat, targetcol, colToAnalyze):  
#     return pd.crosstab(df_cat[targetcol], df_cat[colToAnalyze], margins=True)
    print (pd.crosstab(df_cat[targetcol], df_cat[colToAnalyze], margins=True))
    #return df_dist    
    
    
# Univariate Analysis
def distplot_numericdata(df_numerical):
    """This plots the distribution of data for numerical features
       If column is passed, then only for that col, it generates the plot.
    """
    import os
    import random
    sns.set(color_codes=True)
    colors = ['y','b','g','r']
    if (os.path.isdir('plots') == False):
        os.mkdir('plots')
    if (os.path.isdir('plots/univariate') == False):
        os.mkdir('plots/univariate')
    if (os.path.isdir('plots/bivariate') == False):
        os.mkdir('plots/bivariate')
    
    for i, col in enumerate(df_numerical.columns):
        #print(col)
        #Take the Numerical data under 5-95 % percentile to remove the outliers.
        df_percentile = df_numerical.loc[(df_numerical[col] < np.percentile(df_numerical[col],95)) & (df_numerical[col] > np.percentile(df_numerical[col],5)), [col]]          
        dplot = sns.distplot(df_percentile[col].dropna(), color=random.choice(colors))
        #dplot = sns.distplot(df_numerical[col].dropna(), color=random.choice(colors))
        dplot.set_title(col)
        dplot.set_xlabel(col)
        dplot.set_ylabel("density")
        #plt.savefig("plots/univariate/dplot_" + str(i) + ".png")
        plt.show()
        plt.clf()
        plt.close()    

        
# Bivariate Analysis
def bivariate_analysis(df_numerical, col, plottype):
    """ This plots the columsn with Target column and draw a regression line.
        If column is passed, then only for that col, it generates the plot.
    """
    cols=[]
    if col:
        cols.append(col)
    else:
        cols = df_numerical.columns
    for i, col in enumerate(cols):
        jplot = sns.jointplot(x=df_numerical[col], y=df_numerical['Target'], kind=plottype).set_axis_labels(col,'Target')
        plt.show()
        plt.clf()
        plt.close()
    
from scipy import stats
from sklearn.feature_selection import chi2
    
def Analysis_chi2(df_cat, targetcol, colToAnalyze):
    # Correlation analysis with Target column
    crosstab = pd.crosstab(df_cat[colToAnalyze], df_cat[targetcol])
    print ("Below are Chi-2 test results")
    chi2_stat, p, dof, expected = stats.chi2_contingency(crosstab)
    
    print('dof=%d' % dof)
    #print(expected)
    # interpret test-statistic
    prob = 0.95
    critical = stats.chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, chi2_stat=%.3f' % (prob, critical, chi2_stat))
    if abs(chi2_stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    # interpret p-value
    #alpha = 1.0 - prob
    #print('significance=%.3f, p=%.3f' % (alpha, p))
    #if p <= alpha:
    #    print('Dependent (reject H0)')
    #else:
    #    print('Independent (fail to reject H0)')

def GetRecordsHavingMoreThanOnePercent(df, ColToAnalyze, indexes):   
    df_collect = pd.DataFrame()    
    for i in indexes:
        dff = df[(df[ColToAnalyze] == i)]
        df_collect = df_collect.append(dff, ignore_index=True)
    return df_collect 

def GetFeatureRecordsPercentage(df_categorical, ColToAnalyze):
#     df_count_records = df_categorical.groupby([ColToAnalyze]).agg({ColToAnalyze : 'count' }).sort_values(by=ColToAnalyze,ascending=False)
#     df_count_records['PERCENTAGE'] =(df_count_records[ColToAnalyze]/len(df))*100
    df_count_records = df_categorical.groupby([ColToAnalyze]).count().Status.to_frame('count').reset_index().sort_values(by='count',ascending=False)
    df_count_records['PERCENTAGE'] =(df_count_records['count']/len(df_categorical))*100
    return df_count_records


def GetFeatureRecordsPercentage(df, ColToAnalyze):
#     df_count_records = df_categorical.groupby([ColToAnalyze]).agg({ColToAnalyze : 'count' }).sort_values(by=ColToAnalyze,ascending=False)
#     df_count_records['PERCENTAGE'] =(df_count_records[ColToAnalyze]/len(df))*100
    df_count_records = df.groupby([ColToAnalyze]).count().Status.to_frame('count').reset_index().sort_values(by='count',ascending=False)
    df_count_records['PERCENTAGE'] =(df_count_records['count']/len(df))*100
    return df_count_records

# Program for multi-variate  analysis of categorical features.
def Analysis_MultiVariatePlot(df_cat, colToanalyze, hueColumn, numerical_col, display_x_col, display_y_col): 
    plt.subplots(figsize=(15,6))
    ax = sns.barplot(x=colToanalyze, y=numerical_col, hue = hueColumn, estimator = sum, data=df_cat)
    ax.set(xlabel=display_x_col, ylabel=display_y_col)
    ax.set_title('Multivariate Analysis for :' + display_y_col + ' for respective ' + display_x_col)  
    plt.show()
    
def MultiVariateAnalysisForGivenPercentVolume(df_categorical,ColToAnalyze, hueCol, numerical_col, display_x, display_y, percentage = 0):
    
    print ('                                                                                   ')
    print ('                                                                                   ')
    print ('===================================================================================')
    print ('====== GENERATING STATS FOR ' + ColToAnalyze + ': ================')
    print ('===================================================================================')
    
    if percentage <= 0:
        percentage = 1
    df_Percent = GetFeatureRecordsPercentage(df_categorical, ColToAnalyze)

#     Percent_index = df_Percent[ df_Percent['PERCENTAGE'] > percentage ].index.tolist()
    
    Percent_index = df_Percent[ df_Percent['PERCENTAGE'] > percentage ][ColToAnalyze].tolist()
    
    print ('====== Total Subcategories FOR  ' + ColToAnalyze + ' Are :', len(df_categorical[ColToAnalyze].value_counts()))
    
    print('---------------------------------------\r')
    
    print ('====== Subcategories FOR More Than  ' + str(percentage) +' Percent Data For '+ ColToAnalyze + ': ================')
    print('---------------------------------------\r')
    print(df_Percent[ df_Percent['PERCENTAGE'] > percentage ])
    
    df_gt_1_percent = GetRecordsHavingMoreThanOnePercent(df_categorical, ColToAnalyze, Percent_index)

    print ('====== CROSSTAB STATS FOR ' + ColToAnalyze + ': ================')
    print('---------------------------------------\r')
    Analysis_Crosstab(df_gt_1_percent, hueCol, ColToAnalyze)
    
    Analysis_MultiVariatePlot(df_gt_1_percent,ColToAnalyze, hueCol, numerical_col, display_x, display_y)

    
    # Adding Chi2 results
    print ('====== CHI2 STATS For ' + ColToAnalyze + ': ================')
    print('---------------------------------------\r')
    Analysis_chi2(df_cat=df_gt_1_percent, targetcol=hueCol, colToAnalyze=ColToAnalyze)
        
    Analysis_CountPlot(df_gt_1_percent,ColToAnalyze, hueCol)
    
    
#     return Analysis_Crosstab(df_gt_1_percent, hueCol, ColToAnalyze)

def BiVariateAnalysisForGreaterThanOnePercentVolume(df_categorical,ColToAnalyze, hueCol, percentage = 0):

    print ('===================================================================================')
    print ('====== GENERATING STATS FOR ' + ColToAnalyze + ': ================')
    print ('===================================================================================')
    
    
    if percentage <= 0:
        percentage = 1
    df_Percent = GetFeatureRecordsPercentage(df_categorical, ColToAnalyze)

#     Percent_index = df_Percent[ df_Percent['PERCENTAGE'] > percentage ].index.tolist()
    Percent_index = df_Percent[ df_Percent['PERCENTAGE'] > percentage ][ColToAnalyze].tolist()
    
    print(df_Percent[ df_Percent['PERCENTAGE'] > percentage ])
    
    df_gt_1_percent = GetRecordsHavingMoreThanOnePercent(df_categorical, ColToAnalyze, Percent_index)

    # Adding Chi2 results
    Analysis_chi2(df_cat=df_gt_1_percent, targetcol=hueCol, colToAnalyze=ColToAnalyze)
    
    Analysis_CountPlot(df_gt_1_percent,ColToAnalyze, hueCol)

    return Analysis_Crosstab(df_gt_1_percent, hueCol, ColToAnalyze)

