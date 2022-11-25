# %% [markdown]
# # Feature Selection
# ###### Reference: https://scikit-learn.org/stable/modules/feature_selection.html

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
import configparser
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from comet_ml import Experiment
from sklearn import preprocessing
class Directory:
    DATA_DIR = "/Users/raphaelbourque/Desktop/data_science_class/Hockey_primer_2/data/" # Modify the path accordingly
    FIG_DIR = "/Users/raphaelbourque/Desktop/data_science_class/Hockey_primer_2/figures/" # Modify the path accordingly

# %%
config = configparser.ConfigParser()
config.read('./configfile.ini')
type_env = 'comet_ml_prod'
COMET_API_KEY = config[type_env]['api_key']
COMET_PROJECT_NAME = config[type_env]['project_name_feature_selection']
COMET_WORKSPACE = config[type_env]['workspace']

comet_exp_obj = Experiment(api_key=COMET_API_KEY,project_name=COMET_PROJECT_NAME,workspace=COMET_WORKSPACE,log_code=True)
comet_exp_obj.set_name(name="Feature Selection")

# %%
# Load data
X = pd.read_pickle(Directory.DATA_DIR + 'x_train.pkl')
y = pd.read_pickle(Directory.DATA_DIR + 'y_train.pkl')
X.pop('game_id');
X.pop('season');
X.pop('is_goal');

# %% [markdown]
# ## Important note: we do feature selection using only the train set, to avoid leakage of information from the test set. Some feature selection techniques (embedded and wrapper, specifically) imply both X and y, but are also restricted to the train set for this same reason of avoiding information leakage. In the case of wrapper methods, we use nested k-fold cross-validation *within* the train set. 

# %% [markdown]
# ## Describe and visualize all variables

# %%
X.describe().round(2)

# %%
X.hist(bins=30,figsize=(20,15));

# %%
# Here we restrict the density plots to the variables that are truely continuous
X[['distance','angle','distance_from_last_event','speed','x_coordinate','y_coordinate','game_seconds','change_in_shot_angle']].plot(kind='density',subplots=True, layout=(6,5), sharex=False,figsize=(20,15));

# %%
# We also restrict boxplots to continuous variables
X[['distance','angle','distance_from_last_event','speed','x_coordinate','y_coordinate','game_seconds','change_in_shot_angle']].plot(kind='box',subplots=True, layout=(6,5), sharex=False, sharey=False, figsize=(20,15));

# %%
# Correlations among all variables
fig, ax = plt.subplots(figsize=(20, 20));
# sns.set(rc={"figure.figsize":(20, 20)})
ax = sns.heatmap(X.corr(),annot = True);
comet_exp_obj.log_figure(figure_name="Feature correlation structure", figure=plt,overwrite=False, step=None)
# Save figure
path = Directory.FIG_DIR + "figure_Heatmap"
fig.subplots_adjust(bottom=0.4)
fig.savefig(path)

# %% [markdown]
# ## (A) Filter Methods

# %%
# (1) Variance threshold
from sklearn.feature_selection import VarianceThreshold

# %%
proportion = 0.8 # threshold to remove all features that are either one or zero in more than this proportion of the sample
t = proportion * (1 - proportion)
sel = VarianceThreshold(threshold=t)
sel.fit_transform(X)
variance = pd.DataFrame(np.log(sel.variances_), index = X.columns, columns=['variance'])

# %%
# We plot variance, and the threshold as a red line
fig, ax = plt.subplots(figsize=(14, 10));
ax = sns.lineplot(data=variance);
ax = sns.lineplot(data=variance);
ax.axhline(np.log(t), color='red')
plt.xticks(rotation=90);
ax.set(ylabel='variance, *log scale*')
ax.set(title='Feature variance threshold')
comet_exp_obj.log_figure(figure_name="Feature variance and threshold", figure=plt,overwrite=False, step=None)

# Save figure
path = Directory.FIG_DIR + "figure_Feature_variance_threshold"
fig.subplots_adjust(bottom=0.4)
fig.savefig(path)

# %%
variance_filter = X.columns[pd.Series(sel.variances_).rank(ascending=False)<=5].tolist()
print(variance_filter)

# %% [markdown]
# ## For some of the methods that follow, we scale the data, because otherwise methods that use regression do not converge

# %%
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = pd.DataFrame(scaler.transform(X))
X_scaled.columns = X.columns

# %%
# (2) Univariate feature selection
from sklearn import feature_selection

# %%
feature_selection_methods = [feature_selection.f_classif, feature_selection.mutual_info_classif,
feature_selection.SelectFpr, feature_selection.SelectFdr, feature_selection.SelectFwe,feature_selection.SelectPercentile]
features = pd.DataFrame(columns = ['model_name'] +  X.columns.values.tolist(), 
index = range(0,len(feature_selection_methods)))

for i, f in enumerate(feature_selection_methods):
    name = f.__name__
    print(name)
    try: 
        scores_list = feature_selection.SelectKBest(f,k='all').fit(X, y).scores_
        scores_list = (scores_list - np.min(scores_list))/np.ptp(scores_list) # Scale between 0 and 1
        scores_list = scores_list.tolist()
        p_values_list = feature_selection.SelectKBest(f,k='all').fit(X, y).pvalues_.tolist()
    except Exception as e: 
        print(e)
        try: 
            scores_list = feature_selection.SelectKBest(f).fit(X, y).scores_
            scores_list = (scores_list - np.min(scores_list))/np.ptp(scores_list) # Scale between 0 and 1
            scores_list = scores_list.tolist()
            p_values_list = feature_selection.SelectKBest(f).fit(X, y).pvalues_.tolist
        except Exception as e: 
            print(e) 
            pass
    
    features.loc[i] = [name] + scores_list

# %%
fig, ax = plt.subplots(figsize=(14, 10));
ax = sns.barplot(data=features);
ax.set(ylabel='feature score across all selection methods, *scaled from 0 to 1*')
ax.set(title='Univariate filter for each variable across 6 different methods')
plt.xticks(rotation=90);
comet_exp_obj.log_figure(figure_name="Univariate filter for each variable across 6 different methods", figure=plt,overwrite=False, step=None)

# Save figure
path = Directory.FIG_DIR + "figure_Univariate_filter"
fig.subplots_adjust(bottom=0.4)
fig.savefig(path)

# %%
univariate_filter = X.columns[features.mean().rank(ascending=False)<=5].tolist()
print(univariate_filter)

# %% [markdown]
# ## (B) Wapper Methods

# %%
# Forward search
model = LogisticRegression()
sfs = feature_selection.SequentialFeatureSelector(model, n_features_to_select=5, direction='forward', cv=5)
sfs.fit(X_scaled, y)
logistic_forward_search_wrapper = sfs.get_feature_names_out().tolist()
print(logistic_forward_search_wrapper)

# %%
# Backward search = does not converge
# model = LogisticRegression()
# sfs = feature_selection.SequentialFeatureSelector(model, n_features_to_select=5, direction='backward', cv=5)
# sfs.fit(X_scaled, y)
# logistic_backward_search_wrapper = sfs.get_feature_names_out().tolist()
# print(logistic_backward_search_wrapper)

# %% [markdown]
# ## (C) Embedded Methods

# %%
# L2-penalized logistic regression 

# %%
logistic_regession_result = feature_selection.SelectFromModel(estimator=LogisticRegression()).fit(X_scaled, y)

# %%
embedded_select_from_logistic = X.columns[pd.Series(abs(logistic_regession_result.estimator_.coef_[0])).rank(ascending=False)<=5].tolist()

# %%
print(embedded_select_from_logistic)

# %%
# SVC
from sklearn.svm import LinearSVC

# %%
SVC_regession_result = feature_selection.SelectFromModel(estimator=LinearSVC(C=0.01, penalty="l1", dual=False)).fit(X_scaled, y)

# %%
embedded_select_from_SVC = X.columns[pd.Series(abs(SVC_regession_result.estimator_.coef_[0])).rank(ascending=False)<=5].tolist()

# %%
print(embedded_select_from_SVC)

# %%
res = list(zip(abs(logistic_regession_result.estimator_.coef_[0]).tolist(),abs(SVC_regession_result.estimator_.coef_[0]).tolist()))
res = pd.DataFrame(res, index = X.columns, columns=['logistic_regression','linear_SVC'])

# %%
fig, ax = plt.subplots(figsize=(14, 10));
ax = sns.lineplot(data=res);
plt.xticks(rotation=90);
ax.set(ylabel='coefficient, absolute value')
ax.set(title='Feature selection from model coefficient')
fig.subplots_adjust(bottom=0.4)
comet_exp_obj.log_figure(figure_name="Feature selection from model coefficient", figure=plt,overwrite=False, step=None)

# Save figure
path = Directory.FIG_DIR + "figure_Feature_selection_from_model_coefficient"
fig.savefig(path)

# %%
comet_exp_obj.end()

# %%
A = [variance_filter, univariate_filter, logistic_forward_search_wrapper, embedded_select_from_logistic, embedded_select_from_SVC]

# %%
from functools import reduce

# %%
print(list(reduce(set.intersection, [set(x) for x in A ])))

# %%
print(list(reduce(set.union, [set(x) for x in A ])))


