# -*- coding: utf-8 -*-
"""
author: vengozhang
"""
import matplotlib.pyplot as plt
from seaborn import load_dataset
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

df_boston = pd.read_csv(r".\CSV_data\boston.csv")

#%%  utils   inspired from  https://github.com/vd1371/PyImbalReg and https://www.coder.work/article/7549880
def sample_uniform(data, nr=0.3):
    np.random.seed(1)
    df = data
    x = np.sort(df.iloc[:,0])
    f_x = np.gradient(x)*np.ones(len(x))
    sample_probs = f_x/np.sum(f_x)
#samples = np.random.choice(x, p=sample_probs, size=1000)
    n_target = int(nr*len(df))  # 采样数量
    df_samples_ = df.iloc[:,0].sort_values().sample(
        n = n_target,
        weights=sample_probs,
        replace=True,
        )
    df_samples_index = df_samples_.index
    df_samples = df.loc[df_samples_index]
    plt.hist(df_samples_, histtype="step", density=True)
    return df_samples
def evaluate(data, train_set, test_set, bins, method):
	'''Creating a function to evaluate the splitting by showing the histogram'''

	print (f" ------- With {method} ---------- ")
	print (f"Mean: Data: {data.iloc[:, 0].mean():.2f} " \
					f"train_set {train_set.iloc[:, 0].mean():.2f} " \
					f"test_set {test_set.iloc[:, 0].mean():.2f}")

	print (f"STD: Data: {data.iloc[:, 0].std():.2f} " \
				f"train_set {train_set.iloc[:, 0].std():.2f} " \
				f"test_set {test_set.iloc[:, 0].std():.2f}" )
	print (" ------------------------------------")

	fig, axs = plt.subplots(3)
	axs[0].hist(data.iloc[:, 0], bins = bins)
	axs[1].hist(train_set.iloc[:, 0], bins = bins)
	axs[2].hist(test_set.iloc[:, 0], bins = bins)
	plt.show()
	del fig, axs


#%%   boston     sample ok!!!     √
plt.hist(df_boston.iloc[:,0], histtype="step", density=True)
df_boston_s = sample_uniform(df_boston,0.6)
plt.hist(df_boston_s.iloc[:,0], histtype="step", density=True)
df_boston_tr = pd.concat([df_boston,df_boston_s]).drop_duplicates(keep=False)
df_boston_te = df_boston_s.drop_duplicates()
bins = 5
df_boston_test, df_boston_val = train_test_split(df_boston_te, test_size = 0.5,random_state = 2)
evaluate(df_boston_tr,df_boston_test, df_boston_val, bins, "sklearn")

print(len(df_boston_tr),len(df_boston_test),len(df_boston_val), len(df_boston_te))

df_boston_tr.to_csv(r".\CSV_data\boston_train.csv")
df_boston_test.to_csv(r".\CSV_data\boston_test.csv")
df_boston_val.to_csv(r".\CSV_data\boston_val.csv")

