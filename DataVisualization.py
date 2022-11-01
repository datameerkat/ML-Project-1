# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:42:31 2022

@author: max44
"""
from ExtractData import *
# Exercise 4.2.5

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

#shown_attributes = ["Age","Height","Weight","FCVC","NCP","CAEC","CH2O","FAF","TUE","obesity"]
#shown_attributes = ["Age","Height","TUE","obesity","CH2O","FCVC", "Weight","NCP"]


sns.set_theme(style="ticks")
X_df = pd.DataFrame(X[:,:], columns = attributeNames)
obesity = pd.DataFrame(raw_data[:,16], columns = ["obesity"])
weight = pd.DataFrame(raw_data[:,3], columns = ["Weight"])
X_df = pd.concat([X_df,obesity,weight], axis=1)
X_df = X_df.rename(columns={"obesity": "BMI class"})

percent = 0.7
remove_n = int(percent * 2111)
drop_indices = np.random.choice(X_df.index, remove_n, replace=False)
X_df_dropped = X_df.drop(drop_indices)

shown_attributes = ["Age","Height","Weight",'FCVC','NCP','BMI class','FAVC']
X_df_trimmed = X_df[shown_attributes]

#sns.pairplot(X_df, hue = "obesity")
sns.pairplot(X_df_trimmed, hue='FAVC')

shown_attributes = ["CAEC",'CH2O','FAF','TUE','CALC','BMI class']
X_df_trimmed = X_df[shown_attributes]

sns.pairplot(X_df_trimmed)
#standardize
#scaler = StandardScaler()
#X_df_scaled = scaler.fit_transform(X_df)

##swarmplot
#ax = sns.swarmplot(data=df, x="body_mass_g", y="obesity", hue="species")
#ax.set(ylabel="")
#ax = sns.swarmplot(data=X_df)
#ax.set(ylabel="")

#combine BMI classes

#randomly drop samples
shown_attributes = ["Age","Height","Weight",'FCVC','NCP','BMI class']
X_df_config = X_df_dropped[shown_attributes]

#sns.pairplot(X_df, hue = "obesity")
sns.pairplot(X_df_config)

shown_attributes = ["CAEC",'CH2O','FAF','TUE','CALC','BMI class']
X_df_config = X_df_dropped[shown_attributes]

sns.pairplot(X_df_config)

#Weight over FAVC
shown_attributes = ["Weight",'FAVC']
X_df_trimmed = X_df[shown_attributes]
sns.displot(X_df_trimmed, x="Weight", hue="FAVC", kind="kde", fill=True)

#Weight over family_history_with_overweight
shown_attributes = ["Weight",'family_history_with_overweight']
X_df_trimmed = X_df[shown_attributes]
X_df_trimmed = X_df_trimmed.rename(columns={"family_history_with_overweight": "FHO"})
sns.displot(X_df_trimmed, x="Weight", hue="FHO", kind="kde", fill=True)

#Weight over SMOKE
shown_attributes = ["Weight",'SMOKE']
X_df_trimmed = X_df[shown_attributes]
sns.displot(X_df_trimmed, x="Weight", hue="SMOKE", kind="kde", fill=True)

#Weight over SCC
shown_attributes = ["Weight",'SCC']
X_df_trimmed = X_df[shown_attributes]
sns.displot(X_df_trimmed, x="Weight", hue="SCC", kind="kde", fill=True)