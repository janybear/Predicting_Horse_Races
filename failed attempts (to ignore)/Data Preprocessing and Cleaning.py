# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:43:35 2024

@author: Compute2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

#importing raw data
details_rawdata = pd.read_csv('C:/Users/Compute2/Desktop/Jany/TASK/Raw Data/race_details_20240101_20240131.csv', encoding='utf-8' )
results_rawdata = pd.read_csv('C:/Users/Compute2/Desktop/Jany/TASK/Raw Data/race_results_20240101_20240131.csv', encoding='utf-8') #to display Turkish characters correctly

def normalize(df, column_name):
    df_norm = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())   
    return df_norm

def convert_to_milliseconds(time_str):
    mins, sec, millisec = time_str.split('.')
    total_milliseconds = (int(mins)*60*1000)+ (int(sec)*1000)+ int(millisec)
    return total_milliseconds

#check no of unique items
def no_unique_cat(df, column):
    unique_values = set(df[column].unique())
    return print(f"no. of unique {column}: ", len(unique_values))

df1=results_rawdata
df2=details_rawdata
#Creating new columns, combining data
df_merged = df1.merge(df2[['race_date', 'race_date', 'col_to_add1', 'col_to_add2']], 
                      how='left', 
                      left_on=['col_match1', 'col_match2'], 
                      right_on=['col_match1', 'col_match2'])



#1. cleaning, numerical encoding/categorizing, or normalizing (to [0,1]):
#checking no. of unique categories
no_unique_cat(results_rawdata, 'horse_origin')
no_unique_cat(results_rawdata, 'horse_trainer')
no_unique_cat(results_rawdata, 'jockey_name')
no_unique_cat(results_rawdata, 'horse_owner')
no_unique_cat(results_rawdata, 'horse_accessories')
    
#dropped columns
features = results_rawdata.drop(['jockey_name','horse_owner', 'horse_trainer', 'race_city','horse_name','race_date', 'horse_sire','horse_dam', 'hors_broodmare_sire','horse_margin', 'horse_late_start', 'Horse_starting_box_no' ] , axis=1)
print(len(features))
#horse_starting_box removed because its often random to ensure a fair start for all horses, no info gain here . 'horse_origin''horse_accessories'

#removing repeat rows (lowers skewing)
features = features.drop_duplicates()
print(len(features))

#check for missing data
import missingno as msno
msno.matrix(results_rawdata)
plt.show()

#horse finish_time
features = features.rename(columns={'horse_race_degree': 'finish_time'})
features['finish_time'] = features['finish_time'].replace('Derecesiz', np.nan)# fill undetermined time 
features['finish_time'] = features['finish_time'].apply(lambda x: convert_to_milliseconds(x) if pd.notnull(x) else np.nan)
features['finish_time'] = normalize(features, 'finish_time')

#horse_age
features['horse_age'] = features['horse_age'].str.replace('y', '')  
features['horse_age'] = features['horse_age'].apply(pd.to_numeric, errors='coerce')
features['horse_age'] = normalize(features, 'horse_age')

#jockey_weight
features['jockey_weight'] = normalize(features, 'jockey_weight')

#could normalize horse_win_value to the max
features['horse_win_value'] = normalize(features, 'horse_win_value')

#psf
features['horse_psf_rate'] = normalize(features, 'horse_psf_rate')
features['horse_psf_rank'] = normalize(features, 'horse_psf_rank')
features['horse_rate'] = normalize(features, 'horse_rate')

#horse_sex
unique_values = pd.get_dummies(features['horse_sex'], drop_first=False).astype(int)
features = pd.concat([features, unique_values], axis=1)
features.drop('horse_sex', axis=1, inplace=True)



# horse_accessories
unique_values = pd.get_dummies(features['horse_accessories'], drop_first=False).astype(int)
features = pd.concat([features, unique_values], axis=1)
features.drop('horse_accessories', axis=1, inplace=True)


#ensure all features are numerical:
#features = features.apply(pd.to_numeric, errors='coerce')

# determining features and target
max_no_of_racers=int(max(features['result']))
bin_counts = features['result'].value_counts().sort_index()
# Create the histogram plot with Seaborn
plt.figure(dpi=300)  # Increase DPI for high-resolution
sns.barplot(x=bin_counts.index, y=bin_counts.values, color='cornflowerblue')
plt.xlabel('Number of horses in a single race')
plt.ylabel('Frequency')
plt.title('Histogram of Number of horses in a single race')
plt.show()

indicesof0 = features[features['result'] == 0].index #removing rows where result=0
features = features.drop(indicesof0)
bin_counts = features['result'].value_counts().sort_index() #check

features = features[features['result'] < 9] #to cut down sample size to top 6 horses only
features.reset_index(drop=True, inplace=True)
race_outcome = features[['result']].copy()
features = features.drop('result', axis=1)

#4. feature engineering



#convert classes to python 0-based indexing
race_outcome = race_outcome -1

#check for missing data
import missingno as msno
msno.matrix(features)
plt.show()

#SAVE TO CSV
features.to_csv(r'C:/Users/Compute2/Desktop/Jany/TASK/Results/initial attempt/features_cleaned.csv', index=False)
race_outcome.to_csv(r'C:/Users/Compute2/Desktop/Jany/TASK/Results/initial attempt/race_outcome_cleaned.csv', index=False)