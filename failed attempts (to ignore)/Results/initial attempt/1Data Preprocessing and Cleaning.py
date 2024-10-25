# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:43:35 2024

@author: Compute2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


#1. cleaning, numerical encoding/categorizing, or normalizing (to [0,1]):
    
#dropped columns
results_df = results_rawdata.drop(['horse_trainer', 'horse_owner', 'race_city','horse_name','horse_origin', 'horse_sire','horse_dam', 'hors_broodmare_sire','horse_margin', 'horse_late_start', 'horse_accessories', 'Horse_starting_box_no' ] , axis=1)
print(len(results_df))
#horse_starting_box removed because its often random to ensure a fair start for all horses, no info gain here

#removing repeat rows (lowers skewing)
results_df = results_df.drop_duplicates()
print(len(results_df))

#check for missing data
import missingno as msno
msno.matrix(results_rawdata)
plt.show()

#race_date
results_df['race_date'] = results_df['race_date'].str.replace('2024-01-', '')  

#horse finish_time
results_df = results_df.rename(columns={'horse_race_degree': 'finish_time'})
results_df['finish_time'] = results_df['finish_time'].replace('Derecesiz', np.nan)# fill undetermined time 
results_df['finish_time'] = results_df['finish_time'].apply(lambda x: convert_to_milliseconds(x) if pd.notnull(x) else np.nan)
results_df['finish_time'] = normalize(results_df, 'finish_time')

#horse_age
results_df['horse_age'] = results_df['horse_age'].str.replace('y', '')  
results_df['horse_age'] = results_df['horse_age'].apply(pd.to_numeric, errors='coerce')
results_df['horse_age'] = normalize(results_df, 'horse_age')

#jockey_weight
results_df['jockey_weight'] = normalize(results_df, 'jockey_weight')


#could normalize horse_win_value per race to the max
results_df['horse_win_value'] = normalize(results_df, 'horse_win_value')

results_df['horse_psf_rate'] = normalize(results_df, 'horse_psf_rate')
results_df['horse_psf_rank'] = normalize(results_df, 'horse_psf_rank')
results_df['horse_rate'] = normalize(results_df, 'horse_rate')

#horse_sex
position = results_df.columns.get_loc('horse_sex')
horse_sex_encoded = pd.get_dummies(results_df['horse_sex']).astype(int)
results_df = results_df.drop('horse_sex', axis=1) # dropping original column
results_df = pd.concat([results_df.iloc[:, :position], horse_sex_encoded, results_df.iloc[:, position:]], axis=1)

# #checking how much info we can get from the jockey-trainer-owner relations
unique_values = set(results_df['jockey_name'].unique())
position = results_df.columns.get_loc('jockey_name')
# print("no. of unique jockeys: ", len(unique_values))
for value in unique_values: #one-hot encoding of the unique values  
    results_df.insert(position, value, results_df['jockey_name'].apply(lambda x: 1 if value in x.split() else 0))
    position += 1 
results_df = results_df.drop('jockey_name', axis=1) # dropping original column

#a jockey is also an owner in one data point in 'horse_owner'


#ensure all results_df are numerical:
results_df = results_df.apply(pd.to_numeric, errors='coerce')
results_df['result'] = results_df['result']-1 #convert classes to python 0-based indexing


# determining results_df and target
max_no_of_racers=int(max(results_df['result']))
bin_counts = results_df['result'].value_counts().sort_index()
# Create the histogram plot with Seaborn
plt.figure(dpi=300)  # Increase DPI for high-resolution
sns.barplot(x=bin_counts.index, y=bin_counts.values, color='cornflowerblue')
plt.xlabel('Number of horses in a single race')
plt.ylabel('Frequency')
plt.title('Histogram of Number of horses in a single race')
plt.show()

indicesof0 = results_df[results_df['result'] == 0].index #removing rows where result=0
results_df = results_df.drop(indicesof0)
race_outcome = results_df[['result']].copy()
features = results_df.drop('result', axis=1)





#4. feature engineering





#check for missing data
import missingno as msno
msno.matrix(results_df)
plt.show()

#SAVE TO CSV
results_df.to_csv(r'C:\Users\Compute2\Desktop\Jany\TASK\XGB Cleaned Data\results_df_cleaned.csv', index=False)
race_outcome.to_csv(r'C:\Users\Compute2\Desktop\Jany\TASK\XGB Cleaned Data\race_outcome_cleaned.csv', index=False)
features.to_csv(r'C:\Users\Compute2\Desktop\Jany\TASK\XGB Cleaned Data\features_cleaned.csv', index=False)