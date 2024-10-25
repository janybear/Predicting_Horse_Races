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

#1. cleaning, numerical encoding/categorizing, or normalizing (to [0,1]):
    
#dropped columns
features = results_rawdata.drop(['race_city','horse_name','race_date', 'horse_sire','horse_dam', 'hors_broodmare_sire','horse_margin', 'horse_late_start', 'Horse_starting_box_no' ] , axis=1)
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

# #race_city
# position = features.columns.get_loc('race_city')
# race_city_encoded = pd.get_dummies(features['race_city']).astype(int) #one-hot encoding of the unique values
# features = features.drop('race_city', axis=1) # dropping original column
# features = pd.concat([features.iloc[:, :position], race_city_encoded, features.iloc[:, position:]], axis=1)

#could normalize horse_win_value per race to the max
features['horse_win_value'] = normalize(features, 'horse_win_value')

features['horse_psf_rate'] = normalize(features, 'horse_psf_rate')
features['horse_psf_rank'] = normalize(features, 'horse_psf_rank')
features['horse_rate'] = normalize(features, 'horse_rate')

#horse_sex
features = pd.get_dummies(features, columns=['horse_sex'])
# position = features.columns.get_loc('horse_sex')
# horse_sex_encoded = pd.get_dummies(features['horse_sex']).astype(int)
# features = features.drop('horse_sex', axis=1) # dropping original column
# features = pd.concat([features.iloc[:, :position], horse_sex_encoded, features.iloc[:, position:]], axis=1)

# #checking how much info we can get from the jockey-trainer-owner relations
# for value in unique_values: #one-hot encoding of the unique values  
#     features.insert(position, value, features['jockey_name'].apply(lambda x: 1 if value in x.split() else 0))
#     position += 1 
# features = features.drop('jockey_name', axis=1) # dropping original column

# encoder = ce.BinaryEncoder(cols=['jockey_name']) #binary encoding
# features = encoder.fit_transform(features)
#features = pd.get_dummies(features, columns=['jockey_name'])

#a jockey is also an owner in one data point here
#one jockey is also a trainer here

#horse_margins - dropped due to tooooo much missing info for now
# features['horse_margin'] = features['horse_margin'].str.replace(' Lengths', '')
# features['horse_margin'] = features['horse_margin'].str.replace('Half', '0.5')
# features['horse_margin'] = pd.to_numeric(features['horse_margin'], errors='coerce') #convert to numeric

#checking no. of unique categories
no_unique_cat(features, 'horse_origin')
no_unique_cat(features, 'horse_trainer')
no_unique_cat(features, 'jockey_name')
no_unique_cat(features, 'horse_owner')
no_unique_cat(features, 'horse_accessories')

# #horse_accessories
position = features.columns.get_loc('horse_accessories')
unique_values = set(features['horse_accessories'].unique())
for value in unique_values: #one-hot encoding of the unique values  
    features.insert(position, value, features['horse_accessories'].apply(lambda x: 1 if value in x.split() else 0))
    position += 1 
features = features.drop('horse_accessories', axis=1) # dropping original column



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