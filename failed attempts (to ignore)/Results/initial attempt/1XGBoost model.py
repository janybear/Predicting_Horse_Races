# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:22:49 2024

@author: Compute2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#MODEL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

#importing cleaned data
results_df = pd.read_csv(r'C:/Users/Compute2/Desktop/Jany/TASK/XGB Cleaned Data/results_df_cleaned.csv', encoding='utf-8' )

######################################################################################

#create a dictionary of dfs, one for each date
race_day_dict = {race_date: group for race_date, group in results_df.groupby('race_date')}
for key, df in race_day_dict.items():
    print(f"Top 5 rows of DataFrame: {key}")
    print(df.head(5))  # Print top 5 rows
    print("---------")


keys = list(race_day_dict.keys())
random.shuffle(keys)
split_index = int(len(keys) *0.8) #24-7

# Create two new dictionaries based on the split
train_dict = {key: race_day_dict[key] for key in keys[:split_index]}
test_dict = {key: race_day_dict[key] for key in keys[split_index:]}
 

models = []  # List to store models for each race
predictions = []  # List to store predictions for each race
#looping to train models
for race_day in train_dict:
    outcomes = race_day[['result']].copy()
    features = race_day.drop('result', 'race_date', axis=1)
    X = features
    y = outcomes
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    # Store the model and its predictions
    models.append(model)
    predictions.append(model.predict(X))
        
        
        
   



#####################################################################################







from xgboost import XGBClassifier

# # wrap XGBClassifier with BaggingClassifier
# bagging_model = BaggingClassifier(
#     base_estimator=xgb_model,  # base model
#     n_estimators=10,  #number of base estimators (number of XGB models)
#     max_samples=0.8,  # use 80% of the data for each estimator
#     max_features=1.0,  # use all features for each estimator
#     bootstrap=True,  
#     random_state=7
# )
# #fit the bagging model
# bagging_model.fit(X_train, y_train)
# # Make predictions
# y_pred = bagging_model.predict(X_test)


xgb_model = XGBClassifier(
    use_label_encoder=True,
    eval_metric='mlogloss',  
    num_class=6,  # Number of classes for mcc
    n_estimators=500,  #  number of trees
    max_depth=10,  # depth of each tree
    learning_rate=0.0001,  # Smaller learning rate to allow more trees
    subsample=0.8,  # % of the data for each tree
    colsample_bytree=0.8,  # % of the features for each tree
    gamma=0,  # No penalty for splitting; encourages deeper trees
    random_state=7
)

eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
y_pred = xgb_model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

cm = confusion_matrix(y_test, y_pred)

total_samples = np.sum(cm)

cm_percent = cm / total_samples * 100

correct_classifications = np.trace(cm)  # sum of TP and TN
incorrect_classifications = total_samples - correct_classifications
correct_percentage = (correct_classifications / total_samples) * 100
incorrect_percentage = (incorrect_classifications / total_samples) * 100

print(f"Percentage of Correctly Classified Data: {correct_percentage:.2f}%")
print(f"Percentage of Incorrectly Classified Data: {incorrect_percentage:.2f}%")


labels = np.array([
    [f"TN \n\n\n{cm_percent[0,0]:.2f}%", f"FP \n\n\n{cm_percent[0,1]:.2f}%"],
    [f"FN \n\n\n{cm_percent[1,0]:.2f}%", f"TP \n\n\n{cm_percent[1,1]:.2f}%"]
])



results = xgb_model.evals_result()
# Plot the training and validation loss
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
plt.figure(dpi=1000)  # Increase DPI for high-resolution image3, 4)
train_accuracy = accuracy_score(y_train, xgb_model.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
#train_test_accuracy_label = f'Train Accuracy: {train_accuracy:.2f}\nTest Accuracy: {test_accuracy:.2f}'
#plt.text(2, 0.4, train_test_accuracy_label, fontsize=12, color='black', 
        # bbox=dict(facecolor='white', alpha=0.5))
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss (mlogloss)')
plt.title('XGBoost Log Loss During Training')
plt.legend()
plt.show()


from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
plt.figure(dpi=1000)
sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues")
plt.title('Confusion Matrix (Percentage)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()