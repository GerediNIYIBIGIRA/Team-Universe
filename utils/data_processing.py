# import pandas as pd

# def load_data():
#     file_path = 'data/Labour force survey_2019.dta'
#     data = pd.read_stata(file_path)
#     relevant_columns = [
#         'province', 'code_dis', 'age5', 'age10', 'usualhrs', 'act_hrs', 
#         'work_agr', 'LFS_workforce', 'TVT2', 'UD', 'A04', 'A01'
#     ]
#     data = data[relevant_columns]
#     return data.ffill()

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_data():
    file_path = 'data/youth_labour_df_updated.csv'
    data = pd.read_csv(file_path)
    relevant_columns = [
    #     'A01', 'A04', 'A08', 'B08', 'B16C', 'D01B2', 'E01B1', 'neetyoung',
    #    'attained', 'UR1', 'TRU', 'age5', 'age10', 'work_agr', 'LFS_workforce',
    #    'target_employed16', 'province', 'code_dis', 'birth_year',
    #    'working_year',
      #    'A01', 'A04', 'A08', 'B08', 'B16C', 'D01B2', 'E01B1',
      #  'neetyoung', 'attained', 'UR1', 'TRU', 'age5', 'age10', 'work_agr',
      #  'LFS_workforce', 'target_employed16', 'province', 'code_dis',
      #  'usualhrs', 'acthrs', 'TVT2', 'UD', 'birth_year', 'working_year'
         'A01', 'A04', 'A08', 'B08', 'B16C', 'D01B2', 'E01B1',
       'neetyoung', 'attained', 'UR1', 'TRU', 'age5', 'age10', 'work_agr',
       'LFS_workforce', 'target_employed16', 'province', 'code_dis',
       'usualhrs', 'acthrs', 'TVT2', 'UD', 'birth_year','working_year'
    ]
    data = data[relevant_columns]
    return data

# load_data()


# def run_prediction_model(data):
#     # Target variable is assumed to be 'UD' (unemployment rate)
#     # Use other columns as features
#     X = data.drop(columns=['UD'])
#     y = data['UD']

#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Initialize and fit the model (RandomForestRegressor as an example)
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     # Predict on test set or new data
#     y_pred = model.predict(X_test)

#     # Return predictions and model for further analysis
#     return y_pred, model
