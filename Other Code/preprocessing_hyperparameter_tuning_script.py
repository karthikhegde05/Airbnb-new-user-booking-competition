# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import math
import csv


datadir = "/kaggle/input/airbnb-new-user/"
train_og = pd.read_csv(datadir + "train.csv")
test_og = pd.read_csv(datadir + "test.csv")
sessions = pd.read_csv(datadir + "sessions.csv")


# Extracting instances which belong to class non-NDF for training
train = train_og[train_og['date_first_booking'].notnull()]
test = test_og[test_og['date_first_booking'].notnull()]

# user id for later submission
id_test = test['id'].to_numpy()




class Preprocessing():
    
    def __init__(self, train, test, sessions):
        """
        Instantiate attributes
        """
        self.train = train
        self.test = test
        self.sessions = sessions
        self.y = train['country_destination']
        self.boolSession = False
        self.listCol = [] # list of columns to be trained on 
        self.fieldDict = {}
        

        
        
    def ExtractFeaturesFromSessions(self):
        """
        Extracting features from sessions data file.
        First grouped using user id and then extracted features like count, nunique, etc
        """
        
        self.boolSession = True
        
        # Imputation of the columns of sessions data table
        self.sessions[['user_id', 'action', 'action_type', 'action_detail']] = self.sessions[['user_id', 'action', 'action_type', 'action_detail']].fillna('Missing')
        self.sessions['secs_elapsed'] = self.sessions['secs_elapsed'].fillna(self.sessions['secs_elapsed'].mean())

        # extracting features using 'action' column
        sess_action_nunique = pd.Series(self.sessions.groupby('user_id')['action'].nunique(), name='action_nunique')
        sess_action_count = pd.Series(self.sessions.groupby('user_id')['action'].count(), name='action_count')


        # extracting features using 'action_type' column
        sess_action_type_nunique = pd.Series(self.sessions.groupby('user_id')['action_type'].nunique(), name='action_type_nunique')
        sess_action_type_count = pd.Series(self.sessions.groupby('user_id')['action_type'].count(), name='action_type_count')


        # extracting features using 'action_detail' column
        sess_action_detail_nunique = pd.Series(self.sessions.groupby('user_id')['action_detail'].nunique(), name='action_detail_nunique')
        sess_action_detail_count = pd.Series(self.sessions.groupby('user_id')['action_detail'].count(), name='action_detail_count')

        # extracting features using 'secs_elapsed' column
        sess_secs_elapsed_nunique = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].nunique(), name='sec_elapsed_nunique')
        sess_secs_elapsed_count = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].count(), name='sec_elapsed_count')
        sess_secs_elapsed_mean = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].agg(pd.Series.mean), name='sec_elapsed_mean')
        sess_secs_elapsed_median = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].agg(pd.Series.median), name='sec_elapsed_median')
        sess_secs_elapsed_std = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].agg(pd.Series.std), name='sec_elapsed_std')
        sess_secs_elapsed_skew = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].agg(pd.Series.skew), name='sec_elapsed_skew')
        sess_secs_elapsed_kurt = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].agg(pd.Series.kurtosis), name='sec_elapsed_kurt')
        sess_secs_elapsed_min = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].agg(pd.Series.min), name='sec_elapsed_min')
        sess_secs_elapsed_max = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].agg(pd.Series.max), name='sec_elapsed_max')
        sess_secs_elapsed_sum = pd.Series(self.sessions.groupby('user_id')['secs_elapsed'].agg(pd.Series.sum), name='sec_elapsed_sum')

        # concating all the features generated so far
        sess_ext = pd.concat([sess_action_nunique, sess_action_count,
                              sess_action_type_nunique, sess_action_type_count,
                              sess_action_detail_nunique, sess_action_detail_count,
                              sess_secs_elapsed_nunique, sess_secs_elapsed_count, sess_secs_elapsed_mean, 
                              sess_secs_elapsed_median, sess_secs_elapsed_std, sess_secs_elapsed_skew, sess_secs_elapsed_kurt, 
                              sess_secs_elapsed_min, sess_secs_elapsed_max, sess_secs_elapsed_sum], axis='columns')
        
        def extract_single_top(data):
            """
            If a list is obtained as a feature, take the first value as the particular value.
            """
            if(type(data)==np.ndarray):
                return data[0]
            return data        
        
        
        # take the unique values if had obtained a list
        for col in list(sess_ext.columns):
            sess_ext[col] = sess_ext[col].apply(lambda x: extract_single_top(x))
            
        self.listCol = ['action_nunique',
                       'action_count', 'action_type_nunique',
                       'action_type_count', 'action_detail_nunique',
                       'action_detail_count',  'sec_elapsed_nunique',
                       'sec_elapsed_count', 'sec_elapsed_mean',
                       'sec_elapsed_median', 'sec_elapsed_std', 'sec_elapsed_skew',
                       'sec_elapsed_kurt', 'sec_elapsed_min', 'sec_elapsed_max',
                       'sec_elapsed_sum']


        # Merge the sessions data with train and with test according to the user ids
            
        self.train = self.train.set_index('id')
        self.train = self.train.join(sess_ext, how='inner')
        self.train = self.train.reset_index()
        self.train = self.train.rename(columns={'index':'id'})
        self.test = self.test.set_index('id')
        self.test  = self.test.join(sess_ext)
        self.test = self.test.reset_index()
        
    
        
    
    def CreateFeature(self):
        """
        Create 'month' feature from 'date_first_booking'
        """
        self.train["month"] = self.train["date_first_booking"].apply(lambda x: int(x[5:7]))
        self.test["month"] = self.test["date_first_booking"].apply(lambda x: int(x[5:7]))
    
    
    def MissingValueTreatment(self):
        """
        Imputation of missing values
        """
        train = self.train
        test = self.test
        
        train['age'] = train['age'].fillna(30)
        test['age'] = test['age'].fillna(30)
        train['first_affiliate_tracked'] = train['first_affiliate_tracked'].fillna('none')
        
        # Imputing features generated from sessions data
        if(self.boolSession == True):
            train = self.train
            test = self.test
            
            lst_col = ['action_nunique',
                       'action_count', 'action_type_nunique',
                       'action_type_count', 'action_detail_nunique',
                       'action_detail_count',  'sec_elapsed_nunique',
                       'sec_elapsed_count', 'sec_elapsed_mean',
                       'sec_elapsed_median', 'sec_elapsed_std', 'sec_elapsed_skew',
                       'sec_elapsed_kurt', 'sec_elapsed_min', 'sec_elapsed_max',
                       'sec_elapsed_sum']
            
            lst_miss_treat_cont = [] # will be filled with mean values
            lst_miss_treat_cat = [] # will be filled with mode values
            for col in lst_col:
                if(test[col].nunique()>500):
                    lst_miss_treat_cont.append(col)
                else:
                    lst_miss_treat_cat.append(col) 


            for col in lst_miss_treat_cont:
                train[col] = train[col].fillna(train[col].mean())
                test[col] = test[col].fillna(test[col].mean())

            for col in lst_miss_treat_cat:
                train[col] = train[col].fillna(train[col].mode())
                test[col] = test[col].fillna(test[col].mode())

            # some error causing this to not get filled    
            test['action_nunique'] = test['action_nunique'].fillna(12.00)
            test['action_type_nunique'] = test['action_type_nunique'].fillna(6.00)
            test['action_detail_nunique'] = test['action_detail_nunique'].fillna(12.00)
            
        
        self.train = train
        self.test = test
        
        
    def treatingOutliers(self):
        """
        Treating outliers in 'age' feature
        """
        train = self.train
        test = self.test
        
        train = train[(train['age']>18) & (train['age']<80)]
        test.loc[test.age > 80, 'age'] = 30
        test.loc[test.age < 18, 'age'] = 30
        
        self.train = train
        self.test = test
              

    def CombiningCategories(self):
        """
        Process different categorical classes of categorical features
        according to the frequency
        """
        lst_label_encode = ['gender', 'signup_method', 'signup_flow', 'language', 
                            'affiliate_channel', 'affiliate_provider', 
                            'first_affiliate_tracked', 'signup_app', 
                            'first_device_type', 'first_browser']
        
        train = self.train
        test = self.test
        
        # combining the category classes of features, which are known to have smaller frequency (value_counts)
        train.loc[train.gender == "-unknown-", 'gender'] = "unknown"
        test.loc[test.gender == "-unknown-", 'gender'] = "unknown"
        train.loc[train.first_device_type == "Android Tablet", "first_device_type"] = "Android Phone"
        train.loc[train.first_device_type == "iPad", "first_device_type"] = "iPhone"
        test.loc[test.first_device_type == "Android Tablet", "first_device_type"] = "Android Phone"
        test.loc[test.first_device_type == "iPad", "first_device_type"] = "iPhone"
        train.loc[train.first_browser == "Chrome Mobile", "first_browser"] = "Chrome"
        test.loc[test.first_browser == "Chrome Mobile", "first_browser"] = "Chrome"

        # Combine the categorical classes if not listed for each feature.
        # we also tried modifying these.
        field_dict = {}
        field_dict["gender"] = [ "MALE", "FEMALE"]
        field_dict["signup_method"] = ["basic", 'facebook']
        field_dict["signup_flow"] = [0, 25, 24, 23, 12, 8]
        field_dict["language"] = ['en', 'fr', 'es', 'de', 'it' ]
        field_dict["affiliate_channel"] = list(train["affiliate_channel"].unique())
        field_dict["affiliate_provider"] = ['direct', 'google', 'facebook', 'other', 'padmapper']
        field_dict["first_affiliate_tracked"] = ['untracked', 'omg', 'linked', 'tracked-other', 'product']
        field_dict["signup_app"] = ['Web', "iOS", 'Android', "Moweb"]
        field_dict["first_device_type"] = ["Mac Desktop", "Windows Desktop", "iPhone", "Android Phone" ]
        field_dict["first_browser"] = ["Chrome", "Safari", "Firefox", "Mobile Safari", "IE"]
        
        self.fieldDict = field_dict # updated global fieldDict
        
        def process_field(x,field_list):
            """
            Helper function that takes only the classes listed and combines others.
            """
            if x in field_list:
                return x
            return -1

        for i in lst_label_encode:
            train[i] = train[i].apply(lambda x: process_field(x, field_dict[i]))
            test[i] = test[i].apply(lambda x: process_field(x, field_dict[i]))
            
        self.train = train
        self.test = test


    def LabelEncoding(self):
        """
        Applying label encoding scheme on categorical features
        """ 
        lst_label_encode = ['gender', 'signup_method', 'signup_flow', 'language', 
                            'affiliate_channel', 'affiliate_provider', 
                            'first_affiliate_tracked', 'signup_app', 
                            'first_device_type', 'first_browser']
        
        train = self.train
        test = self.test
        
        self.y = train['country_destination'] # update the y values
        
        label_encoder = LabelEncoder()

        self.listCol.extend(lst_label_encode)
        self.listCol.extend(['month', 'age'])
        train = train[self.listCol] # selecting only required features to train on
        test = test[self.listCol] # retaining only required features 
        
        
        for col in lst_label_encode:
            train[col] = train[col].astype(str)
            test[col] = test[col].astype(str)
            #print(col)
            label_encoder.fit(train[col])
            if(col=='signup_method'):
                label_encoder.classes_ = np.append(label_encoder.classes_, '-1') # count = 101 in test set
            train[col] = label_encoder.transform(train[col])
            test[col] = label_encoder.transform(test[col])
   
            
        # train set is now ready to be trained
        self.train = train
        self.test = test
        
    
    def OneHotEncoding(self):
        """
        Applying one hot encoding scheme on categorical features
        """
        lst_one_hot = ['gender', 'signup_method', 'signup_flow', 'language', 
                        'affiliate_channel', 'affiliate_provider', 
                        'first_affiliate_tracked', 'signup_app', 
                        'first_device_type', 'first_browser']
        train = self.train
        test = self.test
        
        self.y = train['country_destination'] # update the y values
        

        self.listCol.extend(lst_one_hot)
        self.listCol.extend(['month', 'age'])
        
        train = train[self.listCol] # selecting only required features to train on
        test = test[self.listCol] # retaining only required features         
        
        def create_rename_dict(x, field_name):
            """
            helper function for renaming the feature titles after one hot encoding
            """
            rename_dict = {}
            for i in x:
                string = str(i)
                string = field_name + "_" + string 
                rename_dict[i] = string
            rename_dict[-1] = field_name + "_misc"    
            return rename_dict
        
        field_dict = self.fieldDict
        
        for i in lst_one_hot:
            onehot = pd.get_dummies(train[i])
            onehot = onehot.rename(columns = create_rename_dict(field_dict[i], i))
            train[i] = onehot
            onehot = pd.get_dummies(test[i])
            onehot = onehot.rename(columns = create_rename_dict(field_dict[i], i))
            test[i] = onehot
        
        # train set is now ready to be trained
        self.train = train
        self.test = test
        
        
    def PolyTransformer(self):
        """
        For the application of polynomial transformation of top ranked features extracted from 
        feature importance table of first XGBoost model
        """
        
        train = self.train
        test = self.test
        
        
        # this list depends on the categorical encoding scheme and on whether sessions data was used
        list_from_feature_imp = ['first_device_type',
                                 'signup_flow',
                                 'affiliate_channel',
                                 'first_affiliate_tracked',
                                 'language',
                                 'month',
                                 'age',
                                 'affiliate_provider',
                                 'signup_app',
                                 'first_browser']   
        
        # Create the polynomial object with specified degree
        poly_transformer = PolynomialFeatures(degree = 2) # degree is another hyperparameter
        
        poly_features_train = train[list_from_feature_imp]
        poly_features_test = test[list_from_feature_imp]

        poly_transformer.fit(poly_features_train)

        poly_features_train = poly_transformer.transform(poly_features_train)
        poly_features_test = poly_transformer.transform(poly_features_test)

        # renaming the columns of poly_features 
        poly_features_train = pd.DataFrame(poly_features_train, columns=poly_transformer.get_feature_names(list_from_feature_imp))
        poly_features_test = pd.DataFrame(poly_features_test, columns=poly_transformer.get_feature_names(list_from_feature_imp))

        # ignoring the first column - ('1' degree 0 variable)
        poly_features_train = poly_features_train.iloc[:, 1:]
        poly_features_test = poly_features_test.iloc[:, 1:]

        # because ndf value holders were removed we need to do this for aligning indices
        new_index_train = pd.Series(train.index, name='new_index_train')
        new_index_test = pd.Series(test.index, name='new_index_test')

        poly_features_train = pd.concat([new_index_train, poly_features_train], axis = 'columns')
        poly_features_test = pd.concat([new_index_test, poly_features_test], axis = 'columns')

        poly_features_train = poly_features_train.set_index('new_index_train')
        poly_features_test = poly_features_test.set_index('new_index_test')
        
        # drop the original columns and later we will add the features generated from polynomial transformer.
        train = train.drop(list_from_feature_imp, axis='columns')
        test = test.drop(list_from_feature_imp, axis='columns')

        # adding the features generated from polynomial transformer
        # these include the original features as well 
        train = pd.concat([train, poly_features_train], axis='columns')
        test = pd.concat([test, poly_features_test], axis='columns')
        
        # train set is now ready to be trained
        self.train = train
        self.test = test

    
    def get_train_data(self):
        return self.train
    
    def get_test_data(self):
        return self.test
    
    def get_y(self):
        return self.y
      
        
# Cleaning and preprocessing data

Cleaned_data = Preprocessing(train, test, sessions)
# Cleaned_data.ExtractFeaturesFromSessions() # uncomment to consider sessions data for the training
Cleaned_data.CreateFeature()
Cleaned_data.MissingValueTreatment()
Cleaned_data.treatingOutliers()
Cleaned_data.CombiningCategories()
Cleaned_data.LabelEncoding()
# Cleaned_data.PolyTransformer() # uncomment to pass features into polynomial transformer

X_train = Cleaned_data.get_train_data()
y = Cleaned_data.get_y()
X_test = Cleaned_data.get_test_data()
        
        
        
# Create predictions according to the competition requirement

targets = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NL', 'PT', 'US','other']

def create_preds(y_probs):
    y_pred = []
    for i in y_probs:
        max = [0,0,0]

        for j in range(len(i)):
            if(i[j] > i[max[0]]):
                max[2] = max[1]
                max[1] = max[0]
                max[0] = j
            elif(i[j] > i[max[1]]):
                max[2] = max[1]
                max[1] = j
            elif(i[j] > i[max[2]]):
                max[2] = j

        y_pred.append([targets[max[0]], targets[max[1]], targets[max[2]] ])
    
    return y_pred



# Computes NDGC metric on the predictions 
def ndgc(preds, truth):
    score = 0
    for i in range(len(preds)):
        
        for j in range(3):
            if(preds[i][j] == truth[i]):
                score += 1/math.log2(2+j)
                
    score = score/len(truth)
    return score


# xgboost with tuned parameters
model = xgb.XGBClassifier(n_estimators = 800,
                          eta = 0.001, 
                          reg_lambda = 1000, 
                          max_depth = 3, 
                          gamma = 3,
                          tree_method = "gpu_hist")



class ValidationMethods():
    
    def __init__(self, train, test, y, model):
        self.train = train
        self.test = test
        self.y = y
        self.model = model
        
        
    def timeSeriesSplit(self):
        """
        TimeSeriesSplit (a.k.a Nested CV) gives a good judgement for time series data
        """
        
        train = self.train
        y = self.y
        
        train_ndgc = []
        validation_ndgc = []

        
        tscv = TimeSeriesSplit(max_train_size=None, n_splits=5)
        
        for train_index, test_index in tscv.split(train):
            print("started new")
            X_train, X_test = train.iloc[train_index], train.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.model.fit(X_train, y_train)

            # prediction on training subset
            train_probs = self.model.predict_proba(X_train)
            train_preds = create_preds(train_probs)

            # prediction on validation subset
            validation_probs = self.model.predict_proba(X_test)
            validation_preds = create_preds(validation_probs)

            train_ndgc.append(ndgc(train_preds, y_train.to_numpy()))
            validation_ndgc.append(ndgc(validation_preds, y_test.to_numpy()))

        print('training ndgc score: {}'.format(np.mean(train_ndgc)))    
        print ('Nested TS Validation ndgc score: {}'.format(np.mean(validation_ndgc)))
        
    def timeSeriesSplitWithSmote(self):
        """
        Validation for the Application of Smote on data (after generating features from sessions data)
        """
        train = self.train
        y = self.y
        
        train_ndgc = []
        validation_ndgc = []

        tscv = TimeSeriesSplit(max_train_size=None, n_splits=3)
        sm = SMOTE(sampling_strategy = 'minority', random_state=21, k_neighbors=2, n_jobs=-1)
        
        for train_index, test_index in tscv.split(train):
            print("started new")
            X_train, X_test = train.iloc[train_index], train.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_sampled_subset, y_sampled_subset = sm.fit_sample(X_train, y_train)
            print(X_sampled_subset.shape)

            self.model.fit(X_sampled_subset, y_sampled_subset)


            # prediction on training subset
            train_probs = self.model.predict_proba(X_sampled_subset)
            train_preds = create_preds(train_probs)

            # prediction on validation subset
            validation_probs = self.model.predict_proba(X_test)
            validation_preds = create_preds(validation_probs)

            train_ndgc.append(ndgc(train_preds, y_sampled_subset.to_numpy()))
            validation_ndgc.append(ndgc(validation_preds, y_test.to_numpy()))

        print('training ndgc score: {}'.format(np.mean(train_ndgc)))    
        print ('Nested TS Validation ndgc score: {}'.format(np.mean(validation_ndgc)))
        
    
    def KFoldCrossValidation(self):
        """
        K-fold cross validation
        """
        
        train = self.train
        y = self.y
        
        train_ndgc = []
        cross_val_ndgc = []
        
        skf = StratifiedKFold(n_splits=5)
        for train_index, validation_index in skf.split(train, y):
            print("Started new")
            X_train = train.iloc[train_index]
            y_train = y.iloc[train_index].to_numpy()
            X_test = train.iloc[validation_index]
            y_test = y.iloc[validation_index].to_numpy()

            self.model.fit(X_train, y_train) 

            # prediction on training subset
            train_probs = self.model.predict_proba(X_train)
            train_preds = create_preds(train_probs)

            # prediction on validation subset
            validation_probs = self.model.predict_proba(X_test)
            validation_preds = create_preds(validation_probs)

            # appending to the train_ndgc list
            train_ndgc.append(ndgc(train_preds, y_train))

            # appending to the cross_val_ndgc list
            cross_val_ndgc.append(ndgc(validation_preds, y_test))

        print('training ndgc score: {}'.format(np.mean(train_ndgc)))    
        print ('Cross Validated ndgc score: {}'.format(np.mean(cross_val_ndgc)))

        
    def HoldOutSetValidation(self):
        """
        80% - 20% train test split for validation
        """
        
        train = self.train
        y = self.y
        
        X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        # prediction on training subset
        train_probs = self.model.predict_proba(X_train)
        train_preds = create_preds(train_probs)

        # prediction on validation subset
        validation_probs = self.model.predict_proba(X_test)
        validation_preds = create_preds(validation_probs)

        print('training ndgc score: {}'.format(ndgc(train_preds, y_train.to_numpy())))    
        print ('Hold out set Validation ndgc score: {}'.format(ndgc(validation_preds, y_test.to_numpy())))
        
    
    def BayesianOptOnCV(self):
        """
        We tried Bayesian Optimization with k-fold cross validation
        Here it is implemented as a mehtod
        """
        
        skf = StratifiedKFold(n_splits=5)
        train = self.train
        y = self.y
        
        
        param_hyperopt= {
                        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(1)),
                        'max_depth': scope.int(hp.quniform('max_depth', 3, 5, 1)),
                        'n_estimators': 80,
                        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
                        'reg_lambda': hp.quniform('reg_lambda', 700, 900, 10),
                        'gamma' : hp.quniform('gamma', 0, 20, 1),
                        'tree_method' : "gpu_hist"
                        }
        
        cross_val_ndgc = []
        def objective_function(params):
            model = xgb.XGBClassifier(**params)
            for train_index, validation_index in skf.split(X, y):

                X_train = train.iloc[train_index]
                y_train = y.iloc[train_index]
                X_test = X.iloc[validation_index]
                y_test = y.iloc[validation_index].to_numpy()

                self.model.fit(X_train, y_train) 
                validation_probs = self.model.predict_proba(X_test)
                validation_preds = create_preds(validation_probs)


                cross_val_ndgc.append(ndgc(validation_preds, y_test))  

            score = np.mean(cross_val_ndgc)
            return {'loss': -score, 'status': STATUS_OK}
    
        
        def bayesian_optimization():
            trials = Trials()
            best_param = fmin(objective_function, 
                              param_hyperopt, 
                              algo=tpe.suggest, 
                              max_evals=20, 
                              trials=trials,
                              rstate= np.random.RandomState(1))

            loss = [x['result']['loss'] for x in trials.trials]
            best_param_values = [x for x in best_param.values()]
            return best_param
        
        
        best_param = bayesian_optimization()
    
    
    def ApplySmote(self):
        """
        If validation is made in terms of TimeSeriesSplit with smote then apply smote to the entire data
        """
        sm = SMOTE(sampling_strategy = 'minority', random_state=21, k_neighbors=2, n_jobs=-1)
        
        self.train, self.y = sm.fit_sample(self.train, self.y)
    
    
    def TrainEntireSet(self):
        """
        After hyperparameter tuning, train the best model on the entire set to get better score
        """
        # ApplySmote(self) # uncomment this if wanted smote
        self.model.fit(self.train, self.y)
        test_probs = self.model.predict_proba(self.test)
        test_pred = create_preds(test_probs)
    
        # returning the predictions of the model on the test set
        return test_pred
    
    
# Calling validation

Validate_data = ValidationMethods(X_train, X_test, y, model)
Validate_data.timeSeriesSplit()
test_pred = Validate_data.TrainEntireSet()



def create_test_prediction(ids, preds):
    fields = ['id','country_destination']
    final = []
    final.append(fields)
    for i in range(len(ids)):
        for j in range(3):
            final.append([ids[i],preds[i][j]])
    
    ndf = test_og[test_og["date_first_booking"].isnull()]["id"].to_numpy()
    
    for i in ndf:
        final.append([i,"NDF"])
    
    with open("submission.csv", 'w') as f:
        w = csv.writer(f)
        w.writerows(final)
        
        
        
create_test_prediction(id_test, test_pred)

