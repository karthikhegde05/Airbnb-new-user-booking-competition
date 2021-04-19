
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import csv

#class for final pipeline
class Pipeline:
    
    def __init__(self, dataset, preprocessor, model):
        
        self.dataset = dataset                             
        self.preprocessor = preprocessor
        self.model = model
        
    #calls the preprocessor function on the dataset
    #outputs explained:
    #X: training features
    #y: ground truth
    #X_test: test features
    #test_id: the ids corresponding to X_test. Needed for creating final csv files
    #test_id_ndf: the ids for ndf found using date_first_booking = NULL
    def process(self):
        
        self.X , self.y, self.X_test, self.test_id, self.test_id_ndf = preprocessor(self.dataset)
        
    #train the model on processed data    
    def train(self):
        
        self.model.fit(self.X, self.y)
        
    #create the predictions on the test data    
    def predict(self):
        
        targets = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NL', 'PT', 'US','other']

        y_probs = self.model.predict_proba(self.X_test)
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
        
        self.predictions = y_pred
            
    #create the final csv for submission        
    def create_csv(self, filename):
    
        fields = ['id','country_destination']
        final = []
        final.append(fields)
        for i in range(len(self.test_id)):
            for j in range(3):
                final.append([self.test_id[i],self.predictions[i][j]])

        for i in self.test_id_ndf:
            final.append([i,"NDF"])

        with open(filename + ".csv", 'w') as f:
            w = csv.writer(f)
            w.writerows(final)

#class for dataset            
class Dataset:

    def __init__(self, datadir):
        # datadir = "/kaggle/input/airbnb-new-user/"
        self.train = pd.read_csv(datadir + "train.csv")
        self.test = pd.read_csv(datadir + "test.csv")
        self.session = pd.read_csv(datadir + "sessions.csv")


#helper function for pre processing.
def process_field(x,field_list):
     if x in field_list:
        return x
     return -1

#helper function for pre processing
def create_rename_dict(x, field_name):
    
    rename_dict = {}
    for i in x:
        string = str(i)
        string = field_name + "_" + string 
        rename_dict[i] = string
    rename_dict[-1] = field_name + "_misc"    
    return rename_dict

#helper function for pre processing
def used_translate(x,a):
    if x in a:
        return 1
    return 0

#function to undertake all preprocessing required
def preprocessor(dataset):

    train = dataset.train
    test = dataset.test
    session = dataset.session
    
    #seperate ndf from remaining
    test_id_ndf = test[test['date_first_booking'].isnull()].id.to_list()

    train = train[train['date_first_booking'].notnull()]
    test = test[test['date_first_booking'].notnull()]  
    
    translated = session[(session["action"] == 'ajax_google_translate') |(session["action"] == 'ajax_google_translate_description')|(session["action"] == 'ajax_google_translate_reviews')].user_id.to_list()
    
    
    #list to specify which features to train on. We can add and remove based on what we want
    features =  ['language']
    

    #some binning to reduce unecessary unique values
    train.loc[train.gender == "-unknown-", 'gender'] = "unknown"
    test.loc[test.gender == "-unknown-", 'gender'] = "unknown"
    train.loc[train.first_device_type == "Android Tablet", "first_device_type"] = "Android Phone"
    test.loc[test.first_device_type == "Android Tablet", "first_device_type"] = "Android Phone"
    train.loc[train.first_device_type == "iPad", "first_device_type"] = "iPhone"
    test.loc[test.first_device_type == "iPad", "first_device_type"] = "iPhone"
    train.loc[train.first_browser == "Chrome Mobile", "first_browser"] = "Chrome"
    test.loc[test.first_browser == "Chrome Mobile", "first_browser"] = "Chrome"

    #specify which unique values get their own one hot encoding. Values not in the list will all be binned into misclaneous
    field_dict = {}
    field_dict["gender"] = [ "MALE", "FEMALE"]
    field_dict["signup_method"] = ["basic", 'facebook']
    field_dict["signup_flow"] = [0, 25, 24, 23, 12, 8]
    field_dict["language"] = ['en', 'fr', 'it' ]
    field_dict["affiliate_channel"] = list(train["affiliate_channel"].unique())
    field_dict["affiliate_provider"] = ['direct', 'google', 'facebook', 'other', 'padmapper']
    field_dict["first_affiliate_tracked"] = ['untracked', 'omg', 'linked', 'tracked-other', 'product']
    field_dict["signup_app"] = ['Web', "iOS", 'Android', "Moweb"]
    field_dict["first_device_type"] = ["Mac Desktop", "Windows Desktop", "iPhone", "Android Phone" ]
    field_dict["first_browser"] = ["Chrome", "Safari", "Firefox", "Mobile Safari", "IE"]     
    
    #Processes each field according to the field dicts specified above. Here the misclaneous values are binned
    for i in features:
        train.loc[:,i] = train[i].apply(lambda x: process_field(x, field_dict[i]))
        test.loc[:,i] = test[i].apply(lambda x: process_field(x, field_dict[i]))
    
    #
    train["used_translate"] = train["id"].apply(lambda x: used_translate(x, translated))    
    test["used_translate"] = test["id"].apply(lambda x: used_translate(x, translated))
    
    #create new dataframe for final processed features
    X = pd.DataFrame()
    X_test = pd.DataFrame()
    test_id = test['id'].to_list()
    
    
    #perform one hot encoding
    for i in features:
        onehot = pd.get_dummies(train[i])
        onehot = onehot.rename(columns = create_rename_dict(field_dict[i], i))
        X = pd.concat([X, onehot], axis=1, sort=False)
        
        onehot = pd.get_dummies(test[i])
        onehot = onehot.rename(columns = create_rename_dict(field_dict[i], i))
        X_test = pd.concat([X_test, onehot], axis=1, sort=False) 
        
    X["translate"] = train["used_translate"]
    X_test["translate"] = test["used_translate"]  
    
    #ground truth
    y = pd.DataFrame()
    y["country_destination"] = train["country_destination"]
         
    return X , y, X_test, test_id, test_id_ndf  

model = xgb.XGBClassifier( n_estimators = 100 , eta = 0.5, reg_lambda = 1, tree_method = "gpu_hist")

dataset = Dataset("/kaggle/input/airbnb-new-user/")

pipeline = Pipeline(dataset, preprocessor, model)

pipeline.process()
pipeline.train()
pipeline.predict()
pipeline.create_csv("final_submission")






