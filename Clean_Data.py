import numpy as np, pandas as pd
import matplotlib.pyplot as plt

class GetRandomDataSet:

    # class variable: shared by all instances
    X = pd.DataFrame([[1,2,3,4,5],
                  [23, np.NaN,2,4,5],
                  [3,23,np.NaN, np.NaN],
                  [2,4,6,7,8],
                  [546,352,55,66,7]],
                  columns=['f1','f2','f3','f4','f5'])
    
    # class variable
    y = pd.Series([1,2,3,4,5])

    # constructor
    def __init__(self, no_of_samples = 100 , no_of_features = 15):
        # instance variables: unique to each instance
        self.s = no_of_samples 
        self.f = no_of_features

    def get_default_X(self):
        # return the class variable
        return GetRandomDataSet.X, GetRandomDataSet.y

    def get_X_y(self, include_nan=True):
        from sklearn.datasets import make_classification
        mk_class = make_classification
        X, y = mk_class(n_samples = self.s, 
                        n_features = self.f, 
                        n_redundant=2, 
                        n_classes = 3, 
                        n_informative=3, 
                        random_state=1)
        
        # make some of the values NaN when include_nan = True
        if include_nan == True:
            rand_list = []
            for i in range(6):
                r = np.random.randint(low = 0, high = self.s)
                if r not in rand_list:
                    rand_list.append(r)
            for i in rand_list:
                for j in range(self.f):
                    X[i,j] = np.NaN 

        # convert the numpy array to pandas dataframe
        X = pd.DataFrame(X)
        X.columns = ['f'+str(i) for i in range(1,self.f+1)] # add column names
        y = pd.Series(y)
        y.name = 'target'
        return X, y

class CleanData(GetRandomDataSet):

    def __init__(self, X=None, y=None): # default data is None
        if X is None or y is None:
            print("Using default data with NaN values")
            # call the constructor of the parent class
            super().__init__() 
            # get the data from the parent class
            self.X, self.y = self.get_X_y(include_nan=True)
        else: # if data is provided during object creation
            self.X = pd.DataFrame(X)
            self.y = pd.Series(y, name = 'Target')

        # create complete dataset adding target variable
        self.dataset = pd.concat([self.X, self.y], axis=1)
        self.features = self.X.columns
        self.sample_size = self.X.shape[0]
        self.feature_size = self.X.shape[1]

    # get data info:
    def get_info(self):
        print("-"*61)
        print(" "*20+"Data Info:")
        print("-"*61)
        print("No. of samples: \t", self.X.shape[0])
        print("No. of features: \t", self.X.shape[1])
        if len(list(self.X.columns)) < 25:
            print("Feature Names: \t\t", list(self.X.columns))
        else: 
            print("Feature Names: \t\t", list(self.X.columns)[:26], "...truncated")
        print("Targets: \t\t", self.y.unique())
        print("Target Value Count: \t", dict(self.y.value_counts()))
        print("Total Missing values: \t", pd.isna(self.X).sum().sum())
        print("Outliers Detected: \t", self.detect_outliers())
        print("Duplicates: \t\t", self.X.duplicated().sum())
        print("Whether scaled? \t", self.check_scaling())
        print("-"*61)

    # --------------------------------------------------------------
    # Functions Before Separating Features from Target variable: (dataset considered)

    def drop_nan(self): # on dataset
        if self.dataset.isna().sum().sum() > 0:
            self.dataset.dropna(inplace = True)
            return self.dataset
        else:
            return "No missing values found"
        
    def impute_nan(self, method): # on dataset
        # use simple imputer to fill missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=method)
        imputer.fit(self.dataset)
        self.dataset = imputer.transform(self.dataset)
        return self.dataset
        
        
    def detect_outliers(self): # on X
        # use IQR method to detect outliers
        outliers = []
        outlier_count = 0
        for i in self.features:
            Q1,Q3 = self.X[i].quantile([0.25,0.75])
            IQR = Q3 - Q1
            lb = Q1 - 1.5*IQR
            ub = Q3 + 1.5*IQR
            for j in self.X[i]:
                if j > ub or j < lb:
                    outlier_count += 1
                    outliers.append(j)
        return outlier_count, outliers[:6]

    def check_scaling(self): # on X
        # checks for only MinMax or same range of values
        # contain ranges of each feature in a dictionary
        feature_ranges = {}
        for feature in self.features:
            max_val = self.X[feature].max()
            min_val = self.X[feature].min()           
            feature_ranges[feature] = (min_val,max_val) # store the value acc to key
     
        # check if all feature ranges are (0,1) - MinMaxScaler used
        res1 = all((0,1) == feature_ranges[f] for f in self.features)
        # check if feature ranges are all same and don't belong to MinMax
        r = feature_ranges[self.features[0]]
        res2 = all(r == feature_ranges[f] for f in self.features)
        # set scaling
        if res1 == True:
            scaling = "MinMaxScaler"
        elif res2 == True:
            scaling = "All features have same scale: "+str(r)
        else:
            scaling = None
        return scaling
    
print("----class run complete----")