import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# Importing libraries and splitting the dataset 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
# Standard scientific Python imports
import matplotlib.pyplot as plt

class KddCup99DataProcessing:
    def __init__(self) -> None:
        # Appending columns to the dataset and adding a new column name ‘target’ to the dataset.
        self.cols ="""duration, 
            protocol_type, 
            service, 
            flag, 
            src_bytes, 
            dst_bytes, 
            land, 
            wrong_fragment, 
            urgent, 
            hot, 
            num_failed_logins, 
            logged_in, 
            num_compromised, 
            root_shell, 
            su_attempted, 
            num_root, 
            num_file_creations, 
            num_shells, 
            num_access_files, 
            num_outbound_cmds, 
            is_host_login, 
            is_guest_login, 
            count, 
            srv_count, 
            serror_rate, 
            srv_serror_rate, 
            rerror_rate, 
            srv_rerror_rate, 
            same_srv_rate, 
            diff_srv_rate, 
            srv_diff_host_rate, 
            dst_host_count, 
            dst_host_srv_count, 
            dst_host_same_srv_rate, 
            dst_host_diff_srv_rate, 
            dst_host_same_src_port_rate, 
            dst_host_srv_diff_host_rate, 
            dst_host_serror_rate, 
            dst_host_srv_serror_rate, 
            dst_host_rerror_rate, 
            dst_host_srv_rerror_rate"""

        columns = []
        for c in self.cols.split(', '):
            if(c.strip()):
                columns.append(c.strip())
        columns.append('target')
        print(len(columns))

        # Creating a dictionary of attack_types
        self.attacks_types = { 
            'normal': 'normal',
            'back': 'dos',
            'buffer_overflow': 'u2r',
            'ftp_write': 'r2l',
            'guess_passwd': 'r2l',
            'imap': 'r2l',
            'ipsweep': 'probe',
            'land': 'dos',
            'loadmodule': 'u2r',
            'multihop': 'r2l',
            'neptune': 'dos',
            'nmap': 'probe',
            'perl': 'u2r',
            'phf': 'r2l',
            'pod': 'dos',
            'portsweep': 'probe',
            'rootkit': 'u2r',
            'satan': 'probe',
            'smurf': 'dos',
            'spy': 'r2l',
            'teardrop': 'dos',
            'warezclient': 'r2l',
            'warezmaster': 'r2l',
        } 

        # Reading the dataset(‘kddcup.data_10_percent.gz’) and adding Attack Type feature in the training
        # dataset where attack type feature has 5 distinct values i.e. dos, normal, probe, r2l, u2r.
        path = "data_sets/kddcup.data_10_percent.gz"
        self.df = pd.read_csv(path, names = columns)
        # Adding Attack Type column 
        self.df['Attack Type'] = self.df.target.apply(lambda r:self.attacks_types[r[:-1]])
        self.df.head()
        self.df['Attack Type'].value_counts()

        # Shape of dataframe and getting data type of each feature 
        self.df.shape

        # Finding missing values of all features.
        self.df.isnull().sum()
        print("COLUMNS: " + self.df.columns)

        # Finding categorical features 
        num_cols = self.df._get_numeric_data().columns
        cate_cols = list(set(self.df.columns)-set(num_cols))
        cate_cols.remove('target')
        cate_cols.remove('Attack Type')
        print(cate_cols)

        # Data Correlation – Find the highly correlated variables using heatmap and ignore them for analysis.
        self.df = self.df.dropna(axis=1) # drop columns with NaN
        # keep columns where there are more than 1 unique values
        #df = df.loc[:, df.nunique() > 1]
        # Keep only numeric columns
        numeric_df = self.df.select_dtypes(include='number')
        # Calculate correlation matrix
        corr = numeric_df.corr()
        # Plot heatmap
        plt.figure(figsize =(15, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.show()

        # This variable is highly correlated with num_compromised and should be ignored for analysis. 
        #(Correlation = 0.9938277978738366) 
        self.df.drop('num_root', axis = 1, inplace = True) 

        # This variable is highly correlated with serror_rate and should be ignored for analysis. 
        #(Correlation = 0.9983615072725952) 
        self.df.drop('srv_serror_rate', axis = 1, inplace = True) 

        # This variable is highly correlated with rerror_rate and should be ignored for analysis. 
        #(Correlation = 0.9947309539817937) 
        self.df.drop('srv_rerror_rate', axis = 1, inplace = True) 

        # This variable is highly correlated with srv_serror_rate and should be ignored for analysis. 
        #(Correlation = 0.9993041091850098) 
        self.df.drop('dst_host_srv_serror_rate', axis = 1, inplace = True) 

        # This variable is highly correlated with rerror_rate and should be ignored for analysis. 
        #(Correlation = 0.9869947924956001) 
        self.df.drop('dst_host_serror_rate', axis = 1, inplace = True) 

        # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis. 
        #(Correlation = 0.9821663427308375) 
        self.df.drop('dst_host_rerror_rate', axis = 1, inplace = True) 

        # This variable is highly correlated with rerror_rate and should be ignored for analysis. 
        #(Correlation = 0.9851995540751249) 
        self.df.drop('dst_host_srv_rerror_rate', axis = 1, inplace = True) 

        # This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
        #(Correlation = 0.9865705438845669)
        self.df.drop('dst_host_same_srv_rate', axis = 1, inplace = True)

        self.df.head()

        # Feature Mapping – Apply feature mapping on features such as : ‘protocol_type’ & ‘flag’.
        # protocol_type feature mapping
        pmap = {'icmp':0, 'tcp':1, 'udp':2}
        self.df['protocol_type'] = self.df['protocol_type'].map(pmap)
        # flag feature mapping
        fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}
        self.df['flag'] = self.df['flag'].map(fmap)

        #attack type feature mapping
        amap = {'dos':0,'normal':1,'probe':2,'r2l':3,'u2r':4}
        self.df['Attack Type'] = self.df['Attack Type'].map(amap)

        # Remove irrelevant features such as ‘service’ before modelling
        self.df.drop('service', axis = 1, inplace = True)

    def modelling(self):
        # Splitting the dataset
        self.df = self.df.drop(['target', ], axis = 1)
        print(self.df.shape)

        # Target variable and train set
        self.labels = self.df[['Attack Type']]
        self.data = self.df.drop(['Attack Type', ], axis = 1)

        sc = MinMaxScaler()
        self.data = sc.fit_transform(self.data)

    def splitTrainTestData(self):
        # Split test and train data
        self.data_train, self.data_test, self.labels_train, self.labels_test = \
            train_test_split(self.data, self.labels, test_size = 0.33, random_state = 42)
        print(self.data_train.shape, self.data_test.shape)
        print(self.labels_train.shape, self.labels_test.shape)
        self.df.to_csv("ids.csv", index=False)
    
    #Visualization
    def bar_graph(self, feature):
        self.df[feature].value_counts().plot(kind="bar")