import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Preprocessing modules
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Metrics
from sklearn.metrics import plot_confusion_matrix, classification_report, plot_roc_curve

# Saving the model
import pickle


# In[298]:


data = pd.read_csv('loan.csv')
data.head()


# In[299]:


df = data.copy()
df.shape


# In[300]:


df.info()


# In[301]:


df['Loan_Status'].value_counts()


# ### Dropping the unique identifying feature: Loan_ID

# In[302]:


df.drop(['Loan_ID'], axis=1, inplace=True)


# ## Exploratory Data Analysis (EDA)

# ### 1. Checking for Multi-collinearity

# In[303]:


corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='RdYlGn')
plt.show()


# ### There isn't enough correlation among the input features in order to drop any of them

# ### 2. Checking for Missing Values

# In[304]:


df.isnull().sum()


# ### There are a lot of missing values present within the dataset

# ### 3. Checking for the presence of Outliers

# In[305]:


plt.figure(figsize=(15, 7))

df.boxplot()
plt.show()


# ### Features with outliers:
# 
# 1. ApplicantIncome
# 
# 2. CoapplicantIncome

# ### Checking the 'Dependent' feature

# In[306]:


# The 'Dependents' feature contains string values with one values: 3+
df['Dependents'].unique()


# ## Plan for Preprocessing the data
# 
# A] For Outliers ==> Will create a Custom Transformer to handle the outliers
# 
# B] Handling Missing Values:
# 
# 1. LoanAmount == Mean
# 
# 2. Gender == Mode
# 
# 3. Married == Mode
# 
# 4. Dependents == Mode
# 
# 5. Self_Employed == Mode
# 
# 6. Loan_Amount_Term == Mode
# 
# 7. Credit_History == Mode
# 
# C] One Hot Encoding Categorical Features: 

# ## X & y Split

# In[307]:


X = df.drop(['Loan_Status'], axis=1)
X.head()


# In[308]:


y = df['Loan_Status']
y.head()


# In[309]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[310]:


X_train.shape, X_test.shape


# ### Dealing with the 'Dependents' feature

# In[311]:


X_train['Dependents'].head()


# In[312]:


dep_map = {
    '0':0,
    '1':1,
    '2':2,
    '3+':3
}


# In[313]:


X_train['Dependents'] = X_train['Dependents'].map(dep_map)


# In[314]:


X_train['Dependents'].head()


# ### Building the Preprocessing Pipeline for Handling Outliers & Missing Values

# In[315]:


outliers_features = ['ApplicantIncome', 'CoapplicantIncome']

mean_features = ['LoanAmount']

mode_features = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']


# ### Creating a Custom Transformer to handle Outliers

# In[316]:


# My custom Transformer to handle the Outliers

from sklearn.base import TransformerMixin
import pandas as pd
pd.options.mode.chained_assignment = None  # To ignore the warning for not returning a copied DataFrame

class HandleOutliers(TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        from scipy.stats import shapiro

        for col in X.columns : # Looping through all columns within the given DataFrame
            
            # If p-value < 0.05 == Skewed Distribution, else Normal Distribution
            
            if shapiro(X[col]).pvalue < 0.05 :

                # IQR method to handle outliers with Skewed Distribution
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)

                iqr = q3 - q1

                lower_boundary = q1 - 1.5 * q1
                upper_boundary = q3 + 1.5 * q3

                X.loc[X[col] <= lower_boundary, col] = lower_boundary
                X.loc[X[col] >= upper_boundary, col] = upper_boundary

                
            else :

                # 3-Sigma method to handle outliers with Normal Distribution
                lower_boundary = X[col].mean() - 3 * X[col].std()
                upper_boundary = X[col].mean() + 3 * X[col].std()

                X.loc[X[col] <= lower_boundary, col] = lower_boundary
                X.loc[X[col] >= upper_boundary, col] = upper_boundary
                
        return X


# In[317]:


outliers_pipe = Pipeline([
    ('Outliers', HandleOutliers())
])

mean_pipe = Pipeline([
    ('Mean_Imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))
])

mode_pipe = Pipeline([
    ('Mode_Imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
])


# In[318]:


preprocess_pipe = ColumnTransformer([
    ('Handle_Outliers', outliers_pipe, outliers_features),
    ('Impute_Mean', mean_pipe, mean_features),
    ('Impute_Mode', mode_pipe, mode_features)
], remainder='passthrough')


# In[319]:


X_train_preprocessed = preprocess_pipe.fit_transform(X_train)


# In[320]:


X_train_preprocessed


# In[321]:


X_train_preprocessed.shape


# In[322]:


X_train_preprocessed[0]


# In[323]:


X_train.head()


# In[324]:


X_train.shape


# In[325]:


cols = outliers_features+mean_features+mode_features+['Education', 'Property_Area']


# In[326]:


len(cols)


# In[327]:


X_train = pd.DataFrame(X_train_preprocessed, columns=cols)


# In[328]:


X_train.head()


# In[329]:


X_train.isnull().sum()


# In[330]:


X_train.info()


# In[331]:


# Converting 'object' to 'float32'
to_int = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Dependents', 'Loan_Amount_Term', 'Credit_History']

for col in X_train.columns:
    if col in to_int:
        X_train[col] = X_train[col].astype(np.float32)


# In[332]:


X_train.info()


# ### One Hot Encoding

# In[333]:


# Appending all the features with dtype == 'object' (Categorical Features) in a list
cat_features = []

for col in X_train.columns :
    if X_train[col].dtype == 'object' :
        cat_features.append(col)
        
cat_features


# In[334]:


for col in X_train.columns:
    if col in cat_features:
        print(X_train[col].unique())


# In[335]:


dummies_gender = pd.get_dummies(X_train['Gender'], drop_first=True)
dummies_married = pd.get_dummies(X_train['Married'], prefix = 'Married', drop_first=True)
dummies_self_emp = pd.get_dummies(X_train['Self_Employed'], prefix = 'Self_Employed', drop_first=True)
dummies_edu = pd.get_dummies(X_train['Education'], drop_first=True)
dummies_prop = pd.get_dummies(X_train['Property_Area'], drop_first=True)


# In[336]:


encode_df = pd.concat([dummies_gender, dummies_married, dummies_self_emp, dummies_edu, dummies_prop], axis=1)
encode_df.head()


# In[337]:


X_train.drop(cat_features, axis=1, inplace=True)


# In[338]:


X_train = pd.concat([X_train, encode_df], axis=1)
X_train.head()


# In[339]:


# Conventioning for simplicity
X_train['Not_Graduate'] = X_train['Not Graduate']
X_train.drop(['Not Graduate'], axis=1, inplace=True)


# In[340]:


X_train.head()


# ## Preprocessing the Test Data separately

# In[341]:


X_test.head()


# In[342]:


X_test.shape


# In[343]:


X_test['Dependents'] = X_test['Dependents'].map(dep_map)

# Using transform method for Test Data
X_test_preprocessed = preprocess_pipe.transform(X_test)

X_test = pd.DataFrame(X_test_preprocessed, columns=cols)

dummies_gender = pd.get_dummies(X_test['Gender'], drop_first=True)
dummies_married = pd.get_dummies(X_test['Married'], prefix = 'Married', drop_first=True)
dummies_self_emp = pd.get_dummies(X_test['Self_Employed'], prefix = 'Self_Employed', drop_first=True)
dummies_edu = pd.get_dummies(X_test['Education'], drop_first=True)
dummies_prop = pd.get_dummies(X_test['Property_Area'], drop_first=True)

encode_df = pd.concat([dummies_gender, dummies_married, dummies_self_emp, dummies_edu, dummies_prop], axis=1)

X_test.drop(cat_features, axis=1, inplace=True)

X_test = pd.concat([X_test, encode_df], axis=1)

X_test['Not_Graduate'] = X_test['Not Graduate']

X_test.drop(['Not Graduate'], axis=1, inplace=True)


# In[344]:


X_test.head()


# ## Target Mappings

# In[345]:


y_train.unique()


# In[346]:


y_map = {
    'N':0,
    'Y':1
}


# In[347]:


y_train = y_train.map(y_map)


# In[348]:


y_test = y_test.map(y_map)


# ## Model Bulding

# ### 1. RandomForest Model

# In[349]:


rf = RandomForestClassifier()


# In[350]:


rf.fit(X_train, y_train)


# In[351]:


rf.score(X_test, y_test)


# ### 2. AdaBoost Model

# In[352]:


ad = AdaBoostClassifier()


# In[353]:


ad.fit(X_train, y_train)


# In[354]:


ad.score(X_test, y_test)


# ## Tuning the Hyperparameters of AdaBoost Model

# In[355]:


params = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 1],
    'n_estimators': [10, 25, 30, 50, 70, 100]
}


# In[356]:


rs_cv = RandomizedSearchCV(ad, params, cv=5, n_iter=8)


# In[357]:


best_model = rs_cv.fit(X_train, y_train)


# In[358]:


best_model.score(X_test, y_test)


# In[359]:


best_model.best_estimator_


# ## Model Performance

# In[360]:


y_pred = best_model.predict(X_test)


# In[361]:

def values():
    return X_test, y_test


plot_confusion_matrix(best_model, X_test, y_test)
plt.title('Confusion Matrix\n')
plt.show()
image_path = os.path.join('static', 'confuse.png')
    
if os.path.exists(image_path):
  os.remove(image_path)
plt.savefig(image_path)

plt.close()


# In[362]:


print("Classification Report:\n\n", classification_report(y_test, y_pred))


# In[363]:


plot_roc_curve(best_model, X_test, y_test)
plt.title('ROC-AUC\n')
plt.show()


# ## Saving (Dumping) the model

# In[364]:


# open a file, where you want to store the data
file = open('loan_eligibility_adaboost.pkl', 'wb')

# dump information to that file
pickle.dump(best_model, file)


# In[365]:


model = open('loan_eligibility_adaboost.pkl', 'rb')


# ## Predictions

# In[366]:


X_test.tail()


# In[367]:


y_test.tail()


# In[368]:


best_model.predict([[4053.0,2426.0,158,1,360,0,1,0,1,0,1,1]])


# In[369]:


best_model.predict([[1958.0,1456.0,60,2,300,1,1,1,0,0,1,1]])


# In[370]:




# In[ ]:





# In[ ]:





# In[ ]:




