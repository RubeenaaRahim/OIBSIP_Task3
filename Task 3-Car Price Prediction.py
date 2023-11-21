#!/usr/bin/env python
# coding: utf-8

# ## CAR PRICE PREDICTION WITH MACHINE LEARNING

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_rows", None,"display.max_columns", None)
warnings.simplefilter(action='ignore')
plt.style.use('seaborn')


# ## Reading the dataset

# In[2]:


#load dataset
df_main = pd.read_csv("car data.csv")


# ## Basic Information about the data

# In[3]:


df_main.head() # Displays the first 5 entries


# In[4]:


df_main.tail() # Displays the last 5 entries


# In[5]:


df_main.shape  #Displays the number of rows and columns in the dataset


# In[6]:


df_main.columns #List out all columns in the dataset


# In[7]:


df_main.info()


# In[8]:


#numerical stats
df_main.describe()  # Basic statistical information about the data


# In[9]:


#missing values
df_main.isna().sum() # Get the number of null or missing values in the data


# In[10]:


df_main.duplicated().sum()   # Get the total number of duplicated entries in the data


# In[11]:


df_main = df_main.drop_duplicates()  # Removes the duplicates


# ##### Duplcate rows are all deleted.

# ## Data Preprocessing

# In[12]:


df_main['Age'] = 2020 - df_main['Year']
df_main.drop('Year',axis=1,inplace = True)


# In[13]:


df_main.rename(columns = {'Selling_Price':'Selling_Price(lacs)','Present_Price':'Present_Price(lacs)','Owner':'Past_Owners'},inplace = True)


# In[14]:


df_main.head()


# In[ ]:





# ### Handling Outliers

# In[15]:


num_cols = df_main[['Selling_Price(lacs)','Present_Price(lacs)','Driven_kms','Age']]

for i in num_cols:
    print(i)
    sns.boxplot(data=df_main[i])
    plt.show()


# In[16]:


num_cols = df_main[['Selling_Price(lacs)','Present_Price(lacs)','Driven_kms','Age']]

for i in num_cols:
    Q1 = df_main[i].quantile(0.25)
    Q3 = df_main[i].quantile(0.75)

    IQR = Q3-Q1
    low = Q1-1.5*IQR
    up = Q3+1.5*IQR

    for j in df_main[i]:
        if j<=low:
            df_main=df_main.replace(j, low)
        if j>=up:
            df_main=df_main.replace(j, up)


# ## Exploratory Data Analysis (EDA)

# #### Univariate Analysis

# In[17]:


df_main.columns


# In[18]:


cat_cols = df_main[['Fuel_Type','Selling_type','Transmission','Past_Owners']]

for i in cat_cols:
    fig = plt.figure(figsize=[10,4])

    plt.subplot(1,2,1)
    sns.countplot(x=i, data=df_main)
    
    
    plt.show()


# In[19]:


sns.pairplot(data=df_main)
plt.show()


# #### Bivariate/Multi-Variate Analysis

# In[20]:


plt.figure(figsize=(10,7))
sns.heatmap(data=df_main.corr(), annot=True, cmap='copper')
plt.show()


# In[21]:


df_main.corr()['Selling_Price(lacs)']


# In[22]:


df_main.pivot_table(values='Selling_Price(lacs)', index = 'Selling_type', columns= 'Fuel_Type')


# In[23]:


df_main.pivot_table(values='Selling_Price(lacs)', index = 'Selling_type', columns= 'Transmission')


# ## Data Preparation

# ### Encoding the Categorical Variables

# In[24]:


df_main.drop(labels='Car_Name',axis= 1, inplace = True) # Dropping irrelevant columns 


# In[25]:


df_main.head()


# In[26]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for i in cat_cols:
    cat_cols[i] = encoder.fit_transform(df_main[i])


# In[27]:


cat_cols.head()


# ### Scaling Numerical Columns

# In[28]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(num_cols)

cols = num_cols.columns
scaled_dataset = pd.DataFrame(scaled_data, columns=cols)


# In[29]:


scaled_dataset.head()


# In[30]:


# Concatenate both encoded categorical dataframe and scaled numerical dataframe.

df_main = pd.concat([cat_cols, scaled_dataset], axis=1)


# In[31]:


df_main.head()


# In[32]:


df_main.isna().sum() # rechecking for null values


# In[33]:


df_main.dropna(inplace=True) # Dropping rows with null values


# ## Model Building

# In[34]:


# Separating target variable and its features
y = df_main['Selling_Price(lacs)']
X = df_main.drop('Selling_Price(lacs)',axis=1)


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# In[37]:


X_train.fillna(X_train.mean(), inplace=True)
y_train.fillna(y_train.mean(), inplace=True)


# ### Model Creation/Evaluation

# #### Applying regression models
# 1. Linear Regression 
# 2. Ridge Regression
# 3. Lasso Regression
# 4. Random Forest Regression
# 5. Gradient Boosting regression

# In[38]:


from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[39]:


CV = []
R2_train = []
R2_test = []

def car_pred_model(model,model_name):
    # Training model
    model.fit(X_train,y_train)
            
    # R2 score of train set
    y_pred_train = model.predict(X_train)
    R2_train_model = r2_score(y_train,y_pred_train)
    R2_train.append(round(R2_train_model,2))
    
    # R2 score of test set
    y_pred_test = model.predict(X_test)
    R2_test_model = r2_score(y_test,y_pred_test)
    R2_test.append(round(R2_test_model,2))
    
    # R2 mean of train set using Cross validation
    cross_val = cross_val_score(model ,X_train ,y_train ,cv=5)
    cv_mean = cross_val.mean()
    CV.append(round(cv_mean,2))
    
    # Printing results
    print("Train R2-score :",round(R2_train_model,2))
    print("Test R2-score :",round(R2_test_model,2))
    print("Train CV scores :",cross_val)
    print("Train CV mean :",round(cv_mean,2))
    
    # Plotting Graphs 
    # Residual Plot of train data
    fig, ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].set_title('Residual Plot of Train samples')
    sns.distplot((y_train-y_pred_train),hist = False,ax = ax[0])
    ax[0].set_xlabel('y_train - y_pred_train')
    
    # Y_test vs Y_train scatter plot
    ax[1].set_title('y_test vs y_pred_test')
    ax[1].scatter(x = y_test, y = y_pred_test)
    ax[1].set_xlabel('y_test')
    ax[1].set_ylabel('y_pred_test')
    
    plt.show()


# #### Standard Linear Regression or Ordinary Least Squares

# In[40]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
car_pred_model(lr,"Linear_regressor.pkl")


# #### Ridge

# In[41]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

# Creating Ridge model object
rg = Ridge()
# range of alpha 
alpha = np.logspace(-3,3,num=14)

# Creating RandomizedSearchCV to find the best estimator of hyperparameter
rg_rs = RandomizedSearchCV(estimator = rg, param_distributions = dict(alpha=alpha))

car_pred_model(rg_rs,"ridge.pkl")


# #### Lasso

# In[42]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV

ls = Lasso()
alpha = np.logspace(-3,3,num=14) # range for alpha

ls_rs = RandomizedSearchCV(estimator = ls, param_distributions = dict(alpha=alpha))


# In[43]:


car_pred_model(ls_rs,"lasso.pkl")


# #### Random Forest

# In[44]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor()

# Number of trees in Random forest
n_estimators=list(range(500,1000,100))
# Maximum number of levels in a tree
max_depth=list(range(4,9,4))
# Minimum number of samples required to split an internal node
min_samples_split=list(range(4,9,2))
# Minimum number of samples required to be at a leaf node.
min_samples_leaf=[1,2,5,7]
# Number of fearures to be considered at each split
max_features=['auto','sqrt']

# Hyperparameters dict
param_grid = {"n_estimators":n_estimators,
              "max_depth":max_depth,
              "min_samples_split":min_samples_split,
              "min_samples_leaf":min_samples_leaf,
              "max_features":max_features}

rf_rs = RandomizedSearchCV(estimator = rf, param_distributions = param_grid)


# In[45]:


car_pred_model(rf_rs,'random_forest.pkl')


# In[46]:


print(rf_rs.best_estimator_)


# #### Gradient Boosting

# In[47]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

gb = GradientBoostingRegressor()

# Rate at which correcting is being made
learning_rate = [0.001, 0.01, 0.1, 0.2]
# Number of trees in Gradient boosting
n_estimators=list(range(500,1000,100))
# Maximum number of levels in a tree
max_depth=list(range(4,9,4))
# Minimum number of samples required to split an internal node
min_samples_split=list(range(4,9,2))
# Minimum number of samples required to be at a leaf node.
min_samples_leaf=[1,2,5,7]
# Number of fearures to be considered at each split
max_features=['auto','sqrt']

# Hyperparameters dict
param_grid = {"learning_rate":learning_rate,
              "n_estimators":n_estimators,
              "max_depth":max_depth,
              "min_samples_split":min_samples_split,
              "min_samples_leaf":min_samples_leaf,
              "max_features":max_features}

gb_rs = RandomizedSearchCV(estimator = gb, param_distributions = param_grid)


# In[48]:


car_pred_model(gb_rs,"gradient_boosting.pkl")


# In[49]:


Technique = ["LinearRegression","Ridge","Lasso","RandomForestRegressor","GradientBoostingRegressor"]
results=pd.DataFrame({'Model': Technique,'R Squared(Train)': R2_train,'R Squared(Test)': R2_test,'CV score mean(Train)': CV})
display(results)


# ## Inference

# * Gradient Boosting Regressor is the best model because it gives the highest R² value (0.91) for predicting car prices on new data (test set).
# 
# * Random Forest and Gradient Boosting Regressors both perform exceptionally well on the training data.

# In[ ]:




