#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# In[82]:


df = pd.read_csv('Telco_Customer_Dataset.csv')


# **Exploratory Data Analysis**

# In[3]:


#to show all columns
pd.options.display.max_columns = None

Let's get more insight from data!
# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.sample().T


# In[8]:


df.nunique()


# In[9]:


df.dtypes

'SeniorCitizen' variables type may be object 
# In[83]:


df['SeniorCitizen']= df['SeniorCitizen'].astype(object)

Convert 'TotalCharges' variable to numeric 
# In[84]:


df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')


# In[12]:


df.describe().T.style.background_gradient(cmap='RdPu',subset=['mean','std','max','min']) 


# In[13]:


df.isnull().sum()

Let's impute the null rows!#you can drop them immediately
df.dropna(inplace=True)#you can fill them mean value of variable
df.TotalCharges.fillna(df.TotalCharges.mean(), inplace=True)
# In[85]:


#you can look more carefully and notice 
#This customers stayed 0 month with us in this way MonthlyCharges = TotalCharges for them
df[["tenure","MonthlyCharges","TotalCharges"]].sort_values(by=['tenure'], ascending=True).head(15)


# In[86]:


#fill the null rows with MonthlyCharges
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

Let's get rid of duplicates
# In[7]:


df.duplicated().sum()


# In[17]:


df.drop_duplicates(keep=False, inplace=True)


# In[18]:


colors = ['olivedrab','red'] 
ax = (df['Churn'].value_counts()*100.0 /len(df))
label = ax.index

#create pie chart
plt.pie(ax, explode=[0,0.05], labels=label, colors = colors, autopct='%2.2f%%', startangle= 80)
plt.show()


# In[23]:


fig = (df['gender'].value_counts(normalize=True) * 100).plot(kind='bar', color=['darkblue','pink'])
fig.set_xlabel('Gender')
fig.set_ylabel('percentage')
fig.set_title('Gender_distribution')

for patch in fig.patches:
    width, height = patch.get_width(), patch.get_height()
    x, y = patch.get_xy()
    # value, (x, y)
    fig.annotate('{}%'.format(round(height, 1)), (x+0.25*width, y+0.5*height), color='white', weight='bold')


# In[87]:


df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns


# In[88]:


#Drop the gender column because it is already examined
df.drop('gender', axis=1, inplace=True)


# In[89]:


#Drop the customerID column because it is not important not depend variable
df.drop('customerid', axis=1, inplace=True)


# In[90]:


cat_cols = df.drop(['churn'], axis=1).select_dtypes(include='object').columns
fig, axes = plt.subplots(nrows = 5,ncols = 3,figsize = (18,24))

for i, item in enumerate(cat_cols): 
    if i < 5:
        ax = sns.countplot(x=item, data=df, ax=axes[i,0], hue='churn') #Show value counts for two cat_variables
        
    elif i >=5 and i < 10:
        ax = sns.countplot(x=item, data=df, ax=axes[i-5,1], hue='churn')
        
    else:
        ax = sns.countplot(x=item, data=df, ax=axes[i-10,2], hue='churn')


# In[91]:


#Contact time  by Churn 
contact_churn = df.groupby(['contract', 'churn']).size().unstack()  
colors = ['olivedrab','red']
ax = (contact_churn.T*100.0/contact_churn.T.sum()).T.plot(kind='bar', width=0.3,color=colors, stacked=True, rot=0, figsize= (10,6))

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':10},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Contract Type',size = 14)


# In[93]:


#Tenure by Contract Type

fig, axs = plt.subplots(nrows = 1, ncols = 3, sharey = True, figsize=(20,7))

colors=['palegreen','forestgreen','darkgreen']

cols = ['Month-to-month', 'One year', 'Two year']

for i, col in enumerate(cols):
    ax = sns.histplot(df[df['contract'] == col]['tenure'], kde = False, bins = 50, ax = axs[i], color = colors[i])
    ax.set_title('Contract Type: ' + col)


# In[94]:


plt.figure(figsize=(8, 6))
ax.legend(["Not Churn","Churn"],loc='upper right')
sns.kdeplot(x='tenure', hue='churn', data=df, shade=True, color="blue")


# In[96]:


ax = sns.kdeplot(df.monthlycharges[(df["churn"] == 'No') ],
                color="Red", shade = True)
ax = sns.kdeplot(df.monthlycharges[(df["churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of monthly charges by churn')


# ### Feature Engineering
Let's convert 'churn' variable to numeric
# In[98]:


df['churn'].replace(to_replace='Yes', value=1, inplace=True)
df['churn'].replace(to_replace='No', value=0, inplace=True)

Let's show the correlation heatmap 
# In[99]:


plt.figure(figsize=(12,10))  
p=sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap = plt.cm.PuBu)
p.set_title('Correlation of Churn Dataset', fontdict={'fontsize':15}, pad=15);


# In[105]:


df.columns


# In[106]:


df_dummies = pd.get_dummies(df)
df_dummies.head()

Let's show the correlation with bars
# In[102]:


plt.figure(figsize=(15,6))
df_dummies.corr()['churn'].sort_values(ascending=False).plot(kind='bar') 

Let's prioritise features
# In[112]:


# Select Categorical variables
df_cat = df.select_dtypes(include = ['object'])
df_cat_cols = df_cat.columns

# if not drop 'churn' and 'customerid' please check it and drop
#select Numeric variavles
df_num = df.select_dtypes(include = ['int64','int32','float64'])

#Drop Churn variable that independent variable 
df_num = df_num.drop(columns=['churn'])

If you want to learn more about encoding and get_dummies() method please check read.me
# In[113]:


for i in df_cat: 
    df_cat = pd.concat([df_cat,pd.get_dummies(df_cat[str(i)],drop_first=True,prefix=str(i))],axis=1)
    #prefix: change variable name
    #drop_first: reduce variable n to n-1


# In[114]:


df_cat


# In[115]:


#Drop categorical columns 
df_cat = df_cat.drop(columns=df_cat_cols)


# In[116]:


df_cat


# In[117]:


df_num

Let's normalize the numeric variables
# In[118]:


from sklearn.preprocessing import MinMaxScaler

features = df_num.columns.values
scaler = MinMaxScaler(feature_range = (0,1)) #normalize 0-1
scaler.fit(df_num) 
df_num = pd.DataFrame(scaler.transform(df_num))
df_num.columns = features


# In[63]:


features

Let's define Inputs and Targets
# In[120]:


df["churn"] = df["churn"].replace({"Yes":1,"No":0})

# get target
y = df['churn'] 

#merge all columns and get inputs
X = pd.concat([df_num,df_cat],axis=1) 

X.head()


# ### Feature Importance

# In[252]:


#pip install boruta

Let's use the RandomForest classifier to see which features most important 
# In[121]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[122]:


forest = RandomForestClassifier(n_estimators=200)
forest.fit(X, y)


# In[123]:


feature_imp = pd.Series(forest.feature_importances_, index= X.columns)
feature_imp.nlargest(10).plot(kind='barh')

Let's use Boruta algorithms to select which feature most important 
# In[124]:


from boruta import BorutaPy

X = X.to_numpy()
y = y.to_numpy()

# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features
feat_selector.fit(X, y)

# check selected features
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)


# In[125]:


# zip my names, ranks, and decisions in a single iterable
feature_ranks = list(zip(df.columns,  feat_selector.ranking_,  feat_selector.support_))


# In[126]:


# iterate through and print out the results
for feat in feature_ranks:
    print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))


# In[ ]:




