
# coding: utf-8

# In[1]:


## Import warnings. 

import warnings
warnings.filterwarnings("ignore")


# In[2]:


## Import analysis modules

import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc


# In[3]:


## Import visualization modules

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns


# In[4]:


## Read in file

train_original = pd.read_csv('https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv')
train_original.info()


# In[5]:


# Exclude some features to reduce data dimension

train=train_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
total = [train]

train.shape


# In[6]:


# Missing cases for training set:

train.isnull().sum()


# In[7]:


# Age missing cases:

train[train['Age'].isnull()].head()


# In[8]:


# Distribution of Age, condition = Pclass:

train[train.Pclass==1].Age.plot(kind='kde', color='r', label='1st class')
train[train.Pclass==2].Age.plot(kind='kde', color='b', label='2nd class')
train[train.Pclass==3].Age.plot(kind='kde', color='g', label='3rd class')
plt.xlabel('Age')
plt.legend(loc='best')
plt.grid()


# In[9]:


# Create function to replace NaN with the median value for each ticket class:

def fill_missing_age(dataset):
    for i in range(1,4):
        median_age=dataset[dataset["Pclass"]==i]["Age"].median()
        dataset["Age"]=dataset["Age"].fillna(median_age)
        return dataset

train = fill_missing_age(train)


# In[10]:


# Embarked missing cases:

train[train['Embarked'].isnull()]


# In[11]:


# Create Barplot:
sns.barplot(x="Embarked", y="Fare", hue="Sex", data=train)


# In[12]:


# Replace missing cases with C:

train["Embarked"] = train["Embarked"].fillna('C')


# In[13]:


# Re-Check for missing cases:

train.isnull().any()


# In[14]:


# Boxplot for Age:

sns.boxplot(x=train["Survived"], y=train["Age"])


# In[15]:


# discretize Age feature:

for dataset in total:
    dataset.loc[dataset["Age"] <= 9, "Age"] = 0
    dataset.loc[(dataset["Age"] > 9) & (dataset["Age"] <=19), "Age"] = 1
    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 29), "Age"] = 2
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[(dataset["Age"] > 39) & (dataset["Age"] <= 49), "Age"] = 4
    dataset.loc[dataset["Age"] > 49, "Age"] = 5
sns.countplot(x="Age", data=train, hue="Survived")
    


# In[16]:


# Boxplot for Fare:

sns.boxplot(x=train["Survived"], y=train["Fare"])


# In[17]:


# Discretize Fare:

pd.qcut(train["Fare"], 8).value_counts()


# In[18]:


for dataset in total:
    dataset.loc[dataset["Fare"] <= 7.75, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.75) & (dataset["Fare"] <= 7.91), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 9.841), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 9.841) & (dataset["Fare"] <= 14.454), "Fare"] = 3   
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 24.479), "Fare"] = 4
    dataset.loc[(dataset["Fare"] >24.479) & (dataset["Fare"] <= 31), "Fare"] = 5   
    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 69.487), "Fare"] = 6
    dataset.loc[dataset["Fare"] > 69.487, "Fare"] = 7    

sns.countplot(x="Fare", data=train, hue="Survived")


# In[19]:


# Countplot for the number of siblings/spouse:

sns.countplot(x="SibSp", data=train, hue="Survived")


# In[20]:


# Countplot for the number of parents/childrens:

sns.countplot(x="Parch", data=train, hue="Survived")


# In[21]:


# Convert SibSp into binary feature:

for dataset in total:
    dataset.loc[dataset["SibSp"]==0, "SibSp"]=0
    dataset.loc[dataset["SibSp"]!=0, "SibSp"]=1
    
sns.countplot(x="SibSp", data=train, hue="Survived")    
    


# In[22]:


# Convert Parch into binary feature:

for dataset in total:
    dataset.loc[dataset["Parch"]==0, "Parch"]=0
    dataset.loc[dataset["Parch"]!=0, "Parch"]=1
    
sns.countplot(x="Parch", data=train, hue="Survived")    


# In[23]:


# Scikit learn estimators require numeric features:

sex = {'female':0, 'male':1}
embarked = {'C':0, 'Q':1, 'S':2}


# In[25]:


# Convert categorical features to numeric using mapping function:

for dataset in total:
    dataset['Sex'] = dataset['Sex'].map(sex)
    dataset['Embarked'] = dataset['Embarked'].map(embarked)
    
train.head()


# In[27]:


# Total survival rate of train dataset:

survived_cases=0
for i in range(891):
    if train.Survived[i]==1:
        survived_cases =  survived_cases + 1
        
total_survival_rate = float(survived_cases)/float(891)

print('%0.4f' % (total_survival_rate))


# In[28]:


# Survival rate under each feature condition:

def survival_rate(feature):
    rate = train[[feature, 'Survived']].groupby([feature], as_index=False).mean().sort_values(by=[feature], ascending=True)
    sns.factorplot(x=feature, y="Survived", data=rate)


# In[30]:


for feature in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]:
    survival_rate(feature)


# In[31]:


# Inter-relationship between Fare and Pclass:

sns.countplot(x="Fare", data=train, hue="Pclass")


# In[32]:


# Seperate input features from target feature:

x = train.drop("Survived", axis=1)
y = train["Survived"]


# In[33]:


# Split the data into training and validation sets:

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=1)


# In[34]:


# Take a look at the shape:

x_train.shape, y_train.shape


# In[35]:


# Decision Tree Classifier:

clf = DecisionTreeClassifier(random_state=1)


# In[36]:


# Run 10 fold cross validation:

cvs = cross_val_score(clf, x, y, cv=5)
print(cvs)


# In[37]:


# Show cross validation score mean and std:

print("Accuracy: %0.4f (+/- %0.4f)" % (cvs.mean(), cvs.std()*2))


# In[38]:


# Fit the model with data:

clf.fit(x_train, y_train)


# In[39]:


# Accuracy:

acc_decision_tree = round(clf.score(x_train, y_train), 4)
print("Accuracy: %0.4f" %(acc_decision_tree))


# In[40]:


# Predict y given validation set:

predictions = clf.predict(x_test)


# In[41]:


# Take a look at the confusion matrix ([TN,FN],[FP,TP]):

confusion_matrix(y_test, predictions)


# In[42]:


# Precision:

print("Precision: %0.4f" % precision_score(y_test, predictions))


# In[43]:


# Recall score:

print("Recall: %0.4f" % recall_score(y_test, predictions))


# In[44]:


# Print classification report:

print(classification_report(y_test, predictions))


# In[45]:


# Get data to plot ROC Curve:

fp, tp, th = roc_curve(y_test, predictions)
roc_auc = auc(fp, tp)


# In[46]:


# Plot ROC Curve:

plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

