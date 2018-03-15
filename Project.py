# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:34:53 2018

@author: Dhruv
"""
#y_pred,Y_Test,final_s - for error percentage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import section
dataset_1=pd.read_csv('train.csv')
dataset_1 = dataset_1.replace(np.nan, "None")

X=dataset_1.iloc[:,0:80].values
Y=dataset_1.iloc[:,80].values
rows, columns = dataset_1.shape
l_col_name=list(dataset_1)
dict1={}
for i in range(0,len(l_col_name)):
    dict1[i]=l_col_name[i]

#pre-processing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer,StandardScaler
le=LabelEncoder()    
ss=StandardScaler()
ohe=OneHotEncoder(categorical_features=[0])
c=0
l=[]
dictl={}
l1=[]
dictl1={}

for i1 in range(1,80):    
    if (type(X[0][i1])==str):
        #print(i1)
        c=c+1
        le.fit(X[:,i1])
        X[:,i1]=le.transform(X[:,i1])
        l.append(i1)
    else:
        l1.append(i1)
c=0
for i in range(0,len(l)):
    dictl[l[i]]=dict1[l[i]]
for i in range(0,len(l1)):
    dictl1[l1[i]]=dict1[l1[i]]
    
for i1 in range(0,rows):
    for j1 in range(0,columns-1):
        if (type(X[i1][j1])==str):
            X[i1][j1]=0
c1=0

l_final=[] #column list for X_new
l_final.append('Id')
X_new=X[:,0:1]
for i1 in l1:        #columns originally with numeric values being added first
    l_final.append(dict1[i1])
    X_temp=X[:,i1:i1+1]
    if (i1==0):
        X_new=X_temp
    else:
        X_new=np.hstack((X_new,X_temp))
#print(X_new)                 
no=-1
for i1 in l:     #columns originally with categorical values being added after applying OneHotEncoder preprocessing
    no=no+1
    X_temp=X[:,i1:i1+1]
    X_temp=ohe.fit_transform(X_temp)    #one hot encoding
    X_temp=X_temp[:,1:] #removing first column
    col=X_temp.shape[1]
    for i in range(1,col+1):        
        s=dict1[i1]+str(i)
     #   print(s)
        l_final.append(s)
    X_temp.todense()    
    X_temp=X_temp.toarray()    
    X_new=np.hstack((X_new,X_temp))
    #print(i1)
#X_arr=np.array(X_new,np.dtype(l_final,float))
#print(X_new)
#splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_new,Y,test_size=0.2,random_state=0) 

X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

#PCA -> component analysis
#18,22,21,23,31(maybe),35(maybe)
#take user input code here

user_input=[]
print("Enter year: ")
y1=int(input())
user_input.append(y1)
print("Enter type of foundation: ")
y1=input()
if (y1=='CBlock'):
    y1=1
else:
    y1=0
user_input.append(y1)
print("Enter Basement Quality: ")
y1=input()
if (y1=='None'):
    y1=1
else:
    y1=0
user_input.append(y1)
print("Enter Garage Type: ")
y1=input()
if (y1=='None'):
    y1=1
else:
    y1=0
user_input.append(y1)
print("Enter Above ground living area square feet: ")
y1=int(input())
user_input.append(y1)
print("Enter House Style: ")
y1=input()
if (y1=='1Story'):
    y1=1
else:
    y1=0
user_input.append(y1)
print("Enter Land Countour: ")
y1=input()
if (y1=='Low'):
    y1=1
else:
    y1=0
user_input.append(y1)
print("Enter Exterior covering on house: ")
y1=input()
if (y1=='CmentBd'):
    y1=1
else:
    y1=0
user_input.append(y1)
print("Enter Pool Area: ")
y1=int(input())
user_input.append(y1)
print("Enter Garage Quality: ")
y1=input()
if (y1=='Po'):
    y1=1
else:
    y1=0
user_input.append(y1)
print("Enter Sale Condition: ")
y1=input()
if (y1=='Normal'):
    y1=1
else:
    y1=0
user_input.append(y1)
print("Enter Proximity to various conditions: ")
y1=input()
if (y1=='Norm'):
    y1=1
else:
    y1=0
user_input.append(y1)
user_input=np.asarray(user_input) 
x=len(user_input)

l_matrix=[] 

if x>0:
    from sklearn.decomposition import PCA
    pca=PCA(n_components=x,random_state=0)
    X_train=pca.fit_transform(X_train)
    X_test=pca.transform(X_test)
    explained_var_arr=pca.explained_variance_ratio_ #ranking of imp features
    vsum=0
    j=0
    
    for i in range(0,len(explained_var_arr)):
        vsum=vsum+explained_var_arr[i]
        j=i
        if (vsum>=0.5):
            break

#For user input, getting name of important columns
    df=pd.DataFrame(pca.components_)
    for i in range(0,261):
        for j in range(0,x):
            df[i][j]=abs(df[i][j])
    l_max_col_value=[]
    l_max_col_name=[]
    for i in range(0,x):
        list1=list(df.iloc[i,:].values)
        m=max(list1)
        p=list1.index(m)
        l_max_col_value.append(p)
        l_max_col_name.append(l_final[p])
    dict_col_val={}
    for i in range(0,x):
        if (i%4!=0):
            index=-1
            for j in range(0,1168):
                if (X_new[j][l_max_col_value[i]]==1.0):
                    index=j
                    s1=l_max_col_name[i][0:len(l_max_col_name[i])-1]
              #  print(s1)
                    dict_col_val[i]=dataset_1.iloc[index][s1]
                    break
#classifier part
    from sklearn.ensemble import GradientBoostingRegressor
    gbregressor=GradientBoostingRegressor(n_estimators=100,random_state=0)
    gbregressor.fit(X_train,Y_train)
    y_pred_gbr=gbregressor.predict(X_test)
    y_pred_train_gbr=gbregressor.predict(X_train)
    y_pred_train_gbr=y_pred_train_gbr.tolist()
    
    acc_gbr=[]
    for i in range(0,len(y_pred_gbr)):
        acc_gbr.append(abs(y_pred_gbr[i]-Y_test[i])/Y_test[i])
    final_s_gbr=sum(acc_gbr)/len(acc_gbr)
        
    acc_train_gbr=[]
    for i in range(0,len(y_pred_gbr)):
        acc_train_gbr.append(abs(y_pred_train_gbr[i]-Y_train[i])/Y_train[i])
    final_s_train_gbr=sum(acc_train_gbr)/len(acc_train_gbr)

    from sklearn.ensemble import RandomForestRegressor
    rfregressor=RandomForestRegressor(n_estimators=100,random_state=0)
    rfregressor.fit(X_train,Y_train)
    y_pred_rfr=rfregressor.predict(X_test)
    y_pred_train_rfr=rfregressor.predict(X_train)
    y_pred_train_rfr=y_pred_train_rfr.tolist()
    
    acc_rfr=[]
    for i in range(0,len(y_pred_rfr)):
        acc_rfr.append(abs(y_pred_rfr[i]-Y_test[i])/Y_test[i])
    final_s_rfr=sum(acc_rfr)/len(acc_rfr)
        
    acc_train_rfr=[]
    for i in range(0,len(y_pred_rfr)):
        acc_train_rfr.append(abs(y_pred_train_rfr[i]-Y_train[i])/Y_train[i])
    final_s_train_rfr=sum(acc_train_rfr)/len(acc_train_rfr)

    print("-----------------------------------------------")

    from sklearn.svm import SVR
    svmregressor=SVR(kernel='rbf')
    svmregressor.fit(X_train,Y_train)
    y_pred_svm=svmregressor.predict(X_test)
    y_pred_train_svm=svmregressor.predict(X_train)
    y_pred_train_svm=y_pred_train_svm.tolist()

    acc_svm=[]
    for i in range(0,len(y_pred_svm)):
        acc_svm.append(abs(y_pred_svm[i]-Y_test[i])/Y_test[i])
    final_s_svm=sum(acc_svm)/len(acc_svm)

    acc_train_svm=[]
    for i in range(0,len(y_pred_svm)):
        acc_train_svm.append(abs(y_pred_train_svm[i]-Y_train[i])/Y_train[i])
    final_s_train_svm=sum(acc_train_svm)/len(acc_train_svm)

    print("-----------------------------------------------")

    print("-----------------------------------------------")
    
    for i in range(0,len(y_pred_train_rfr)):
        l1=[]
        l1.append((-1)*abs(y_pred_train_rfr[i]-Y_train[i]))
        l1.append((-1)*abs(y_pred_train_svm[i]-Y_train[i]))
        l1.append((-1)*abs(y_pred_train_gbr[i]-Y_train[i]))        
      #  l1.append(abs(y_pred_train_pr[i]-Y_train[i]))    
        l_matrix.append(l1)

#reinforcement learning of classifiers using Upper Confidence Bound to find weights of each classifier
import math
N=1168
d=3
t=0
no_of_selections=[0]*d
score_arr=[0]*d
for i in range(0,N):
    max_bound=0
    ad=0
    for j in range(0,d):
        if no_of_selections[j]>0:
            a=score_arr[j]/no_of_selections[j]
            delta_j=math.sqrt(3/2*(math.log(i+1)/no_of_selections[j]))
            a=a+delta_j
        else:
            a=1e400
        if a>max_bound:
            max_bound=a
            ad=j
    no_of_selections[ad]=no_of_selections[ad]+1
    reward=l_matrix[i][ad]
    score_arr[ad]=score_arr[ad]+reward    

    #  ads_selected.append(ad)
print(no_of_selections)
w1=no_of_selections[0]/sum(no_of_selections)
w2=no_of_selections[1]/sum(no_of_selections)
w3=no_of_selections[2]/sum(no_of_selections)

#predicting user input    
user_input=user_input.reshape(1,-1)
prediction_user=w1*rfregressor.predict(user_input)+w2*svmregressor.predict(user_input)+w3*gbregressor.predict(user_input)
final_pred=w1*rfregressor.predict(X_test)+w2*svmregressor.predict(X_test)+w3*gbregressor.predict(X_test)
acc_test=[]
for i in range(0,len(final_pred)):
    acc_test.append(abs(final_pred[i]-Y_test[i])/Y_test[i])
final_s__test_pred=sum(acc_test)/len(acc_test)
print("Predicted price = ",prediction_user)

#w3=no_of_selections[2]/sum(no_of_selections)
#w4=no_of_selections[3]/sum(no_of_selections)
"""
X[:,79]=le.fit_transform(X[:,79])
X[:,78]=le.fit_transform(X[:,78])
X[:,11]=le.fit_transform(X[:,11])
X[:,2]=le.fit_transform(X[:,2])
ohe=OneHotEncoder(categorical_features=[2,11,78,79])
X=ohe.fit_transform(X)            
convert={"None":0,"Po":1,"Fair":2,"TA":3,"Gd":4,"Ex":5,"No":0,"Yes":1,"FuseA":4,"FuseF":3,"FuseP":2,"Mix":4.5,"SBrkr":5,"Paved":2,"Gravel":1,"Reg":4,"IR1":3,"IR2":2,"IR3":1,"Lvl":4,"Bnk":3,"HLS":2,"Low":1,"AllPub":4,"NoSewr":3,"NoSeWa":2,"ELO":1,"Gtl":3,"Mod":2,"Sev":1,"P":1.5,"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1,"Unf":1,"RFn":2,"Fin":3}
"""
#previous data preprocessing
"""
for i1 in range(0,rows):
    for j1 in range(0,columns-1):
        if (type(X[i1][j1])==str):
            c=c+1
            print(X[i1][j1])
            print("Column = ",j1)
        elif (type(X[i1][j1])==int):
            X[i1][j1]=float(X[i1][j1])
#print(c)
for i1 in range(0,rows):
    for j1 in range(0,columns-1):
        if (type(X[i1][j1])==float):
            c1=c1+1
#
"""            
#classifier section
"""
from xgboost import XGBRegressor
xgbregressor=XGBRegressor(n_estimators=100,learning_rate=0.08,gamma=0,subsample=0.75,colsample_bytree=1,max_depth=7)
xgbregressor.fit(X_train,Y_train)
y_pred_xgb=xgbregressor.predict(X_test)
y_pred_train_xgb=xgbregressor.predict(X_train)
y_pred_train_xgb=y_pred_train_xgb.tolist()

acc_xgb=[]
for i in range(0,len(y_pred_xgb)):
    acc_xgb.append(abs(y_pred_xgb[i]-Y_test[i])/Y_test[i])
final_s_xgb=sum(acc_xgb)/len(acc_xgb)

print("-----------------------------------------------")
#
"""