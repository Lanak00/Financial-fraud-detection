# -*- coding: utf-8 -*-
"""
Created on Fri May 18 22:22:44 2021

@author: Lana
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

creditcards_df = pd.read_csv("CC GENERAL.csv")

creditcards_df.info()
print(creditcards_df.shape)
creditcards_df.columns

#-------------------popunjavanje nedostajucih vrednosti-----------------------
print(creditcards_df.isna().sum())

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(creditcards_df.MINIMUM_PAYMENTS.dropna(), color='navy')
plt.subplot(1,2,2)
sns.distplot(creditcards_df.CREDIT_LIMIT.dropna(), color='navy')
plt.show()

creditcards_df.drop(['CUST_ID'], axis=1, inplace=True)
creditcards_df['CREDIT_LIMIT'].fillna(creditcards_df['CREDIT_LIMIT'].median(),inplace=True)
creditcards_df['MINIMUM_PAYMENTS'].fillna(creditcards_df['MINIMUM_PAYMENTS'].median(),inplace=True)

print(creditcards_df.isna().sum())

# --------------izvlacenje novih korisnih podataka iz dataseta----------------

# grupe na osnovu vrste placanja
creditcards_df.loc[:,['ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']]

print("\nnone \n", creditcards_df[(creditcards_df['ONEOFF_PURCHASES']==0) & (creditcards_df['INSTALLMENTS_PURCHASES']==0)].shape)
print("both \n", creditcards_df[(creditcards_df['ONEOFF_PURCHASES']>0) & (creditcards_df['INSTALLMENTS_PURCHASES']>0)].shape)
print("one-off \n", creditcards_df[(creditcards_df['ONEOFF_PURCHASES']>0) & (creditcards_df['INSTALLMENTS_PURCHASES']==0)].shape)
print("installment \n", creditcards_df[(creditcards_df['ONEOFF_PURCHASES']==0) & (creditcards_df['INSTALLMENTS_PURCHASES']>0)].shape, "\n")

def purchase(creditcards_df):
    if (creditcards_df['ONEOFF_PURCHASES']==0) & (creditcards_df['INSTALLMENTS_PURCHASES']==0):
        return 'none'
    if (creditcards_df['ONEOFF_PURCHASES']>0) & (creditcards_df['INSTALLMENTS_PURCHASES']>0):
         return 'both_oneoff_installment'
    if (creditcards_df['ONEOFF_PURCHASES']>0) & (creditcards_df['INSTALLMENTS_PURCHASES']==0):
        return 'one_off'
    if (creditcards_df['ONEOFF_PURCHASES']==0) & (creditcards_df['INSTALLMENTS_PURCHASES']>0):
        return 'istallment'
    
creditcards_df['purchase_type'] = creditcards_df.apply(purchase,axis=1)
creditcards_df.columns
    
# na mesecnom nivou
creditcards_df['Monthly_avg_purchase'] = creditcards_df['PURCHASES']/creditcards_df['TENURE']
creditcards_df['Monthly_cash_advance'] = creditcards_df['CASH_ADVANCE']/creditcards_df['TENURE']

# koeficijenti 
creditcards_df['limit_usage'] = creditcards_df.apply(lambda x: x['BALANCE']/x['CREDIT_LIMIT'], axis=1)
creditcards_df['payment_minpay'] = creditcards_df.apply(lambda x:x['PAYMENTS']/x['MINIMUM_PAYMENTS'],axis=1)


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.barplot(x='purchase_type',y='Monthly_avg_purchase',data=creditcards_df, palette = 'viridis')
plt.subplot(1,2,2)
sns.barplot(x='purchase_type',y='Monthly_cash_advance',data=creditcards_df, palette= 'viridis')
plt.show()

#------------------------ hendlovanje outlinera ------------------------------

print(creditcards_df.shape) 

plt.figure(figsize=(25,30))
for i, col in enumerate(creditcards_df.columns):
    if creditcards_df[col].dtype != 'object':
        ax = plt.subplot(11, 2, i+1)
        sns.kdeplot(creditcards_df[col], ax=ax, color='navy')
        plt.xlabel(col)
plt.show()

creditcards_df.columns

log_df=creditcards_df.drop(['purchase_type'],axis=1).applymap(lambda x: np.log(x+1)) 

col=['BALANCE','PURCHASES','CASH_ADVANCE','TENURE',
     'PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','CREDIT_LIMIT']

temp_df = log_df[[x for x in log_df.columns if x not in col ]]

temp_df.shape

plt.figure(figsize=(25,30))
for i, col in enumerate(temp_df.columns):
    if temp_df[col].dtype != 'object':
        ax = plt.subplot(7, 2, i+1)
        sns.kdeplot(temp_df[col], ax=ax, color='navy')
        plt.xlabel(col)
plt.show()

temp_df.columns
log_df.columns

#------------------- priprema dataseta za klasterizaciju ----------------------

original_df = pd.concat([creditcards_df, pd.get_dummies(creditcards_df['purchase_type'])],axis=1)

temp_df['purchase_type'] = creditcards_df.loc[:,'purchase_type']
pd.get_dummies(temp_df['purchase_type'])
dummy_df = pd.concat([temp_df,pd.get_dummies(temp_df['purchase_type'])],axis=1)
l = ['purchase_type']
dummy_df = dummy_df.drop(l,axis=1)

dummy_df.columns
dummy_df.shape
plt.figure(figsize=(12,12))
sns.heatmap(dummy_df.corr(), annot=True, cmap = 'viridis')
plt.show()

# --------------------------------- PCA --------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sc=StandardScaler()
dummy_df.shape
scaled_df = sc.fit_transform(dummy_df)

# pronaci broj komponenti 
var_ratio={}
for n in range(2,18):
    pc=PCA(n_components=n)
    pca_df =pc.fit(scaled_df)
    var_ratio[n]=sum(pca_df.explained_variance_ratio_)

print(var_ratio)

pd.Series(var_ratio).plot(color = "navy")

scaled_df.shape

finalpc_df=PCA(n_components=6).fit(scaled_df)
reduced_df = finalpc_df.fit_transform(scaled_df)
dd=pd.DataFrame(reduced_df)
dd.head()
# ------------------------------- Klasterizacija ------------------------------

# odrediti k - metod lakta
cluster_range = range( 1, 21 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( reduced_df )
    cluster_errors.append( clusters.inertia_ )
    
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:21]

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o", color = "navy" )

km_4=KMeans(n_clusters=4,random_state=123)
km_4.fit(reduced_df)

plot_df = pd.DataFrame(reduced_df,columns=['PC_' +str(i) for i in range(6)])
plot_df['Cluster']=km_4.labels_
sns.pairplot(plot_df, hue='Cluster', palette= 'viridis', diag_kind='kde',size=1.85)

# ------------------- interpretacija pocetnog dataseta ------------------------

columns=['BALANCE_FREQUENCY', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'Monthly_avg_purchase',
       'Monthly_cash_advance', 'limit_usage', 'payment_minpay', 'both_oneoff_installment',
       'istallment', 'none', 'one_off']

cluster_df_4=pd.concat([original_df[columns],pd.Series(km_4.labels_,name='Cluster_4')],axis=1)

cluster_4=cluster_df_4.groupby('Cluster_4')\
.apply(lambda x: x[columns].mean()).T
cluster_4

fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(cluster_4.columns))

cash_advance=np.log(cluster_4.loc['Monthly_cash_advance',:].values)
credit_score=(cluster_4.loc['limit_usage',:].values)
purchase= np.log(cluster_4.loc['Monthly_avg_purchase',:].values)
payment=cluster_4.loc['payment_minpay',:].values
installment=cluster_4.loc['istallment',:].values
one_off=cluster_4.loc['one_off',:].values


bar_width=.10
b1=plt.bar(index,cash_advance,color='navy',label='Monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='yellow',label='Credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='lightseagreen',label='Avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='forestgreen',label='Payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='mediumblue',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='greenyellow',label='One_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()

# Percentage of each cluster in the total customer base
s = cluster_df_4.groupby('Cluster_4').apply(lambda x: x['Cluster_4'].value_counts())
print (s),'\n'

percentage = pd.Series((s.values.astype('float')/ cluster_df_4.shape[0])*100,name='Percentage')
print ("Cluster -4 "),'\n'
print (pd.concat([pd.Series(s.values,name='Size'),percentage],axis=1))

