import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os,sys,pdb
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm
class SHOW_DATA:
    def __init__(self, path):
        self._df = pd.read_csv(path)
        print self._df.columns
    def missdata(self):
        total = self._df.isnull().sum().sort_values(ascending=False)
        percent = (self._df.isnull().sum() / self._df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total,percent],axis=1,keys=['total','percent'])
        print missing_data.head(20)
        #for simplify we just drop all nan data
        idx = (missing_data[missing_data['total']>1]).index
        self._df = self._df.drop(idx,1)
        return
    def univariance(self, var):
        print self._df[var].describe()

        scaled = StandardScaler().fit_transform(self._df[var][:,np.newaxis])
        low_range = scaled[scaled[:,0].argsort()][:10]
        high_range = scaled[scaled[:,0].argsort()][-10:]
        print 'outer range(low) of the dist'
        print low_range
        print 'outer range(high) of the dist'
        print high_range

        #show norm 
        fig = plt.figure()
        plt.title('dist before log-adjust(skewness=%.3f kurtosis=%.3f)'%(self._df[var].skew(), self._df[var].kurt())  )
        sns.distplot(self._df[var], fit=norm)
        fig = plt.figure()
        stats.probplot(self._df[var], plot=plt)
        #apply log to make it to be norm
        #in case of positive skewnewss, log transform will make it to be norm
        self._df[var+'LOG'] = np.log(self._df[var])
        fig = plt.figure()
        plt.title('dist after log-adjust(skewness=%.3f kurtosis=%.3f)'%(self._df[var+"LOG"].skew(), self._df[var+"LOG"].kurt())  )
        sns.distplot(self._df[var+'LOG'], fit=norm)
        fig = plt.figure()
        stats.probplot(self._df[var+'LOG'], plot=plt)
        plt.show()
        return
    def bivariance(self,varY,varX):
        data = pd.concat([self._df[varX], self._df[varY]],axis=1)
        data.plot.scatter(x=varX,y=varY,ylim=(0,800000),title='numeric data')
        plt.show()
    def categorical(self,varY,varX):
        data = pd.concat([self._df[varX], self._df[varY]],axis=1)
        f,ax = plt.subplots(figsize=(8,6))
        plt.title('categorical data')
        fig = sns.boxplot(x=varX,y=varY,data=data)
        fig.axis(ymin=0,ymax=800000)
        plt.show()
    def correlation(self,var = ""):
        corrmat = self._df.corr()
        f,ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corrmat,vmax=.8, square=True)
        plt.show()
        if var ==  "":
            return
        k = 5
        cols = corrmat.nlargest(k,var)[var].index
        cm = np.corrcoef(self._df[cols].values.T)
        sns.set(font_scale = 1.25)
        hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},
                yticklabels = cols.values, xticklabels=cols.values)
        plt.show()
    def multivariance(self,varlist):
        sns.set()
        cols = varlist
        sns.pairplot(self._df[cols], size=2.5)
        plt.show()
    def one_hot_code_for_categorical(self):
        return pd.get_dummies(self._df)
    def show(self):
        pd = self.one_hot_code_for_categorical()
        print pd.head()
        self.multivariance(['SalePrice','OverallQual','GrLivArea','GarageCars'])
        self.correlation('SalePrice')
        self.categorical('SalePrice','OverallQual')
        self.bivariance('SalePrice','GrLivArea')
        self.missdata()
        self.univariance("SalePrice")
   


if __name__=="__main__":
    sd = SHOW_DATA('train.csv')
    sd.show()
