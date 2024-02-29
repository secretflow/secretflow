from sklearn.preprocessing import LabelEncoder,QuantileTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 

 
dfdata = pd.read_csv("/root/dataset/criteo/criteo_train_small.txt",sep="\t",header=None)
dfdata.columns = ["label"] + ["I"+str(x) for x in range(1,14)] + [
    "C"+str(x) for x in range(14,40)]
 
cat_cols = [x for x in dfdata.columns if x.startswith('C')]
num_cols = [x for x in dfdata.columns if x.startswith('I')]
num_pipe = Pipeline(steps = [('impute',SimpleImputer()),('quantile',QuantileTransformer())])
 
for col in cat_cols:
    dfdata[col]  = LabelEncoder().fit_transform(dfdata[col])
 
dfdata[num_cols] = num_pipe.fit_transform(dfdata[num_cols])

categories = [dfdata[col].max()+1 for col in cat_cols]

# dftest = pd.read_csv("/root/dataset/criteo/criteo_test_small.txt",sep="\t",header=None)

# dftest.columns = ["label"] + ["I"+str(x) for x in range(1,14)] + [
#     "C"+str(x) for x in range(14,40)]

# cat_cols = [x for x in dftest.columns if x.startswith('C')]
# num_cols = [x for x in dftest.columns if x.startswith('I')]
# num_pipe = Pipeline(steps = [('impute',SimpleImputer()),('quantile',QuantileTransformer())])
 
# for col in cat_cols:
#     dftest[col]  = LabelEncoder().fit_transform(dftest[col])
 
# dftest[num_cols] = num_pipe.fit_transform(dftest[num_cols])


dftrain_val,dftest = train_test_split(dfdata,test_size=0.2)
dftrain,dfval = train_test_split(dftrain_val,test_size=0.2)


dftrain.to_csv("/root/develop/secretflowdev/OSCP/DCN/data/criteo_train_small.csv",index=False)
dfval.to_csv("/root/develop/secretflowdev/OSCP/DCN/data/criteo_val_small.csv",index=False)
dftest.to_csv("/root/develop/secretflowdev/OSCP/DCN/data/criteo_test_small.csv",index=False)


