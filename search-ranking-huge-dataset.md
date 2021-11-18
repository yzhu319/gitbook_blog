# Search-ranking: huge dataset

## Intro

When the data is huge and stored in multiple clusters, we can use Spark computing framework. For machine learning tasks, we will then use frameworks designed for Spark, like MMLSpark (Microsoft Machine Learning for Apache Spark). MMLSpark integrates Spark ML pipelines with LightGBM, enabling highly-scalable solutions to ML training jobs. (MMLSpark requires Scala 2.11, Spark 2.4+, and Python 3.5+. It has API in[ Scala](https://mmlspark.blob.core.windows.net/docs/1.0.0-rc3/scala/index.html#package) and[ PySpark](https://mmlspark.blob.core.windows.net/docs/1.0.0-rc3/pyspark/index.html))

The implementation is briefly outlined below.



### 1) Access data directly from Hadoop system&#x20;

```python
#!/usr/bin/env python
# coding: utf-8
import findspark
findspark.init()

from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pyspark

spark = SparkSession.builder.appName("Yuanzheng_app") \
        .master("local") \
        .appName("Colab") \
        .config('spark.ui.port', '4050') \
        .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()

spark.conf.set("spark.sql.repl.eagerEval.enabled",True)

get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import numpy as np
import pandas as pd
import mmlspark
namenode = "hdfs://shldvfsdh014.tvlport.net:8020"
gds_str = "1P"
pcc_str = "6TC"
date_str = "2021041*"
df_all = spark.read.format("csv").option("header", True).option("inferSchema", True).load(namenode+"/user/yuanzheng.zhu/"+"raw_data_oneway_"+gds_str+"_"+pcc_str+"_"+date_str+".csv")

```

### 2) Split data into training + testing

```python
from pyspark.sql import window as W
import pyspark.sql.functions as f

df_all = df_all.withColumn('qid', f.dense_rank().over(W.Window.orderBy('uniq_key'))).drop("uniq_key")
# re-order, make qid the first column
df_all = df_all.select(['qid'] + df_all.columns[0:-1])

num_qid = df_all.select("qid").distinct().count()
df_train = df_all.filter(df_all.qid <= 0.8*num_qid)
df_test = df_all.filter(df_all.qid > 0.8*num_qid)
```

### 3) Map categorical variables to -- string to index

```python
# https://stackoverflow.com/questions/36942233/apply-stringindexer-to-several-columns-in-a-pyspark-dataframe
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

cat_features = ("outPaxTypes","outOperatingCxr","outCabinClass", \
"s_outBookingClass","outDept_month","outDept_day","outArr_month","outArr_day")

cat_features_index = [column+"_index" for column in list(cat_features)]

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").setHandleInvalid("keep") \
for column in list(cat_features) ]

pipeline = Pipeline(stages=indexers)

df_test_index = pipeline.fit(df_test).transform(df_test).drop(*cat_features)
df_train_index = pipeline.fit(df_train).transform(df_train).drop(*cat_features)


from pyspark.ml.feature import VectorAssembler
feature_cols = df_train_index.columns[2:]
featurizer = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features'
)


train_data = featurizer.transform(df_train_index)['booked','qid', 'features']
test_data = featurizer.transform(df_test_index)['booked','qid', 'features']

```

### 4) Fit a gradient boosting model (LightGBMRanker)

Key parameters to tune:&#x20;

1. leave size
2. boosting rounds

```python
from mmlspark.lightgbm import LightGBMRanker
features_col = 'features'
query_col = 'qid'
label_col = 'booked'

lgbm_ranker = LightGBMRanker(labelCol=label_col,
                             featuresCol=features_col,
                             categoricalSlotNames=cat_features_index,
                             groupCol=query_col,
                             predictionCol='preds',
                             leafPredictionCol='leafPreds',
                             featuresShapCol='importances',
                             repartitionByGroupingColumn=True,
                             numLeaves=32,
                             numIterations=200,
                             evalAt=[1,3,5],
                             verbosity=1,
                             metric='ndcg')

lgbm_ranker_model = lgbm_ranker.fit(train_data)

```

### 5) Grid-search for hyper-parameter tuning

#### Search range

num\_leaves\_list = \[2, 5, 10, 25, 50, 125]

num\_boost\_round\_list = \[100, 200, 500, 1000, 2000, 5000]

#### Results

Optimized parameter to maximize NDCG score:

num\_leaves = 50, num\_boost\_round = 2000

![Grid-search for best hyper-parameters](<.gitbook/assets/image (2).png>)

&#x20;

### 6) Make predictions and save model

```python
predictions = lgbm_ranker_model.transform(test_data)

model_save_name = "lgbm_ranker_enhanced"+gds_str+"_"+pcc_str
lgbm_ranker_model.saveNativeModel("sample_data/" + model_save_name + ".mod")
```

