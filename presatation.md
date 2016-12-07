xgboost Demo with data set
========================================================
author: yingkun.li  
date: 2016-12-07  
autosize: true  


Content
========================================================
- xgboost introduction
- Installation
- Dataset presentation
- Dataset loading
- Basic Training using XGBoost
- Basic prediction using XGBoost
- Advanced features


xgboost Introduction
========================================================
Xgboost is short for eXtreme Gradient Boosting package.  
  
It supports various objective functions, including regression, classification and   ranking. The package is made to be extendible, so that users are also allowed to   define their own objective functions easily.  



Features
========================================================
It has several features:    
  
- Speed: it can automatically do parallel computation on Windows and Linux, with OpenMP. It is generally over 10 times faster than the classical gbm.
- Input Type: it takes several types of input data:  
   + Dense Matrix: R‘s dense matrix, i.e. matrix ;  
   + Sparse Matrix: R‘s sparse matrix, i.e. Matrix::dgCMatrix ;  
   + Data File: local data files ;  
   + xgb.DMatrix: its own class (recommended).  
- Sparsity: it accepts sparse input for both tree booster and linear booster, and  
is optimized for sparse input ;
- Customization: it supports customized objective functions and evaluation functions.


Installation
===============================================
Github version
```{}
##个人实际操作并没有成功，而是安装了CRAN版本
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
```

CRAN version
```{}
install.packages("xgboost")
```

Dataset presentation
============================================
虽然xgboost支持多种数据输入类型，但其推进使用xgb.DMatrix的数据类型。  
如何将一个数据集从data.frame转为xgb.DMatrix类型后续将会介绍。 
本文使用xgboost内置数据集agaricus演示基本用法。  

```r
require(xgboost)
#加载数据
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
#查看train数据集的数据结构
str(train)
```

```
List of 2
 $ data :Formal class 'dgCMatrix' [package "Matrix"] with 6 slots
  .. ..@ i       : int [1:143286] 2 6 8 11 18 20 21 24 28 32 ...
  .. ..@ p       : int [1:127] 0 369 372 3306 5845 6489 6513 8380 8384 10991 ...
  .. ..@ Dim     : int [1:2] 6513 126
  .. ..@ Dimnames:List of 2
  .. .. ..$ : NULL
  .. .. ..$ : chr [1:126] "cap-shape=bell" "cap-shape=conical" "cap-shape=convex" "cap-shape=flat" ...
  .. ..@ x       : num [1:143286] 1 1 1 1 1 1 1 1 1 1 ...
  .. ..@ factors : list()
 $ label: num [1:6513] 1 0 0 1 0 0 0 1 0 0 ...
```

```r
#查看数据的维度信息
dim(train$data);dim(test$data)
```

```
[1] 6513  126
```

```
[1] 1611  126
```


Basic Training using XGBoost
=====================================================
使用xgboost函数对训练集数据train进行建模。  
本案例建模中主要使用的参数：
   - objective = "binary:logistic": 表示进行二分类建模（预测0,1）;
   - max.deph = 2: 最大树深;
   - nthread = 2: 计算时需要使用的cpu核数;
   - nround = 2: 模型的迭代次数.
  
  
xgboost中用于调节模型的参数很多，详细解释可见官方文档：  
<https://github.com/dmlc/xgboost/blob/master/doc/parameter.md>

Code demo
==============================================

```r
#输入数据为sparse matrix格式：稀疏矩阵中存在大量的0，而这一格式中0不占存储。
bstSparse <- xgboost(data = train$data, label = train$label, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
```

```
[0]	train-error:0.046522
[1]	train-error:0.022263
```


```r
#输入数据为Dense matrix格式
bstDense <- xgboost(data = as.matrix(train$data), label = train$label, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
```

```
[0]	train-error:0.046522
[1]	train-error:0.022263
```


```r
#输入数据为xgb.DMatrix格式
dtrain <- xgb.DMatrix(data = train$data, label = train$label)
bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
```

```
[0]	train-error:0.046522
[1]	train-error:0.022263
```

Code demo for Verbose option
=================================================


```r
# verbose = 0, 不会打印模型信息
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 0)
```


```r
# verbose = 1, 打印评价矩阵的信息
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 1)
```

```
[0]	train-error:0.046522
[1]	train-error:0.022263
```


```r
# verbose = 2, 打印所有的模型结果信息（树结构、评价矩阵）
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 2)
```

```
tree prunning end, 1 roots, 6 extra nodes, 0 pruned nodes ,max_depth=2
[0]	train-error:0.046522
tree prunning end, 1 roots, 4 extra nodes, 0 pruned nodes ,max_depth=2
[1]	train-error:0.022263
```

Basic prediction using XGBoost
==================================================

```r
#对测试集数据进行预测
pred <- predict(bst, test$data)
print(head(pred,5))
```

```
[1] 0.28583017 0.92392391 0.28583017 0.28583017 0.05169873
```


```r
#将预测的概率转变为0,1的二分类形式
prediction <- as.numeric(pred > 0.5)
print(head(prediction))
```

```
[1] 0 1 0 0 0 1
```


```r
#计算预测误差
err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error=", err))
```

```
[1] "test-error= 0.0217256362507759"
```

Advanced features
========================================================

```r
#load data
dtrain <- xgb.DMatrix(data = train$data, label=train$label)
dtest <- xgb.DMatrix(data = test$data, label=test$label)
```

- Measure learning progress with xgb.train
   - we use watchlist parameter. It is a list of xgb.DMatrix, each of them tagged with a name.
   - 使用watchlist参数，能看到每一次迭代后，训练集和测试集的模型得分。
   

```r
#使用watchlist参数，观察到测试集的模型得分
watchlist <- list(train=dtrain, test=dtest)

bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nthread = 2, nround=2, watchlist=watchlist, objective = "binary:logistic")
```

```
[0]	train-error:0.046522	test-error:0.042831
[1]	train-error:0.022263	test-error:0.021726
```


```r
#使用eval.metric参数，实现多种模型得分的计算
bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nthread = 2, nround=2, watchlist=watchlist, eval.metric = "error", eval.metric = "logloss", objective = "binary:logistic")
```

```
[0]	train-error:0.046522	train-logloss:0.233366	test-error:0.042831	test-logloss:0.226687
[1]	train-error:0.022263	train-logloss:0.136656	test-error:0.021726	test-logloss:0.137875
```

Summary
=======================================
- xgboost introduction: speed、accuracy、comprehensive
- Installation：install.packages("xgboost")
- Dataset presentation：sparse matri、dense matrix、xxgb.DMatrix
- Dataset loading: xgb.DMatrix
- Basic Training using XGBoost：xgboost
- Basic prediction using XGBoost: predict
- Advanced features: xgb.train


