xgboost-jd_demo
========================================================
author: yingkun.li  
date: 2016-12-08  
autosize: true

Content
========================================================

- data preparation
- grid search
- parameter select
- parameter select code
- best parameter select

data preparation
========================================================


```r
#1.加载数据集
load('df_train_importance_oot_by_prod.RData') #加载后数据集名：df_train_importance
load('df_test_importance_oot_by_prod.RData')  #加载后数据集名：df_test_importance
```


```r
#2.查看数据基本信息
dim(df_train_importance)
```

```
[1] 43109   735
```

```r
dim(df_test_importance)
```

```
[1] 7272  735
```


```r
#1.将训练集变为xgb.DMatrix格式
library(dplyr)
library(xgboost)
data_to_xgbDM=function(df_train,target='y'){
  train_x<- df_train%>%select(-one_of(target))
  train_y <- df_train%>%select(one_of(target))
  train.mat <- as.matrix(train_x)
  dtrain <- xgb.DMatrix(train.mat,label=as.matrix(train_y),missing=NA)
  rm(df_train,train_x,train_y,train.mat);gc();
  return(dtrain)
}
```


```r
df_train_importance=data_to_xgbDM(df_train_importance)
df_test_importance=data_to_xgbDM(df_test_importance)
```


grid search
========================================================


```r
#编写一个函数，实现人工网格调参
#参数列表：
##dtrain:用于训练的数据 包含X Y, xgb.DMatrix格式
##param：参数数据框, data.frame格式
##message：跑模型时每一次的备注与描述 character类型
##cv_nfold：交叉验证的折数 默认为5
##nthreads：计算时使用的cpu核数 默认为4
##cv_nround：迭代的次数
##N_cv_stop：提前迭代结束的次数
##verbose_:是否打印每一次迭代的结果

model_grid_search=function(dtrain,param,message,cv_nfold=5,nthreads=4,cv_nround=30,N_cv_stop=15,verbose_=T){
  ####创建一个数据框 用于保存每一组参数模型结果
  each_paramlst_best <- data.frame()
  
  #创建ls_param用于存储每一次备调的参数
  str_1=paste0(rep(NA,length(param)),collapse = ',')
  text=paste0('ls_param=list(',str_1,')')
  eval(parse(text=text))
  names(ls_param)=names(param)
  num_param=dim(param)[1]
  
  #使用交叉验证xgb.cv训练模型
  for(paramlst_id in 1:num_param){
    t1=Sys.time()
    for(i in 1:length(param[paramlst_id,])){ls_param[i]=param[paramlst_id,][i]}
    #随机种子，0.4版本没有seed参数，0.6版本有
    set.seed(0)
    mdcv <- xgb.cv(data=dtrain, params = ls_param, nthread=nthreads, nfold=cv_nfold, 
                   nrounds=cv_nround,verbose = verbose_, early.stop.round=N_cv_stop)
    
    #记录交叉验证时间
    t2=Sys.time()
    #取出这一组参数能达到的最大score，及达到时的迭代轮路
    max_auc = max(mdcv[, test.auc.mean])
    max_auc_index = which.max(mdcv[, test.auc.mean])
    
    #记录参数
    each_paramlst_best_temp=data.frame()
    each_paramlst_best_temp[1,'paramlst_id'] <- paramlst_id
    each_paramlst_best_temp[1,'max_auc'] <- max_auc
    each_paramlst_best_temp[1,'max_auc_iter'] <- max_auc_index
    temp=cbind(each_paramlst_best_temp,data.frame(ls_param))
    each_paramlst_best=rbind(each_paramlst_best,temp)
    
    #保存结果
    save(each_paramlst_best,file = paste0('each_paramlst_best_',message,'.RData'))
    rm(mdcv);gc()
    #打印消息至控制台
    cat('\n')
    cat(paste0('running to :',paramlst_id,'th param \n'))
    cat(paste0('max_auc:',max_auc,'\n'))
    print(t2-t1)
    cat('\n')
  }
  
  cat(paste0('modle-out-topath:',getwd()))
}
```

parameter select
===================================
- 调参经验顺序
  - 1.控制每一次迭代的树相关参数
      - max_depth：最大树深 [default:6]
      - min_child_weight：叶节点最小样本数权重 [default:1]
      - gamma ：每轮迭代后分裂数所得带来的信息增益 [default:0]
  - 2.控制抽样的参数（主要作用：增加模型的鲁棒性和防止过拟合）
      - subsample：抽取的样本比例，数据框的行 [default:0.8]
      - colsample_bytree：抽取的变量比例，数据框的列 [default:0.8]
  - 3.正则项（防止过拟合）
      - lambda：L2正则 [default:0]
  - 4.正负样本不均衡时的可调参数
      - scale_pos_weight=negative/positive  'gernal positive is 1/yes' [range 0-INF]
  - 5.学习率
      - eta  一般最后调，注意：缩小eta，需相应的调大nround [range 0-1] [default:1]



parameter select code
=====================================================


```r
#备选参数列表构成的数据框
param_step_imp_1 <- expand.grid(objective = "binary:logistic",eval_metric = "auc",
                           max_depth=c(4,6),min_child_weight = c(2,4),gamma = 1,
                           subsample=1,colsample_bytree=0.95,
                           lambda=60,
                           eta=0.135,
                           stringsAsFactors = F)
head(param_step_imp_1,5)
```

```
        objective eval_metric max_depth min_child_weight gamma subsample
1 binary:logistic         auc         4                2     1         1
2 binary:logistic         auc         6                2     1         1
3 binary:logistic         auc         4                4     1         1
4 binary:logistic         auc         6                4     1         1
  colsample_bytree lambda   eta
1             0.95     60 0.135
2             0.95     60 0.135
3             0.95     60 0.135
4             0.95     60 0.135
```


```r
#模型调参
model_grid_search(df_train_importance,param_step_imp_1,'param_out',nthreads=16,cv_nround=200,N_cv_stop=150,verbose_ = F)
```

```

running to :1th param 
max_auc:0.705238
Time difference of 1.217186 mins


running to :2th param 
max_auc:0.702919
Time difference of 1.789686 mins


running to :3th param 
max_auc:0.705245
Time difference of 1.264903 mins


running to :4th param 
max_auc:0.703364
Time difference of 1.814378 mins

modle-out-topath:/home/datashare/crawl_ecomm_jd/lyk_建模
```




best parameter select
===============================================================

```r
#模型调参结果
#modle-out-topath:/home/datashare/crawl_ecomm_jd/lyk_建模
load("/home/datashare/crawl_ecomm_jd/lyk_建模/each_paramlst_best_param_out.RData")
head(each_paramlst_best,5)
```

```
  paramlst_id  max_auc max_auc_iter       objective eval_metric max_depth
1           1 0.705238          154 binary:logistic         auc         4
2           2 0.702919           84 binary:logistic         auc         6
3           3 0.705245          166 binary:logistic         auc         4
4           4 0.703364           92 binary:logistic         auc         6
  min_child_weight gamma subsample colsample_bytree lambda   eta
1                2     1         1             0.95     60 0.135
2                2     1         1             0.95     60 0.135
3                4     1         1             0.95     60 0.135
4                4     1         1             0.95     60 0.135
```

model train
==========================================

```r
#编写一个函数，实现模型训练和独立测试集的auc ks指标计算
#参数列表
##dtrain：训练集 xgb.DMtrix类型
##dtest：独立测试集 xgb.DMtrix类型
##param_ls：参数列表 list类型
##message：模型备注 用于保存importance变量时文件名的备注信息
##importance：是否输出importance变量 
##nthreads：计算使用的cpu
##nrounds:模型迭代次数
##verbose_：是否打印信息（本函数的不足，使用了watchlist参数，与此参数冲突）

model_train=function(dtrain,dtest,param_ls,message,importance=T,nthreads=12,nrounds=200,verbose_=1){
  t1=Sys.time()
  
  #得到训练模型
  set.seed(0)
  watchlists <- list(train=dtrain, test=dtest)
  bst=xgb.train(data=dtrain,params=param_ls,nthread = nthreads, nround = nrounds,
                watchlist=watchlists, verbose = verbose_)
  
  #得到预测结果
  pred_train=predict(bst,dtrain)
  pred_test=predict(bst,dtest)
  
  #计算auc,ks
  # library(pROC)
  # independence_test_auc=auc(test_y$y,pred)
  # print(paste0('independence_test_auc:',independence_test_auc))
  ks_train=KS(pred_train,getinfo(dtrain,'label'))
  auc_train=AUC(pred_train,getinfo(dtrain,'label'))
  
  ks_test=KS(pred_test,getinfo(dtest,'label'))
  auc_test=AUC(pred_test,getinfo(dtest,'label'))
  
  #result
  result=list(ks_train=ks_train,auc_train=auc_train,ks_test=ks_test,auc_test=auc_test)
  cat('\n')
  print(Sys.time()-t1)
  if(importance){
    importance_var=xgb.importance(feature_names=colnames(train_x),model=bst)
    file_name=paste0('importance_var_',message,'.RData')
    save(importance_var,file=file_name)
  }
  
  return(result)
}
```



```r
#计算ks统计指标：tpr-fpr的最大值 ROC曲线中的点的坐标(x,y),其中y-x的最大值
KS<-function(pred,y){
  library(ROCR)
  p<-prediction(as.numeric(pred),y)
  perf<-performance(p,'tpr','fpr')
  ks<-max(attr(perf,'y.values')[[1]]-attr(perf,'x.values')[[1]])
  return(ks)
}

#auc指标：精确度指标
AUC<-function(pred,y){
  library(ROCR)
  p<-prediction(as.numeric(pred),y)
  auc<-performance(p,'auc')
  auc<-unlist(slot(auc,'y.values'))
  return(auc)
}
```



```r
#选出auc最佳的参数，带入模型中建立模型
best_param=function(each_paramlst_best,best_index=1){
  each_paramlst_best$objective=as.character(each_paramlst_best$objective)
  each_paramlst_best$eval_metric=as.character(each_paramlst_best$eval_metric)
  param_ls=list()
  for(i in 1:length(each_paramlst_best)){
  param_ls[[i]]=each_paramlst_best[best_index,i+3]
   }
  names(param_ls)=names(each_paramlst_best)[4:length(each_paramlst_best)]
  return(param_ls)
}
best_param=best_param(each_paramlst_best,3)
best_param
```

```
$objective
[1] "binary:logistic"

$eval_metric
[1] "auc"

$max_depth
[1] 4

$min_child_weight
[1] 4

$gamma
[1] 1

$subsample
[1] 1

$colsample_bytree
[1] 0.95

$lambda
[1] 60

$eta
[1] 0.135
```


```r
#训练模型并计算指标
model_train(df_train_importance,df_test_importance,best_param,'param_bst_imp',importance=F,nthreads=10,nrounds=128,verbose_ = 1)
```

```
[0]	train-auc:0.621403	test-auc:0.594476
[1]	train-auc:0.646416	test-auc:0.609414
[2]	train-auc:0.663093	test-auc:0.628287
[3]	train-auc:0.663850	test-auc:0.629045
[4]	train-auc:0.675769	test-auc:0.639435
[5]	train-auc:0.678071	test-auc:0.639245
[6]	train-auc:0.682812	test-auc:0.640049
[7]	train-auc:0.686980	test-auc:0.643947
[8]	train-auc:0.690062	test-auc:0.644320
[9]	train-auc:0.690602	test-auc:0.644670
[10]	train-auc:0.693892	test-auc:0.647515
 ... ...
[125]	train-auc:0.786332	test-auc:0.678957
[126]	train-auc:0.786529	test-auc:0.678990
[127]	train-auc:0.786779	test-auc:0.678862

Time difference of 17.5189 secs
```

```
$ks_train
[1] 0.4213069

$auc_train
[1] 0.7867787

$ks_test
[1] 0.2745123

$auc_test
[1] 0.6788622
```

