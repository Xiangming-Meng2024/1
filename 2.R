train_set3$svm_train_prob <- svm_train_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
train_set3
train_set3$knn_train_prob <- knn_train_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
train_set3
knn_train_dca <- dca(data = train_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="knn_train_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
rf_train_dca <- dca(data = train_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="rf_train_prob", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (rf_train_dca[["net.benefit"]],"训练集-随机森林-决策曲线.csv")
rf_valid_dca <- dca(data = valid_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="rf_valid_prob", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (rf_valid_dca[["net.benefit"]],"验证集-随机森林-决策曲线.csv")
rf_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                   outcome="Subgroup", # 指定结果变量
                   predictors="rf_test_prob", # 指定预测变量
                   probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                   graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
test_set3
test_set3$rf_test_prob <- rf_test_prob[,2] #将随机森林模型预测的概率添加到训练集test_set3中
test_set3
rf_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                   outcome="Subgroup", # 指定结果变量
                   predictors="rf_test_prob", # 指定预测变量
                   probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                   graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
rf_test_dca
#测试集
rf_test_prob[,2]
test_set3
rf_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                   outcome="Subgroup", # 指定结果变量
                   predictors="rf_test_prob", # 指定预测变量
                   probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                   graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
test_set3$rf_test_prob <- rf_test_prob[,2] #将随机森林模型预测的概率添加到训练集test_set3中
test_set3$rf_test_prob <- rf_test_prob[,2] #将随机森林模型预测的概率添加到训练集test_set3中
test_set3
valid_set3
test_set3$log_test_response <- log_test_response #将logistic回归模型预测的概率添加到训练集test_set3中
test_set3
test_set3 <- test_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为test_set3
test_set3$log_test_response <- log_test_response #将logistic回归模型预测的概率添加到训练集test_set3中
test_set3
test_set3$rf_test_prob <- rf_test_prob[,2] #将随机森林模型预测的概率添加到训练集test_set3中
test_set3
test_set3$svm_test_prob <- svm_test_prob[,2] #将随机森林模型预测的概率添加到训练集test_set3中
test_set3
write.csv (rf_train_dca[["net.benefit"]],"训练集-SVM-决策曲线.csv")
write.csv (knn_train_dca[["net.benefit"]],"训练集-KNN-决策曲线.csv")
test_set3
rf_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                   outcome="Subgroup", # 指定结果变量
                   predictors="svm_test_prob", # 指定预测变量
                   probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                   graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (rf_test_dca[["net.benefit"]],"测试集-SVM-决策曲线.csv")
train_set3
knn_train_dca <- dca(data = train_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="knn_train_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (knn_train_dca[["net.benefit"]],"训练集-KNN-决策曲线.csv")
write.csv (knn_train_dca[["net.benefit"]],"训练集-KNN-决策曲线.csv")
knn_train_dca[["net.benefit"]]
#验证集
knn_valid_prob[,2]
valid_set3$knn_valid_prob <- knn_valid_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
valid_set3
knn_valid_dca <- dca(data = valid_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="knn_valid_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (knn_valid_dca[["net.benefit"]],"验证集-KNN-决策曲线.csv")
test_set3$knn_test_prob <- knn_test_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
test_set3
knn_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="knn_test_prob", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (knn_test_dca[["net.benefit"]],"测试集-KNN-决策曲线.csv")
rf_train_dca[["net.benefit"]]
write.csv (rf_train_dca[["net.benefit"]],"训练集-随机森林-决策曲线.csv")
log_train_dca[["net.benefit"]]
write.csv (log_train_dca[["net.benefit"]],"训练集-logistic-决策曲线.csv")
log_train_dca[["net.benefit"]]
rf_train_dca[["net.benefit"]]
svm_train_dca <- dca(data = train_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="svm_train_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (svm_train_dca[["net.benefit"]],"训练集-SVM-决策曲线.csv")
svm_train_dca[["net.benefit"]]
write.csv (svm_train_dca[["net.benefit"]],"训练集-SVM-决策曲线.csv")
svm_valid_dca <- dca(data = valid_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="svm_valid_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (svm_valid_dca[["net.benefit"]],"验证集-SVM-决策曲线.csv")
rf_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                   outcome="Subgroup", # 指定结果变量
                   predictors="svm_test_prob", # 指定预测变量
                   probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                   graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (svm_test_dca[["net.benefit"]],"测试集-SVM-决策曲线.csv")
svm_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="svm_test_prob", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (svm_test_dca[["net.benefit"]],"测试集-SVM-决策曲线.csv")
log_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="log_test_response", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
log_test_dca[["net.benefit"]]
test_set3
rf_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                   outcome="Subgroup", # 指定结果变量
                   predictors="rf_test_prob", # 指定预测变量
                   probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                   graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (rf_test_dca[["net.benefit"]],"测试集-随机森林-决策曲线.csv")
rf_test_dca[["net.benefit"]]
write.csv (rf_test_dca[["net.benefit"]],"测试集-随机森林-决策曲线.csv")
train_set
X <- as.matrix(train_set[ , 2:17])
Y <- as.matrix (train_set [ , 1])
lambdas <- seq(from=0, to=0.5, length.out=200)
library (glmnet)
install.packages('glmnet')
library (glmnet)
lambdas <- seq(from=0, to=0.5, length.out=200)
set.seed (123)
train_cv.lasso <- cv.glmnet (x=X, y=Y, alpha =1, lambda=lambdas, nfolds=3,  family="binomial")
lambda_1se <- train_cv.lasso$lambda.1se
lambda_1se
lambda_1se.coef <- coef (train_cv.lasso$glmnet.fit, s=lambda_1se)
lambda_1se.coef
View(train_cv.lasso)
write.csv(lambda_1se.coef,"lasso回归变量筛选.csv")
View(lambda_1se.coef)
lambda_1se.coef@x
lambda_1se.coef@Dimnames
#导入数据集
train_set2 <- train_set[, 1:5] #训练集
valid_set2 <- valid_set[, 1:5] #验证集
test_set2 <- test_set [, 1:5] #测试集
#将训练集中的二分类变量改成因子形式
train_set2$Subgroup <- factor (train_set2$Subgroup)
train_set2$Sex <- factor (train_set2$Sex)
#查看更改后的训练集
View (train_set2)
summary(train_set2)
table (train_set2$Subgroup)
dim (train_set2)
#将验证集中的二分类变量改成因子形式
valid_set2$Subgroup <- factor (valid_set2$Subgroup)
valid_set2$Sex <- factor (valid_set2$Sex)
#查看更改后的验证集
View (valid_set2)
summary(valid_set2)
table (valid_set2$Subgroup)
dim (valid_set2)
#将测试集中的二分类变量改成因子形式
test_set2$Subgroup <- factor (test_set2$Subgroup)
test_set2$Sex <- factor (test_set2$Sex)
#查看更改后的验证集
View (test_set2)
summary (test_set2)
table (test_set2$Subgroup)
dim (test_set2)
remotes::install_github("tidymodels/probably")
suppressMessages(library(tidymodels))
suppressMessages(library(probably))
valid_set3
test_set3
train_set3
cal_plot_windowed(test_set3$Subgroup, test$log_train_prob)
test_set3 %>%
  cal_plot_windowed(Subgroup, log_train_prob)
test_set3 %>%
  cal_plot_windowed(Subgroup, all_of(log_train_prob))
test_log_cal <- test_set3 %>%
  mutate(pred_rnd = round(log_train_prob, 2)
  ) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(log_train_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
train_set3 <- train_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为train_set3
train_set3$log_train_prob <- log_train_prob
test_set3
test_set3 %>%
  cal_plot_windowed(Subgroup, all_of(log_test_prob))
test_set3
test_set3 %>%
  cal_plot_windowed(Subgroup, all_of(log_test_response))
test_set3 %>%
  cal_plot_windowed(Subgroup, log_test_response)
test_log_cal <- test_set3 %>%
  cal_plot_windowed(Subgroup, log_test_response)
test_log_cal
View(test_log_cal)
test_log_cal[["data"]][["event_rate"]]
test_log_cal[["data"]][["predicted_midpoint"]]
rms::val.prob(
  p = test_set3$log_test_response,
  y = Subgroup,
  cex = 1
)
rms::val.prob(
  p = test_set3$log_test_response,
  y = test_set3$Subgroup,
  cex = 1
)
rms::val.prob(
  p = test_set3$log_test_response,
  y = test_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = test_set3$rf_test_prob,
  y = test_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
View(test_set3)
rms::val.prob(
  p = test_set3$svm_test_prob,
  y = test_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = test_set3$knn_test_prob,
  y = test_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = test_set3$svm_test_prob,
  y = test_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = test_set3$knn_test_prob,
  y = test_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
test_log_cal <- test_set3 %>%
  cal_plot_windowed(Subgroup, log_test_response)
test_log_cal <- test_set3 %>%
  cal_plot_windowed(Subgroup, log_test_response)
valid_set3
rms::val.prob(
  p = valid_set3$log_valid_response,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = valid_set3$rf_valid_prob,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = valid_set3$log_valid_response,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = valid_set3$rf_valid_prob,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = valid_set3$svm_valid_prob,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = valid_set3$knn_valid_prob,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
rms::val.prob(
  p = valid_set3$svm_valid_prob,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
test_log_cal <- test_set3 %>% arrange(log_test_response) %>%
  cal_plot_windowed(Subgroup, log_test_response)
test_log_cal
test_log_cal[["data"]][["predicted_midpoint"]]
test_log_cal[["data"]][["event_rate"]]
test_log_cal <- test_set3 %>% arrange(log_test_response) %>%
  cal_plot_windowed(Subgroup, log_test_response, step_size = 0.02)
test_log_cal
test_log_cal <- test_set3 %>% arrange(log_test_response) %>%
  cal_plot_windowed(Subgroup, log_test_response, step_size = 0.05)
test_log_cal
test_log_cal <- test_set3 %>% arrange(log_test_response) %>%
  cal_plot_windowed(Subgroup, log_test_response, step_size = 0.03)
test_log_cal
test_log_cal <- test_set3 %>% arrange(log_test_response) %>%
  cal_plot_windowed(Subgroup, log_test_response, step_size = 0.1)
test_log_cal
test_log_cal <- test_set3 %>%
  mutate(pred_rnd = round(log_test_response, 2)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(log_test_response),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_log_cal <- test_set3 %>%
  mutate(pred_rnd = round(log_test_response, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(log_test_response),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
write.csv(test_log_cal,"测试集-logistic-校准曲线-mlr3包.csv")
rms::val.prob(
  p = test_set3$log_test_response, #预测概率
  y = test_set3$Subgroup, #实际类型
  cex = 1, #输出图的字体大小
  logistic.cal = F #是否对结果进行logistic回归，
)
rms::val.prob(
  p = test_set3$log_test_response, #预测概率
  y = test_set3$Subgroup, #实际类型
  cex = 1, #输出图的字体大小
  logistic.cal = T #是否对结果进行logistic回归，
)
rms::val.prob(
  p = test_set3$log_test_response, #预测概率
  y = test_set3$Subgroup, #实际类型
  cex = 1, #输出图的字体大小
  logistic.cal = F #是否输出对结果进行logistic回归后的图，这里选F就可以了
)
test_set3
head (test_set3)
test_log_cal <- test_set3 %>%
  mutate(pred_rnd = round(rf_test_prob, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(rf_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_log_cal <- test_set3 %>%
  mutate(pred_rnd = round(log_test_response, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(log_test_response),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_rf_cal <- test_set3 %>%
  mutate(pred_rnd = round(rf_test_prob, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(rf_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
write.csv(test_rf_cal,"测试集-随机森林-校准曲线-mlr3包.csv")
test_rf_cal
test_rf_cal
head (test_set3)
test_svm_cal <- test_set3 %>%
  mutate(pred_rnd = round(svm_test_prob, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(svm_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_svm_cal
write.csv(test_svm_cal,"测试集-SVM-校准曲线-mlr3包.csv")
test_knn_cal <- test_set3 %>%
  mutate(pred_rnd = round(knn_test_prob, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(knn_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_knn_cal
write.csv(test_svm_cal,"测试集-SVM-校准曲线-mlr3包.csv")
write.csv(test_svm_cal,"测试集-SVM-校准曲线-mlr3包.csv")
write.csv(test_knn_cal,"测试集-KNN-校准曲线-mlr3包.csv")
head (valid_set3)
valid_log_cal <- valid_set3 %>%
  mutate(pred_rnd = round(log_valid_response, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(log_valid_response),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_log_cal
valid_rf_cal <- valid_set3 %>%
  mutate(pred_rnd = round(rf_valid_prob, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(rf_valid_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_rf_cal
valid_svm_cal <- valid_set3 %>%
  mutate(pred_rnd = round(svm_valid_prob, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(svm_valid_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_svm_cal
head (valid_set3)
valid_knn_cal <- valid_set3 %>%
  mutate(pred_rnd = round(knn_valid_prob, 1)) %>%
  group_by(pred_rnd) %>%
  dplyr::summarize(mean_pred = mean(knn_valid_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_knn_cal
write.csv(valid_log_cal,"验证集-logistic-校准曲线-mlr3包.csv")
write.csv(valid_rf_cal,"验证集-随机森林-校准曲线-mlr3包.csv")
write.csv(valid_svm_cal,"验证集-SVM-校准曲线-mlr3包.csv")
write.csv(valid_knn_cal,"验证集-KNN-校准曲线-mlr3包.csv")
View(ALLdata)
View(clinic_set)
View(knn_test_dca)
View(knn_clinic_prob)
View(log_test_dca)
View(clinic_set2)
View(train_set2)
View(train_set3)
View(valid_set3)
View(train_set2)
View(train_set3)
View(valid_set3)
View(train_log_2)
View(train_log)
View(train_set2)
View(train_set)
install.packages("randomForest")
library(randomForest)
set.seed(123)
train_rf1 <- randomForest(formula = Subgroup ~ . , mtry=4, nodesize=1, replace=TRUE, localImp=TRUE, nPerm=1000, data=train_set2)
View(train_rf)
View(train_rf1)
View(train_rf)
set.seed(123)
train_rf1 <- randomForest(formula = Subgroup ~ . , mtry=4, nodesize=1, replace=TRUE, localImp=TRUE, nPerm=1000, data=train_set2)
which.min (train_rf1$err.rate[, 1])
set.seed(123)
train_rf2 <- randomForest(formula = Subgroup ~ ., ntree=16, mtry=4, nodesize=1, replace=TRUE,localImp=TRUE,nPerm=1000,data=train_set)
set.seed(123)
train_rf2 <- randomForest(formula = Subgroup ~ ., ntree=16, mtry=4, nodesize=1, replace=TRUE,localImp=TRUE,nPerm=1000,data=train_set2)
View(train_rf2)
str(train_set2)
library(randomForest)
set.seed(123)
View(rf_test_cal)
View(rf_test_cal_1)
train_rf1 <- randomForest (formula = Subgroup ~ . , data = train_set2 , ntree = 500, mtry=2 nodesize=1, replace=TRUE, localImp=TRUE, nPerm=1000)
train_rf1 <- randomForest (formula = Subgroup ~ . , data = train_set2 , ntree = 500, mtry=2 nodesize=1, replace=TRUE, localImp=TRUE, nPerm=1000)
train_rf1 <- randomForest (formula = Subgroup ~ . , data = train_set2 , ntree = 500, mtry=2, nodesize=1, replace=TRUE, localImp=TRUE, nPerm=1000)
View(train_rf1)
train_rf1
rf_model
str(train_set2)
set.seed(123)
rf_model <- randomForest (formula = Subgroup ~ . , data = train_set2 , ntree = 500, mtry=2, nodesize=1, replace=TRUE, localImp=TRUE, nPerm=1000)
rf_model
