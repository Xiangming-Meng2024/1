library (readxl)
data <- (read_xlsx('模型开发队列'))
library(readxl)
data <- read_excel("模型开发队列.xlsx")
View(data)
class(data)
data <- as.data.frame(data)
class(data)
row.names(data) <- data [ , 1]
data <- data [ , -1]
data <- data [ , -5]
data1 <- data [, -1:3]
data1 <- data [, -c(1:3)]
View(data1)
pr.out <- prcomp (data1, scale = TRUE)
summary(pr.out)
biplot(pr.out, scale=0)
PCA <- pr.out$x
View(pr.out)
pr.out[["x"]]
View(PCA)
summary(pr.out)
library ('corrplot')
data2 <- as.matrix(data1)
View(data2)
corr <- cor (data2)
View(data)
View(corr)
library(readxl)
data <- read_excel("模型开发队列.xlsx")
View(data)
summary(data)
library(readxl)
data <- read_excel("模型开发队列.xlsx")
View(data)
class(data)
data <- as.data.frame(data)
row.names(data) <- data [ , 1]
data <- data [ , -1]
data1 <- data [, -c(1:3)]
pr.out <- prcomp (data1, scale = TRUE)
summary(pr.out)
biplot(pr.out, scale=0)
PCA <- pr.out$x
View(PCA)
library ('corrplot')
data2 <- as.matrix(data1)
corr <- cor (data2)
View(corr)
summary (data)
names (data)
head (data)
y <- "Subgroup"
x1 <- c ("Age" , "AFP" , "DCP" , "ALT" , "AST" , "AST/ALT" , "GGT" , "ALB" , "TBIL" , "DBIL" , "IDBIL" , "TP" , "GLB"   ,   "A/G"   ,  "ALP")
x2 <- "Sex"
x <- c(x1,x2)
results1 <- print (table1, showAllLevels = TRUE)
table1 <- CreateTableOne ( vars = c (x1 , x2),
                           data = data,
                           factorVars = x2,
                           strata = 'Subgroup' , addOverall = TRUE)
library(tableone)
table1 <- CreateTableOne ( vars = c (x1 , x2),
                           data = data,
                           factorVars = x2,
                           strata = 'Subgroup' , addOverall = TRUE)
results1 <- print (table1, showAllLevels = TRUE)
library ("readxl")
x1 <- c ("Age" , "AFP" , "DCP" , "ALT" , "AST" , "AST/ALT" , "GGT" , "ALB" , "TBIL" , "DBIL" , "IDBIL" , "TP" , "GLB"   ,   "A/G"   ,  "ALP")
x2 <- "Sex"
x <- c(x1,x2)
y <- "Subgroup"
table <- CreateTableOne ( vars = x,
                          factorVars = x2,
                          argsApprox = list (correct = FALSE),
                          strata = y,
                          data = data,
                          addOverall = TRUE)
library(tableone)
table <- CreateTableOne ( vars = x,
                          factorVars = x2,
                          argsApprox = list (correct = FALSE),
                          strata = y,
                          data = data,
                          addOverall = TRUE)
results1 <- print (table, showAllLevels = TRUE)
results2 <- print (table, showAllLevels = TRUE, nonnormal = x1)
library ("readxl")
data <- read_xlsx("模型开发队列.xlsx")
library(readxl)
模型开发队列_训练集_验证集_ <- read_excel("模型开发队列（训练集+验证集）.xlsx")
View(模型开发队列_训练集_验证集_)
rm(模型开发队列_训练集_验证集_)
data <- read_xlsx("模型开发队列（训练集+验证集）.xlsx")
class(data)
row.names(data) <- data [ , 1]
data <- as.data.frame(data)
row.names(data) <- data [ , 1]
data <- data [ , -1]
data1 <- data [, -c(1:3)]
pr.out <- prcomp (data1, scale = TRUE)
summary(pr.out)
biplot(pr.out, scale=0)
PCA <- pr.out$x
write.csv (PCA, "R语言-主成分分析.csv")
library ('corrplot')
library (corrplot)
install.packages ('corrplot')
library (corrplot)
data2 <- as.matrix(data1)
corr <- cor (data2)
corr
library ("readxl")
data <- read_xlsx("模型开发队列（训练集+验证集）.xlsx")
library ("readxl")
data <- read_xlsx("模型开发队列（训练集+验证集）.xlsx")
class(data)
data <- as.data.frame(data)
row.names(data) <- data [ , 1]
data <- data [ , -1]
data1 <- data [, -c(1:3)]
pr.out <- prcomp (data1, scale = TRUE)
summary(pr.out)
biplot(pr.out, scale=0)
PCA <- pr.out$x
write.csv (PCA, "R语言-主成分分析.csv")
pr.out <- prcomp (data, scale = TRUE)
summary(pr.out)
biplot(pr.out, scale=0)
PCA <- pr.out$x
write.csv (PCA, "R语言-主成分分析.csv")
data1 <- data [, -1]
pr.out <- prcomp (data1, scale = TRUE)
summary(pr.out)
biplot(pr.out, scale=0)
PCA <- pr.out$x
write.csv (PCA, "R语言-主成分分析.csv")
pr.out$Importance of components
pr.out[["x"]]
data2 <- as.matrix(data1)
corr <- cor (data2)
