train <- read.csv(file.choose())
library(ggplot2)

summary(train)
#find: correlation between armed and manner of death: does armed likely to be killed?

#kaggle people's work
a <- train %>% group_by(body_camera, date2 = cut(date, "month")) %>% summarize(count = n())

#bar plot
names(train)
ggplot(train, aes(race)) + geom_bar(aes(fill = gender))
ggplot(train, aes(race)) + geom_bar(aes(fill = gender), position = "fill")
tab <- table(train$race, train$gender)
prop.table(tab)
#mostly men, with white men with the hightest proportion, thne black and hispanic. After that is white women. 

ggplot(train, aes(race)) + geom_bar(aes(fill = manner_of_death))
ggplot(train, aes(race)) + geom_bar(aes(fill = manner_of_death), position = "fill")
#Native has the hightest shot percentage to death than any other race

ggplot(train, aes(race)) + geom_bar(aes(fill = signs_of_mental_illness))
ggplot(train, aes(race)) + geom_bar(aes(fill = signs_of_mental_illness), position = "fill")
#over a quarter of asian and white have signs of mental illness, while black has the lowest signs of mental illness

ggplot(train, aes(race)) + geom_bar(aes(fill = threat_level))
ggplot(train, aes(race)) + geom_bar(aes(fill = threat_level), position = "fill")
#black people tend to be attacked compare to other races, following by white and others.

ggplot(train, aes(age)) + geom_histogram(bin = 10)
#right skewed meaning most cases are involve younger people
ggplot(train, aes(age)) + geom_histogram(aes(fill = manner_of_death), bin = 10)
ggplot(train, aes(age))  + geom_histogram(aes(fill = body_camera), bin = 10)
ggplot(train, aes(age)) + geom_histogram(aes(fill = threat_level), bin = 10)

ggplot(train, aes(x = gender, y = age)) + geom_boxplot()
#for ages at min, 25th quantile, median and 75th quantile, they are similar for both genders
ggplot(train, aes(x = body_camera, y = age)) + geom_boxplot()
ggplot(train, aes(x = race, y = age)) + geom_boxplot()
#Median for white tend to be higher compare to other races, and they have the youngest and oldest to be shot by the police
#For Black, over 25% were under the age of 25 and have one of the lowest median age along with the 'other' race. 
ggplot(train, aes(x = threat_level, y = age)) + geom_boxplot()
ggplot(train, aes(x = flee, y = age)) + geom_boxplot() 
#The age of people who flee in general tend to be younger, while the median age of people who did not flee are older.
#75% of people who tend to flee on car and foot tend to be younger than around 40 years old
ggplot(train, aes(x = signs_of_mental_illness, y = age)) + geom_boxplot()
#people with mental illness tend to be older
ggplot(train, aes(x = arms_category, y = age)) + geom_boxplot() + theme(axis.text.x = element_text(angle = 45))
#people who were unarmed or with electrical or with vehicles devides have the lowest age median
#this matches the statistics showing that younger people tend to flee on cars
#people with explosives and piercing objects tend to be older in age
ggplot(train, aes(x = arms_category)) + geom_bar(binwidth = 10) + theme(axis.text.x = element_text(angle = 45)) + ggtitle("Total number in each arms category") 
summary(train)
ggplot(train, aes(x = manner_of_death)) + geom_bar() + facet_wrap(~gender)
ggplot(train, aes(x = age)) + geom_histogram(aes(fill = manner_of_death)) + facet_wrap(~gender)

death_mental_illness <- table(train$signs_of_mental_illness, train$manner_of_death)
prop.table(death_mental_illness)
#proportion base on total
prop.table(death_mental_illness, margin = 1)
#proportion base on rows
prop.table(death_mental_illness, margin = 2)
#proportion base on columns

#3 ways including gender
three_tab <- xtabs(~manner_of_death+signs_of_mental_illness+gender, train)
three_tab
three_tab_2 <- ftable(three_tab)
three_tab_2

#if they have mental health: decision tree, random forest
#if flee or not flee, how will you be attacked
#check race
library(dplyr)
library(caret)
library(rpart)
library(knitr)
library(ggplot2)
library(cowplot)
library(randomForest)
library(DMwR)

set.seed(1)
summary(train)
str(train)
train_alg <- train %>% select(-c(id, name, date, armed, city, state))
summary(train_alg)
str(train_alg)

#intrain <- createResample(y = train_alg$body_camera, times = 10, list = FALSE)
intrain <- createDataPartition(y = train_alg$body_camera, p = 0.7, list = FALSE)
train_set <- train_alg[intrain,]
test_set <- train_alg[-intrain,]
nrow(train_set)
nrow(test_set)
str(train_alg)

#check confusion Matrix, Precision, Recall, F1 Score, ROC curve
#try to implement SMOTE, in DMwR package

tree1_decision <- rpart(body_camera~., data = train_set, method = "class")
summary(tree1_decision)

tree1_pred <- predict(tree1_decision, test_set, type = "class")
confusionMatrix(test_set$body_camera ,tree1_pred)
#Accuracy is 88.2% by guessing all false
#tree1.imputed <- rfImpute(body_camera~., data = train_set, iter = 6)
#only checking whether there are any NA
tree1_rf <- randomForest(body_camera~., data = train_set, proximity = TRUE)
tree1_rf
#same accuracy as decision tree, perform poorly on checking whether it is true
#see whether 500 trees are enough by plotting the OOB error
oob.error.data <- data_frame(
  Trees = rep(1:nrow(tree1_rf$err.rate), times = 3), 
  Type = rep(c("OOB", "True", "False"), each = nrow(tree1_rf$err.rate)),
  Error = c(tree1_rf$err.rate[,"OOB"],
  tree1_rf$err.rate[,"True"],
  tree1_rf$err.rate[,"False"]
))

ggplot(data = oob.error.data, aes(x = Trees, y = Error)) + geom_line(aes(color = Type))

tree1_rf$err.rate

#attempt to add more tree and see whether there is any difference, turns out no difference
tree1_rf2 <- randomForest(body_camera~., data = train_set, ntree = 1000, proximity = TRUE)
tree1_rf2
summary(train_set)

#attempt the use of SMOTE; double the positive case, and half of negative case
train1_smote <- SMOTE(body_camera~., data = train_set, perc.over = 100, perc.under = 200)
prop.table(table(train1_smote$body_camera))
#positive and negative cases are 50/50
tree1_rf3 <- randomForest(body_camera~., data = train1_smote,proximity = TRUE)
tree1_rf3
tree1_rf3_predict <- predict(tree1_rf3, test_set, type = "class")
confusionMatrix(test_set$body_camera, tree1_rf3_predict)

oob.error.data1 <- data_frame(Trees = rep(1:nrow(tree1_rf3$err.rate), times = 3), Type = rep(c("OOB", "TRUE", "FALSE"), each = nrow(tree1_rf3$err.rate)), Error = c(tree1_rf3$err.rate[, "OOB"], tree1_rf3$err.rate[,"True"],tree1_rf3$err.rate[,"False"]))

ggplot(oob.error.data1, aes(Trees, Error)) + geom_line(aes(color = Type))





#mental illness modeling
tree2_decision <- rpart(signs_of_mental_illness~., data = train_set, method = "class")
summary(tree2_decision)


tree2_pred <- predict(tree2_decision, test_set, type = "class")
table(Predicted=tree2_pred, Observed = test_set$signs_of_mental_illness)
confusionMatrix(test_set$signs_of_mental_illness, tree2_pred) 
#Accuracy is 75.7% by guessing all false

#use cross validation, random forest
tree2_rf <- randomForest(signs_of_mental_illness~., train_set, proximity = TRUE)
tree2_rf
tree2_rf_predict <- predict(tree2_rf, test_set, type = "class")
confusionMatrix(test_set$signs_of_mental_illness, tree2_rf_predict)
#Accuracy is 75.6%

#apply SMOTE to resample the data
train2_smote <- SMOTE(signs_of_mental_illness~., data = train_set, perc.over = 100, perc.under = 200)
prop.table(table(train2_smote$signs_of_mental_illness))
tree2_rf2 <- randomForest(signs_of_mental_illness~., data = train2_smote, proximity = TRUE)
tree2_rf2
tree2_rf2_predict <- predict(tree2_rf2, test_set, type = "class")
confusionMatrix(test_set$signs_of_mental_illness, tree2_rf2_predict)

oob.err.data2 <- data_frame(Trees = rep(1:nrow(tree2_rf2$err.rate), times = 3), Type = rep(c("OOB", "True","False"), each = nrow(tree2_rf2$err.rate)), Error = c(tree2_rf2$err.rate[,"OOB"], tree2_rf2$err.rate[,"True"],tree2_rf2$err.rate[,"False"]))

ggplot(oob.err.data2, aes(Trees, Error)) + geom_line(aes(color = Type))



