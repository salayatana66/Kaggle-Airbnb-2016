##########################################################################
# XGboost which separates main countries (NDF, US, other) from 
# others, and ignores some countries like AU, CA which seem to lower performance
# LB Score ~ 0.87
##########################################################################
library(reshape2)
library(dplyr)
library(xgboost)
library(caret)
library(pROC)
library(mlr)
setwd('~/leave_academia/kaggle/airbnb')

# Load data ------------------------------------------------------
load('cleaned_data.RData')
target.freq <- summary(na.omit(all.data$response))/length(na.omit(all.data$response))

# Create xgboost data ----------------------------------
xgboost.data <- all.data
indx.to.num <- which(sapply(xgboost.data, is.factor))[-c(1,2)] # leave response, id unaffected
# convert to numeric
for(i in indx.to.num) {
    levels(xgboost.data[, i]) <- c(0:(length(levels(xgboost.data[, i]))-1))
    xgboost.data[, i] <- as.numeric(xgboost.data[, i])
}
# transform date_account_create
xgboost.data$date_account_created_year <- (xgboost.data$date_account_created_year-2010)/4
xgboost.data$date_account_created_month <- (xgboost.data$date_account_created_year)/11
xgboost.data$date_account_created_day <- NULL # remove

# transform timestamp_first_active
xgboost.data$timestamp_first_active_year <- (xgboost.data$timestamp_first_active_year-2009)/5
xgboost.data$timestamp_first_active_month <- (xgboost.data$timestamp_first_active_month)/11
xgboost.data$timestamp_first_active_day <- NULL # remove

# train data
xgboost.train.data <- xgboost.data[!is.na(xgboost.data$response), ]
# test data
xgboost.test.data <- xgboost.data[is.na(xgboost.data$response), ]

##########################################################################
##########################################################################
### Performance metrics
##########################################################################
##########################################################################

# 3 for 'main'
ndcg3 <- function(pred, dtrain, labels = NULL) { # dtrain is the data object
    num.class = 3
    top <- t(apply(matrix(pred, nrow = 3), 2, function(y) order(y)[num.class:(num.class-2)]-1))
    if(is.null(labels)) labs = getinfo(dtrain, 'label')
    val <- ifelse(top == labs,1,0)

    dcg <- function(y) sum((2^y-1)/log(2:(length(y)+1),base=2))
    ndcg <- mean(apply(val,1,dcg))
    return(list(metric = 'ndcg3', value = ndcg))
}

ndgc_metric3 <- function(prediction, reference, weight = c(1,1,1)) {   
    num.class = 3
    name.class = c('NDF', 'other', 'US')
    pred <- as.matrix(prediction)

    # weights
    pred[, 1] <- weight[1] * pred[, 1]
    pred[, 2] <- weight[2] * pred[, 2]
    pred[, 3] <- weight[3] * pred[, 3]

    top <- t(apply(pred, 1, function(y) name.class[rev(order(y))]))
    lgc <- top == as.character(reference) # matrix of agreement
    
    dcg <- function(y) sum( (2^y - 1)/log(2:(length(y)+1), base = 2))
    mean(apply(lgc, 1, dcg))
}

my_ndgcfun3 <- function(task, model, pred, feats, m) {
    ndgc_metric3(prediction = pred$data[3:5], reference = pred$data$truth)
}
my.ndgcfun3 <- makeMeasure(id = 'my.ndgc3', minimize = F,
                           properties = c('classif','classif.multi', 'response'),
                           fun = my_ndgcfun3, best = 1, worst = 0)

# '2' for 'special'
ndcg2s <- function(pred, dtrain, labels = NULL) { # dtrain is the data object
    num.class = 4
    top <- t(apply(matrix(pred, nrow = num.class), 2, function(y) order(y)[num.class:(num.class-1)]-1))
    if(is.null(labels)) labs = getinfo(dtrain, 'label')
    val <- ifelse(top == labs,1,0)

    dcg <- function(y) sum((2^y-1)/log(2:(length(y)+1),base=2))
    ndcg <- mean(apply(val,1,dcg))
    return(list(metric = 'ndcg2s', value = ndcg))
}

ndgc_metric2s <- function(prediction, reference, weight = c(1,1,1,1)) {   
    num.class = 4
    name.class = c('ES', 'FR', 'GB', 'IT')
    pred <- as.matrix(prediction)

    # weights
    pred[, 1] <- weight[1] * pred[, 1]
    pred[, 2] <- weight[2] * pred[, 2]
    pred[, 3] <- weight[3] * pred[, 3]
    pred[, 4] <- weight[4] * pred[, 4]

    top <- t(apply(pred, 1, function(y) name.class[rev(order(y))[1:2]])) # just best 2 classes
    lgc <- top == as.character(reference) # matrix of agreement
    
    dcg <- function(y) sum( (2^y - 1)/log(2:(length(y)+1), base = 2))
    mean(apply(lgc, 1, dcg))
}

my_ndgcfun2s <- function(task, model, pred, feats, m) {
    ndgc_metric2s(prediction = pred$data[3:6], reference = pred$data$truth)
}
my.ndgcfun2s <- makeMeasure(id = 'my.ndgc2s', minimize = F,
                            properties = c('classif','classif.multi', 'response'),
                            fun = my_ndgcfun2s, best = 1, worst = 0)


##########################################################################
##########################################################################
# Tuning on 'main'
##########################################################################
##########################################################################
# 'main' data
main.class.indx <- which(xgboost.train.data$response %in% c('US', 'NDF', 'other'))

xgboost.main.data <- with(xgboost.train.data, data.frame(response = response[main.class.indx,
                                                                             drop = T],
                                                         xgboost.train.data[main.class.indx, main.features]))

# Step1: task and learner
train.task <- makeClassifTask(data = xgboost.main.data, target = 'response')
set.seed(954) # is a seed needed when creating the learner???
lrn.main <- makeLearner('classif.xgboost', predict.type = "prob")
lrn.main$par.vals <- list( # parameters not for tuning
    eta = 0.0739,
    gamma = 0.6037,
    #nthread             = 30,
    nrounds             = 15,
    eval_metric = ndcg3,
    objective = "multi:softprob", 
    num_class = 3,
    #print.every.n       = 5,
    #depth = 20,
    colsample_bytree = 0.9837,
    max_depth = 10,
    #min_child_weight = 3,
    subsample = 0.6826
)

ps.set <- makeParamSet(
    makeNumericParam('eta', lower = 0.07, upper = 0.2),
    makeNumericParam('gamma', lower = 0.2, upper = 0.8),
    makeNumericParam('subsample', lower = 0.5, upper = 1),
    makeNumericParam('colsample_bytree', lower = 0.5, upper = 1),
    makeDiscreteParam('max_depth', values = c(10, 15, 20, 25)),
    makeDiscreteParam('nrounds', values = c(15, 20, 25, 30, 35))
)

# Step2: search and validation 
ctrl <- makeTuneControlRandom(budget = 30, maxit = 30)
rdesc <- makeResampleDesc('CV', iters = 5L, stratify = T)

# Step3: tune
set.seed(43)
tune.lrn.main <- tuneParams(lrn.main, task = train.task, resampling = rdesc,
                             par.set = ps.set, control = ctrl,
                             measures = list(my.ndgcfun3, acc))

lrn.main <- setHyperPars(lrn.main, par.vals = tune.lrn.main$x)

# Step3+1/2: improve on the previous search
lrn.main$par.vals <- list( # parameters not for tuning
    eta = 0.15,
    gamma = 0.6037,
    #nthread             = 30,
    nrounds             = 25,
    eval_metric = ndcg3,
    objective = "multi:softprob", 
    num_class = 3,
    #print.every.n       = 5,
    #depth = 20,
    colsample_bytree = 1,
    max_depth = 10,
    #min_child_weight = 3,
    subsample = 0.6826
)

ps.set <- makeParamSet(
    makeDiscreteParam('eta', values = c(0.07, 0.1, 0.15)),
    makeDiscreteParam('colsample_bytree', values = c(0.8, 1)),
    makeDiscreteParam('max_depth', values = c(10, 15, 20)),
    makeDiscreteParam('nrounds', values = c(15, 25))
)

ctrl = makeTuneControlGrid()
rdesc =  makeResampleDesc('CV', iters = 5L, stratify = T)

set.seed(77)
tune.lrn.main <-  tuneParams(lrn.main, task = train.task, resampling = rdesc,
                             par.set = ps.set, control = ctrl,
                             measures = list(my.ndgcfun3, acc))
# Step 4: assess performance via cross-validation
validated.lrn <- crossval(lrn.main, train.task, iter = 10, measures = my.ndgcfun3, show.info = TRUE)

##########################################################################
##########################################################################
### MLR tuning for 'special'
##########################################################################
##########################################################################
special.class.indx <- which(xgboost.train.data$response %in% c('FR', 'IT', 'GB', 'ES'))
xgboost.special.data <- with(xgboost.train.data, data.frame(response = response[special.class.indx,
                                                                             drop = T],
                                                         xgboost.train.data[special.class.indx,features.special]))

# Step1: task and learners
train.task <- makeClassifTask(data = xgboost.special.data, target = 'response')
set.seed(544)
lrn.special <- makeLearner('classif.xgboost', predict.type = "prob")

lrn.special$par.vals <- list( # parameters not for tuning
    eta = 0.12,
    gamma = 0.0808,
    #nthread             = 30,
    nrounds             = 20,
    eval_metric = ndcg2s,
    objective = "multi:softprob", 
    num_class = 4,
    #print.every.n       = 5,
    #depth = 20,
    colsample_bytree = 0.544,
    max_depth = 10,
    #min_child_weight = 3,
    subsample = 0.87
)

ps.set <- makeParamSet(
    makeNumericParam('eta', lower = 0.05, upper = 0.2),
    makeNumericParam('gamma', lower = 0, upper = 1),
    makeNumericParam('subsample', lower = 0.5, upper = 1),
    makeNumericParam('colsample_bytree', lower = 0.5, upper = 1),
    makeDiscreteParam('max_depth', values = c(10, 15, 20)),
    makeDiscreteParam('nrounds', values = c(15, 20, 25, 30, 35))
)

# Step2: search and validation 
ctrl <- makeTuneControlRandom(budget = 30, maxit = 30)
rdesc <- makeResampleDesc('CV', iters = 10L, stratify = T)

# Step3: tune
set.seed(43)
tune.lrn.special <- tuneParams(lrn.special, task = train.task, resampling = rdesc,
                             par.set = ps.set, control = ctrl,
                             measures = list(my.ndgcfun2s, acc))

lrn.special <- setHyperPars(lrn.special, par.vals = tune.lrn.special$x)

# Step 4:  assess performace via cross-validation
set.seed(47)
validated.lrn <- crossval(lrn.special, train.task, iter = 10, measures = my.ndgcfun2s, show.info = TRUE)

##########################################################################
##########################################################################
### Create predictions
##########################################################################
##########################################################################

# Helper --------------------------------------------------------------------------
# predictor for main countries

create.xgb.main <- function(prediction, weights = c(1,1,1)) { # x can be passed as a data.frame
    num.main.countries <- c(0:2)
    names(num.main.countries) <- c('NDF', 'other', 'US')

    pred <- prediction$data[1:3]
    pred[,1] <- weights[1] * pred[, 1]
    pred[,2] <- weights[2] * pred[, 2]
    pred[,3] <- weights[3] * pred[, 3]
    colnames(pred) <- names(num.main.countries)

    stringPreds <- as.data.frame(t(apply(pred, 1, function(z) names(sort(z)[3:1]))))
    colnames(stringPreds) <- c('V1', 'V2', 'V3')

    stringPreds
}

create.xgb.special <- function(prediction, weights = c(0.9485752, 1.1017281, 0.8690824, 1.1246816)) {
    num.main.countries <- c(0:3)
    names(num.main.countries) <- c('ES', 'FR', 'GB', 'IT')

    pred <- prediction$data[1:4]
    pred[, 1] <- weights[1] * pred[, 1]
    pred[, 2] <- weights[2] * pred[, 2]
    pred[, 3] <- weights[3] * pred[, 3]
    pred[, 4] <- weights[4] * pred[, 4]

    colnames(pred) <- names(num.main.countries)

    stringPreds <- as.data.frame(t(apply(pred, 1, function(z) names(sort(z)[4:3]))))
    colnames(stringPreds) <- c('V1', 'V2')

    stringPreds
}

format.pred <- function(pred.main, pred.special) {
    colnames(pred.main) <- c('V1', 'V2', 'V3')
    colnames(pred.special) <- c('V4', 'V5')
    preds <- data.frame(id = xgboost.test.data$id, row = 1:dim(xgboost.test.data)[1],
                        pred.main, pred.special)
    to_shape <- melt(preds, id.vars = c('id','row'))
    to_shape <- arrange(to_shape, row, variable)
    to_shape <- to_shape[c('id', 'value')]
    names(to_shape) <- c('id', 'country')

    to_shape
}

##########################################################################
# Train and predict
##########################################################################

# Step1: predictions on main
train.task <- makeClassifTask(data = xgboost.main.data, target = 'response')
set.seed(995)
lrn.main <- makeLearner('classif.xgboost', predict.type = "prob")
lrn.main$par.vals <- list( # parameters not for tuning
    eta = 0.15,
    gamma = 0.6037,
    #nthread             = 30,
    nrounds             = 25,
    eval_metric = ndcg3,
    objective = "multi:softprob", 
    num_class = 3,
    #print.every.n       = 5,
    #depth = 20,
    colsample_bytree = 0.8,
    max_depth = 10,
    #min_child_weight = 3,
    subsample = 0.6826
)
set.seed(124)
global.main.train <- train(lrn.main, train.task)
tt <- predict(global.main.train, newdata = xgboost.test.data[,-c(1,2)])
prediction <- list()
prediction$data <- tt$data
test.main.pred <- create.xgb.main(prediction = prediction)


# Step 2: predictions on special
train.task <- makeClassifTask(data = xgboost.special.data, target = 'response')
set.seed(544) # seed might be useless
lrn.special <- makeLearner('classif.xgboost', predict.type = "prob")
lrn.special$par.vals <- list( # parameters not for tuning
    eta = 0.12,
    gamma = 0.0808,
    #nthread             = 30,
    nrounds             = 20,
    eval_metric = ndcg2s,
    objective = "multi:softprob", 
    num_class = 4,
    #print.every.n       = 5,
    #depth = 20,
    colsample_bytree = 0.544,
    max_depth = 10,
    #min_child_weight = 3,
    subsample = 0.87
)

set.seed(99)
global.special.train <- train(lrn.special, train.task)
tt <- predict(global.special.train, newdata = xgboost.test.data[,-c(1,2)])
prediction <- list()
prediction$data <- tt$data
test.special.pred <- create.xgb.special(prediction = prediction)

# create output 
test.out <- format.pred(test.main.pred, test.special.pred)

# write
write.table(file = '2016feb8_stepwise.csv', test.out, quote = F, row.names = F, sep = ',') # = 0.87098
