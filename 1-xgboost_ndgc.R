##########################################################################
## XGboost with custom ndgc5 metric, LB score ~ 0.85
##########################################################################
library(xgboost)
library(reshape2)
library(dplyr)
setwd('~/leave_academia/kaggle/airbnb')

## ndgc metric to evaluate goodness of predictions for xgboost
ndcg5 <- function(pred, dtrain, labels = NULL) { # dtrain is the data object
    num.class = 12
    top <- t(apply(matrix(pred, nrow = 12), 2, function(y) order(y)[num.class:(num.class-4)]-1))
    labs = ifelse(!is.null(labels), labels, getinfo(dtrain,'label'))
    val <- ifelse(top == labs,1,0)

    dcg <- function(y) sum((2^y-1)/log(2:(length(y)+1),base=2))
    ndcg <- mean(apply(val,1,dcg))
    return(list(metric = 'ndcg5', value = ndcg))
}

load('cleaned_data.RData')

# resplit in train/test
df_train = all.data[!(is.na(all.data$response)),]
df_test = all.data[is.na(all.data$response), ]

# create numerical indices for countries
num.countries = c(0:11)
names(num.countries) = as.character(levels(target.data))

# parameters
params = list(eta = 0.1,
              gamma = 0,
               max_depth = 9,               
               subsample = 0.5,
               colsample_bytree = 0.5,
               eval_metric = ndcg5,
               objective = "multi:softprob",
               num_class = 12,
              nthread = 3)

# train with timing
tstart = proc.time()
set.seed(124)
xgb <- xgboost(data = as.matrix(df_train), 
               label = num.countries[target.data], params = params,
               nround=30)
tend = proc.time()

# output and create predictions
preds <- predict(xgb, as.matrix(df_test))
preds <- matrix(preds, nrow = 12)
rownames(preds) <- names(num.countries)
stringPreds <- as.data.frame(t(apply(preds, 2, function(x) names(sort(x)[12:8])))) # best 5
stringPreds <- cbind(row = 1:nrow(test.data), id = test.data$id, stringPreds)

                                        # reshape stringPreds for output
to_shape <- melt(stringPreds, id.vars = c('id', 'row'))
to_shape <- arrange(to_shape,row,variable)
to_shape <- to_shape[c('id', 'value')]
names(to_shape) <- c('id', 'country')
write.table(file = '2015dec26mod14.csv', to_shape, quote = F, row.names = F, sep = ',')

##########################################################################
# Helper function to create predictions
##########################################################################
createPredictions <- function(dftest, model = xgb) {
    # vector to convert country names to numeric // needed by xgboost
    num.countries = c(0:11)
    names(num.countries) = as.character(levels(target.data))

    # create predictions
    preds <- predict(xgb, as.matrix(dftest))
    preds <- matrix(preds, nrow = 12)
    rownames(preds) <- names(num.countries)
    stringPreds <- as.data.frame(t(apply(preds, 2, function(x) names(sort(x)[12:8]))))
    stringPreds <- cbind(row = 1:nrow(test.data), id = test.data$id, stringPreds)

    # reshape for output
    to_shape <- melt(stringPreds, id.vars = c('id', 'row'))
    to_shape <- arrange(to_shape,row,variable)
    to_shape <- to_shape[c('id', 'value')]
    names(to_shape) <- c('id', 'country')

    to_shape
}
