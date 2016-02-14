##########################################################################
# Load and preprocess the data
##########################################################################
library(dummies)
library(dplyr)
library(reshape2)
library(caret)
# load data
setwd('~/leave_academia/kaggle/airbnb')
train.data <- read.csv(file = 'train_users_2.csv', header = T, stringsAsFactors = F)
test.data <- read.csv(file = 'test_users.csv', header = T, stringsAsFactors = F)

# response and fake responses
train.data <- data.frame(id = train.data[, 1], response = train.data[, dim(train.data)[2]],
                         train.data[, -c(1,dim(train.data)[2])])
test.data <- data.frame(id = test.data[, 1], response = NA, test.data[, -1])

# bind together
all.data <- rbind(train.data, test.data)
rm(train.data, test.data)

# remove date first booking and convert the date account created to (year, month, day)

all.data['date_first_booking'] <- NULL
actmp <- all.data$date_account_created
actmp <- as.POSIXlt(actmp)
all.data$date_account_created_year = actmp$year + 1900
all.data$date_account_created_month = actmp$mon
all.data$date_account_created_day = actmp$mday
all.data$date_account_created <- NULL

# helper functions for processing timestamp_first_active

extract_year <- function(x) {
    y <- strsplit(as.character(x),split = '')
    as.integer(paste(y[[1]][1:4],collapse=''))
}

all.data$timestamp_first_active_year = sapply(all.data$timestamp_first_active,extract_year)

extract_month <- function(x) {
    y <- strsplit(as.character(x),split = '')
    as.integer(paste(y[[1]][5:6],collapse=''))
}

all.data$timestamp_first_active_month = sapply(all.data$timestamp_first_active,extract_month)

extract_day <- function(x) {
    y <- strsplit(as.character(x),split = '')
    as.integer(paste(y[[1]][7:8],collapse=''))
}

all.data$timestamp_first_active_day = sapply(all.data$timestamp_first_active,extract_day)

all.data$timestamp_first_active <- NULL

# clean the age column
# 41% of ages are missing;
# some ages are absurd: 2014
# some people put their age as a date of birth or year
indx.age.year <- with(all.data, which(age > 1800 & age < 1995))
all.data[indx.age.year,'age'] = all.data[indx.age.year,
                                         'date_account_created_year']- all.data[indx.age.year,'age']
all.data[which(all.data$age >= 2000), 'age'] <- NA
all.data[which(all.data$age >= 90), 'age'] <- 95
all.data[which(all.data$age <= 15), 'age'] <- NA

# rescale age
indx = which(is.na(all.data$age))
rg.scaler <- preProcess(all.data[-indx, 'age', drop = F], method = c('range'))
all.data[-indx, 'age'] <- predict(rg.scaler, all.data[-indx, 'age', drop = F])

all.data[indx,'age']=-1

# clean up categorical variables to factor

all.data$signup_method = as.factor(all.data$signup_method)

all.data$language = factor(all.data$language)
all.data$affiliate_channel = factor(all.data$affiliate_channel)
all.data$affiliate_provider = factor(all.data$affiliate_provider)

all.data$first_affiliate_tracked[all.data$first_affiliate_tracked==''] = 'untracked'
all.data$first_affiliate_tracked = factor(all.data$first_affiliate_tracked)

all.data$first_browser = factor(all.data$first_browser)
all.data$signup_app = factor(all.data$signup_app)
all.data$first_device_type = factor(all.data$first_device_type)
all.data$gender <- factor(all.data$gender)

# just normalize signup flow
rg.scaler <- preProcess(all.data['signup_flow'], method = c('range'))
all.data['signup_flow'] <- predict(rg.scaler, all.data['signup_flow'])


# create target 
all.data$response <- factor(all.data$response)

##########################################################################
# Extract OS information from sessions 
##########################################################################

sessions.data <- read.csv(file = 'sessions.csv', header = T, stringsAsFactors = F)
all.sessions <- sessions.data
for(i in 2:5) {
    all.sessions[i] <- factor(all.sessions[[i]])
}

# Remove users with empty id
indx <- which(all.sessions['user_id'] == '')
all.sessions <- all.sessions[-indx,]

# Merge some devices
levels(all.sessions$device_type)[2] <- 'Android'
levels(all.sessions$device_type)[3] <- 'Android'

# Device used
gr2 <- all.sessions[c('user_id', 'device_type')]
gr2 <- group_by(gr2, user_id, device_type)
gr2 <- summarise(gr2, count = n())
indx <- which(gr2$device_type == '-unknown-')
gr2 <- gr2[-indx,]

# drop '-unknown-' from list of factors
gr2$device_type <- gr2$device_type[, drop = T]

# create # of uses vars
dlev = levels(gr2$device_type)
for(i in 1:length(dlev)) {
    gr2[paste('Uses', dlev[i], sep='_')] <- 0
    gr2[which(gr2$device_type == dlev[i]),
        paste('Uses', dlev[i], sep='_')] <- gr2[which(gr2$device_type == dlev[i]),
                                                'count']
}

# summarize dummy vars
gr2 <- gr2[,-c(2:3)]
gr2 <-  group_by(gr2,user_id)
gr2 <- summarise_each(gr2, funs(sum), matches('Uses'))

# cut used by 3rd quantiles
uses_stats <- apply(gr2[,-1],2,function(x) summary(x[which(x>0)]))
for(i in c(2:dim(gr2)[2])) {
    indxx <- which(gr2[,i] >= uses_stats[5, i-1])
    gr2[indxx, i] <- uses_stats[5, i-1]
}
#scale
for(i in c(2:dim(gr2)[2])) {
    rg.scaler <- preProcess(gr2[,i], method = c('range'))
    gr2[, i] <- predict(rg.scaler, gr2[, i])
}

os_used <- gr2
rm(gr2)

# merge with all.data
# dummy var signaling whether additional osinfo is available
indx.common <- which(all.data$id %in% os_used$user_id)
all.data$OsInfo <- 0
all.data[indx.common, 'OsInfo'] <- 1

os_empty <- data.frame(user_id = all.data[-indx.common, 'id'],
                       matrix(0, ncol = dim(os_used)[2]-1,
                              nrow = dim(all.data)[1] - length(indx.common)))
colnames(os_empty) <- colnames(os_used) # needed before rbind
os_used <- rbind(os_used, os_empty)
os_used <- os_used[match(all.data$id, os_used$user_id),] # reorder

##########################################################################
# Extract action info from sessions
##########################################################################

common.indx <- which(sessions.data$user_id %in% all.data$id)
sessions.data <- sessions.data[common.indx, -4] # remove os_info already extracted
action <- factor(sessions.data$action)
action_type <- factor(sessions.data$action_type)

action.num <- c(6, 7, 21, 22, 23, 39, 47, 53, 58, 62, 63, 65,
                69, 70, 71, 74, 91, 92, 101, 132, 133, 134, 146,
                147, 148, 166, 251, 253, 254, 261, 262, 266,
                268, 274, 279, 280, 298, 305, 306, 307, 308, 316,
                319, 324, 325, 326,  331, 332, 333, 336, 341)

action.nms <- levels(action)[action.num]
action.sel <- which(sessions.data$action %in% action.nms) 

# some secs elapsed are NAs

sessions.data <- with(sessions.data, data.frame(id = user_id[action.sel],
                      action = factor(action[action.sel]), # drop unused levels
                      action_type = action_type[action.sel],
                      secs_elapsed = secs_elapsed[action.sel]))
action_type <- factor(sessions.data$action_type) # action_type seems useless
sessions.data$action_type <- NULL
actype <- sessions.data

# actype to wide data-frame format
dlev = levels(actype$action)
for(i in 1:length(dlev)) {
    actype[paste('Action', dlev[i], sep='_')] <- 0
    actype[which(actype$action == dlev[i]),
       paste('Action', dlev[i], sep='_')] <- actype[which(actype$action == dlev[i]),
                                                    'secs_elapsed']
}
actype$secs_elapsed <- NULL
actype$action <- NULL
actstats <- sapply(actype[,-c(1:2)], function(x) summary(x[x>0], na.rm=T))

# impute NAs by median
for(i in 3:dim(actype)[2]) {
    indx <- actype[, i] > 0
    actype[is.na(actype[, i]),i] <- median(actype[indx, i], na.rm = T)
}

# summarize dummy vars
actype <-  group_by(actype,id)
actype <- summarise_each(actype, funs(sum), matches('Action_'))

actstats <- sapply(actype[,-1], function(x) summary(x[x>0], na.rm=T))

# scaling by type
var.log.scaler <- c(2, 4, 5, 6, 9, 10, 11, 14, 15, 16, 17,
                    18, 20, 22, 23, 24, 28, 29, 33, 34, 36,
                    38, 42, 45, 47, 49, 50, 52)
var.scaler <- c(8, 25, 12, 48, 51)
var.binary <- c(3, 21, 31, 35, 37, 39, 40, 41, 46)
var.cut.scaler <- c(13, 19, 26, 27, 30, 32, 43, 44)

# log(1+ ) + rangescaler
actype[, var.log.scaler] <- log(1 + actype[, var.log.scaler])
range.log.scaler <- preProcess(actype[, var.log.scaler], method = c('range'))
actype[, var.log.scaler] <- predict(range.log.scaler, actype[, var.log.scaler])

# rangescaler
range.scaler <- preProcess(actype[, var.scaler], method = c('range'))
actype[, var.scaler] <- predict(range.scaler, actype[, var.scaler])

# binary levels to 0,...
for(i in var.binary) {
     actype[[i]] <- factor(actype[[i]])
     levels(actype[[i]]) <- c(1:length(levels(actype[[i]])))
     actype[[i]] <- as.numeric(actype[[i]]) - 1
}

# normalize at 3rd quantile and scale
for(i in var.cut.scaler) {
    indx <- actype[[i]] > 0
    q3 <- sapply(actype[indx, i], quantile, probs = 0.75)
    actype[[i]] <- actype[[i]]/q3
}

range.cut.scaler <- preProcess(actype[var.cut.scaler], method = c('range'))
actype[var.cut.scaler] <- predict(range.cut.scaler, actype[var.cut.scaler])

# prepare to bind
all.data$ActionInfo <- 0
common.indx <- which(all.data$id %in% actype$id)
all.data[common.indx, 'ActionInfo'] <- 1


actype_empty <- data.frame(id = all.data[-common.indx, 'id'],
                       matrix(0, ncol = dim(actype)[2]-1,
                              nrow = dim(all.data)[1] - length(common.indx)))
colnames(actype_empty) <- colnames(actype) # needed before rbind
actype <- rbind(actype, actype_empty)
actype <- actype[match(all.data$id, actype$id),] # reorder


#bind
all.data <- data.frame(all.data, os_used[, -1], actype[, -1])


save(all.data, file = 'cleaned_data.RData')


