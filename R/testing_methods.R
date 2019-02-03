
require(xgboost)
library(data.table)
source("R/processing.R")

# readAndSplit(path = "data/train.csv",
#              target_name = "event",
#              name = "train")

train = fread("data/train.csv")

set.seed(0127)
train_ind = sample.int(n = nrow(train), size = .6 * nrow(train),
                       replace = FALSE)

train_small = train[train_ind]

train_small[, experiment := NULL]

train_small[, time_bin := cut(time, breaks = c(0, 90, 180, 270, 365),
                              labels = FALSE)]



# coerce proper variables to categorical
train_small = coerceClass(train_small, c("crew", "seat"),
                  fun = as.factor)

# get the numeric columns (exluding time)
cols = names(train_small)[vapply(train_small, is.numeric,
                         FUN.VALUE = logical(1))]

# remove time 
cols = cols[cols != "time"]

# calculate vector of lower and upper bounds for each numeric column
lb = vapply(train_small[, ..cols], function(x) mean(x) - 6*sd(x),
            FUN.VALUE = numeric(1))
ub = vapply(train_small[, ..cols], function(x) mean(x) + 6*sd(x),
            FUN.VALUE = numeric(1))

# get index of observations to remove
ind = apply(train_small[, ..cols], 1, function(x){
  return(!any(x < lb | x > ub))
})

# remove rows containing outliers
train_small = train_small[ind]
y_train_small = y_train_small[ind]

# now put time back into cols
cols = c(cols, "time")

scaled = scale(train_small[, ..cols])

km = kmeans(scaled, centers = 4)

# run k-means clustering on data

# run LDA on numeric columns
# lda = MASS::lda(train_small[, ..cols],
#                 grouping = as.factor(y$event))

# fit lda to training train_smalla
preds = predict(lda, train_small[, ..cols])

# assign predictions to train_small
train_small = cbind(train_small, preds$posterior)

# code variables to be numeric
char_vars = names(train_small)[sapply(train_small, is.factor)]
train_small = train_small[, (char_vars) := lapply(.SD, multiCodeVars),
          .SDcols = char_vars]
