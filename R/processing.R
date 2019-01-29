
# The goal of this function is to read in a file and split into two data.tables
# One with the independent variables and one with the dependent variable and id.

readAndSplit = function(path, target_name, name = "train"){
  
  file = data.table::fread(path)
  
  # if it is the test set, keep the id column in the response portion
  if (name == "test"){
    assign("x_test",
           file[, !'id', with = FALSE],
           envir = parent.frame())
    assign("y_test",
           file[, "id", with = FALSE],
           envir = parent.frame())
  }
  
  # if it isn't the test set, no need to worry about the id column (not there)
  assign(paste("x", name, sep = "_"),
         file[, setdiff(names(file), target_name), with = FALSE],
         envir = parent.frame())
  assign(paste("y", name, sep = "_"),
         file[, target_name, with = FALSE, drop = FALSE],
         envir = parent.frame())
}


# given a data.table and column names, coerce to desired classes

coerceClass = function(data, columns, fun = "as.factor", ...){
  
  stopifnot(data.table::is.data.table(data))
  
  data = data.table::copy(data)
  func = match.fun(fun)
  
  for(name in columns){
    data.table::set(
      data, j = name, value = func(data[[name]])
      )
  }
  
  return(data)

}

multiCodeVars = function(fac){
  
  stopifnot(is.factor(fac) || is.character(fac))
  
  fac = as.factor(fac)
  levs = levels(fac)
  vec = integer(length(fac))
  
  for (i in seq_along(levs)) {
    vec[which(fac == levs[i])] = i - 1
  }
  
  return(vec)
}


# format multiclass predictions from xgboost model

formatXgbProbs = function(xgbMod, newdat, classes = LETTERS[1:4]){
  
  # error checking
  stopifnot(inherits(newdat, "xgb.DMatrix"))
  
  # get predictions from model on new data
  probs = predict(xgbMod, newdat)
  
  # create a matrix of predictions
  prob_mat = matrix(probs, ncol = length(classes), byrow = TRUE)
  colnames(prob_mat) = classes
  
  return(prob_mat)
}


processTrainData = function(data, y){
  
  stopifnot(data.table::is.data.table(data))
  stopifnot(nrow(data) == nrow(y))
  
  dat = data.table::copy(data)
  
  dat$experiment = NULL
  
  # coerce proper variables to categorical
  dat = coerceClass(dat, c("crew", "seat"),
                        fun = as.factor)
  
  # get the numeric columns (exluding time)
  cols = names(dat)[vapply(dat, is.numeric,
                               FUN.VALUE = logical(1))]
  
  # remove time 
  cols = cols[cols != "time"]
  
  # calculate vector of lower and upper bounds for each numeric column
  lb = vapply(dat[, ..cols], function(x) mean(x) - 6*sd(x),
              FUN.VALUE = numeric(1))
  ub = vapply(dat[, ..cols], function(x) mean(x) + 6*sd(x),
              FUN.VALUE = numeric(1))
  
  # get index of observations to remove
  ind = apply(dat[, ..cols], 1, function(x){
    return(!any(x < lb | x > ub))
  })
  
  # remove rows containing outliers
  dat = dat[ind]
  y = y[ind]
  
  # now put time back into cols
  cols = c(cols, "time")
  
  # run LDA on numeric columns
  lda = MASS::lda(dat[, ..cols],
                  grouping = as.factor(y$event))
  
  # fit lda to training data
  preds = predict(lda, dat[, ..cols])
  
  # assign predictions to dat
  dat = cbind(dat, preds$posterior)
  
  # code variables to be numeric
  char_vars = names(dat)[sapply(dat, is.factor)]
  dat = dat[, (char_vars) := lapply(.SD, multiCodeVars),
                            .SDcols = char_vars]
  
  
  # return processed data and lda model
  return(
    list(x_train = dat,
         y_train = y,
         lda_mod = lda,
         lda_cols = cols)
  )
  
}

processTestData = function(data, lda_model, lda_cols){
  
  dat = data.table::copy(data)
  data.table::setDT(dat)
  
  # get rid of experiment column
  dat$experiment = NULL
  
  # coerce classes to factor to code them properly
  dat = coerceClass(dat, c("crew", "seat"), fun = as.factor)
  
  # make predictions using lda
  preds = predict(lda_model, dat[, ..lda_cols])$posterior
  
  # bind predictions to test data
  dat = cbind(dat, preds)
  
  # code nominal variables to start at 0
  char_vars = names(dat)[sapply(dat, is.factor)]
  dat = dat[, (char_vars) := lapply(.SD, multiCodeVars),
              .SDcols = char_vars]
  
  return(dat)
}














