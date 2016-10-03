
#
#
# Tadeu: Não rode esse programa no seu laptop! Não tem memória suficiente !!!!!!!!!!!!!!!!!!!!!!
#
#
#
#
#


rm(list=ls(all=TRUE))

require(Matrix)

# read response data
y = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/url_y.rds" )
X = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/url_Xt.rds" )

n = length(y)

# splitting in training and test sets 

# training set
n_training = 1.5*(10^(6))
X_training = X[, 1:n_training]
save(X_training, file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/X_training.Rdata")

# test set 1
n_test_1 = 0.5*(10^(6))
X_test_1 = X[, (n_training + 1):(n_training + n_test_1)]
save(X_test_1, file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/X_test_1.Rdata")

# test set 2
n_test_2 = n - n_training - n_test_1
X_test_2 = X[, (n_training + n_test_1 + 1):n ]
save(X_test_2, file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/X_test_2.Rdata")


y_training = y[1:n_training]
y_test_1 = y[ (n_training + 1):(n_training + n_test_1)]
y_test_2 = y[ (n_training + n_test_1 + 1):n ]
