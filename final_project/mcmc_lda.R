library(lda)
require("reshape2")
data = read.table("docword.nips.txt", skip = 3)
data = as.matrix(data)
colnames(data) = NULL
data[,2] = data[,2] - 1
n = 1500

data <- apply (data, c (1, 2), function (x) {
  (as.integer(x))
})

ind = data[,1]

doc = list()

for ( i in 1 : n){

  doc[[i]] = t(as.matrix(data[which(ind == i), 2:3]))

}




K <- 20 ## Num clusters

tic = proc.time()
result <- lda.collapsed.gibbs.sampler(doc, K,  vocab, 1000,  0.1, 0.1, compute.log.likelihood=TRUE)
proc.time() - tic
# top words in the cluster
top.words <- top.topic.words(result$topics, 10, by.score=TRUE)
top.words
