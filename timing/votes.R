setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("tools/read_votes.R")

n_test <- 1000
dir <- 'times_mnist1000'

votes_list <- read_votes(read_votes_file(paste(dir, 'votes_100', sep = '/')), n_test)
top_votes <- read_votes(read_votes_file(paste(dir, 'top_votes_100', sep = '/')), n_test)$votes
votes <- votes_list$votes
par <- votes_list$parameters

i <- 10
v <- votes[[i]]
t <- top_votes[[i]]
par[i, ]
summary(v[, 1])
# v[ ,1]

vmean <- colMeans(v)
barplot(vmean, col = 'skyblue', main = 'mean votes for 100-nn')

v[1:30, 1:10]
t[1:30, 1:10]
