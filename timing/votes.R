rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("tools/read_votes.R")

n_test <- 1000
dir <- 'times_mnist1000'

votes_list <- read_votes(read_votes_file(paste(dir, 'votes_100', sep = '/')), n_test)
top_votes <- read_votes(read_votes_file(paste(dir, 'top_votes_100', sep = '/')), n_test)$votes
votes <- votes_list$votes
par <- votes_list$parameters
cs_sizes <- votes_list$cs_sizes

i <- 8
v <- votes[[i]]
t <- top_votes[[i]]
s <- cs_sizes[[i]]
par[i, ]
summary(v[, 1])
summary(t[, 1])
# v[ ,1]

vmean <- colMeans(v)
barplot(vmean, col = 'skyblue', main = 'mean votes for 100-nn')

cbind(v[1:20, 1:10], s[1:20])
cbind(t[1:20, 1:10], s[1:20])

######################################################
# regression model for vote count of 1-nn

target_recall <- .9
df <- data.frame(x = t[ ,1], y = v[ ,1])
m <- glm(y ~ x, family="poisson", data = df)


pred <- predict.glm(m, type = 'response')
df$q <- qpois(1 - target_recall, pred)
df$s <- df$y >= df$q

mean(df$s)
