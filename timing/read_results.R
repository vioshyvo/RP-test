source("~/git/rp_test/timing/tools/read_res.R")

res <- read_times(filename="times_mnist1000/mrpt_times_test20")
t <- tail(res, 20)

###################################################
# Plot query time (true & estimated) vs. recall 

k <- 100
k100 <- res[res$k == k, ]
# log scale
plot(k100$recall, log10(k100$query.time), type = 'l', col = 'red', lwd = 2, 
     xlab = 'recall', ylab = 'log(query time)', main = paste('k =', k))
lines(k100$recall, log10(pmax(0.0001, k100$est..query.time)), type = 'l', col = 'blue', lwd = 2)
legend('topleft', bty = 'n', legend = c('true', 'estimated'), col = c('red', 'blue'), lwd = 2)


# normal scale
plot(k100$recall, k100$est..query.time, type = 'l', col = 'blue', ylim = c(0, 1.50), lwd = 2,
     xlab = 'recall', ylab = 'query time (s. / 100 queries)', main = paste('k =', k))
lines(k100$recall, k100$query.time, type = 'l', col = 'red', lwd = 2)
legend('topleft', bty = 'n', legend = c('true', 'estimated'), col = c('red', 'blue'), lwd = 2)


###################################################
# Plot exact search time vs. |S| 

plot(k100$n_elected, k100$exact.time, pch = 20, col = 'blue', cex = .6)

beta <- fit_theil_sen(k100$n_elected, k100$exact.time)
grid <- 0:10000
lines(grid, beta[1] + beta[2] * grid, lwd = 2, col = 'blue')


###################################################
# Plot estimated projection time vs true projection time

plot(res$est..proj..time, res$projection.time, pch = 20, col = 'blue', cex = .6,
     xlab = 'Estimated projection time', ylab = 'True projection time', main = 'Projection time')
grid <- seq(0,10,by=.01)
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')

###################################################
# Plot estimated voting time vs true voting time

plot(res$est..voting.time, res$voting.time, pch = 20, col = res$votes, cex = .6,
     xlab = 'Estimated voting time', ylab = 'True voting time', main = 'Voting time')
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')


depth <- 4
depth <- depth + 1
plot(res$est..voting.time[res$depth == depth], res$voting.time[res$depth == depth], pch = 20, col = res$votes[res$depth == depth], cex = .6,
     xlab = 'Estimated voting time', ylab = 'True voting time', 
     main = paste('Voting time, depth =', depth), xlim = c(0, 0.12), ylim = c(0, 0.12) )
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')

###################################################
# Plot estimated exact time vs true exact time

pdf(paste0('fig/mnist_times_k', k, '_nsim100.pdf'), width=8, height=6, paper='special') 
plot(res$est..exact.time, res$exact.time, pch = 20, col = 'blue', cex = .6,
     xlab = 'Estimated exact time', ylab = 'True exact time', main = 'Exact time')
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')
dev.off()

###################################################
# Plot estimated query time vs true query time

pdf(paste0('fig/mnist_qtimes_k', k, '_nsim100.pdf'), width=8, height=6, paper='special') 
plot(res$est..query.time, res$query.time, pch = 20, col = 'blue', cex = .6,
     xlab = 'Estimated query time', ylab = 'True query time', main = 'Query time')
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')
dev.off()

###################################################
# Plot recall vs. |S|

plot(k100$n_elected, k100$recall, pch = 20, col = k100$depth - 4, cex = .6)
legend("bottomright", legend = 5:9, title = "depth", col = (5:9) - 4, bty = 'n', lty = 1, lwd = 2)

##################################################################
# Plot procection, voting, and exact search times vs. recall 

k <- 100
k100 <- res[res$k == k, ]

ylim <- switch(as.character(k), '1' = c(-4, -1.5), '10' = c(-4, -1), '100' = c(-4, -0.5))

# log scale for k = 100 ylim = c(-4, -0.5), k = 10 ylim = c(-4, -1), k = 1 ylin = c(-4,-1.5)
# pdf(paste0('fig/mnist_times_median_log_k', k, '.pdf'), width=8, height=6, paper='special') 
plot(k100$recall, log10(k100$query.time), type = 'l', col = 'red', lwd = 2, 
     xlab = 'recall', ylab = 'log(query time)', main = paste('k =', k), ylim = ylim, bty = 'n')
lines(k100$recall, log10(k100$projection.time), type = 'l', col = 'green', lwd = 2)
lines(k100$recall, log10(k100$voting.time), type = 'l', col = 'blue', lwd = 2)
lines(k100$recall, log10(k100$exact.time), type = 'l', col = 'purple', lwd = 2)

legend('topleft', bty = 'n', legend = c('query', 'projection', 'voting', 'exact'),
       col = c('red', 'green', 'blue', 'purple'), lwd = 2)
# dev.off()

# normal scale 
k <- 10
k100 <- res[res$k == k, ]
ylim <- switch(as.character(k), '1' = c(0, 0.015), '10' = c(0, 0.035), '100' = c(0, 0.07))

# pdf(paste0('fig/mnist_times_median_k', k, '.pdf'), width=8, height=6, paper='special') 
plot(k100$recall, k100$query.time, type = 'l', col = 'red', lwd = 2, ylim = ylim,
     xlab = 'recall', ylab = 'log(query time)', main = paste('k =', k))
lines(k100$recall, k100$projection.time, type = 'l', col = 'green', lwd = 2)
lines(k100$recall, k100$voting.time, type = 'l', col = 'blue', lwd = 2)
lines(k100$recall, k100$exact.time, type = 'l', col = 'purple', lwd = 2)

legend('topleft', bty = 'n', legend = c('query', 'projection', 'voting', 'exact'),
       col = c('red', 'green', 'blue', 'purple'), lwd = 2)
# dev.off()
