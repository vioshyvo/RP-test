source("~/git/rp_test/timing/tools/read_res.R")

res <- read_times("mrpt_times_new20")
t <- tail(res, 20)

###################################################
# Plot query time vs. recall 

k100 <- res[res$k == 100, ]
# plot(k100$recall, log10(pmax(0.0001, k100$est..query.time)), type = 'l', col = 'blue')
# lines(k100$recall, log10(k100$query.time), type = 'l', col = 'red')
# 
# plot(k100$recall, k100$est..query.time, type = 'l', col = 'blue', ylim = c(-0.05, 0.2))
# lines(k100$recall, k100$query.time, type = 'l', col = 'red')
# abline(h = 0)

###################################################
# Plot exact search time vs. |S| 

plot(k100$n_elected, k100$exact.time, pch = 20, col = 'blue', cex = .6)

beta <- fit_theil_sen(k100$n_elected, k100$exact.time)
grid <- 0:10000
lines(grid, beta[1] + beta[2] * grid, lwd = 2, col = 'blue')

###################################################
# Plot exact search time vs. |S|  for timed code

ex <- read_exact("exact_times6")

for(k in c(1, 10, 100)) {
  exk100 <- ex[ex$k == k, ]
  points(exk100$n_elected, exk100$exact_time * 100, pch = 20, col = k + 1, cex = .6)
  
  beta <- fit_theil_sen(exk100$n_elected, exk100$exact_time * 100)
  grid <- 0:10000
  lines(grid, beta[1] + beta[2] * grid, lwd = 2, col = k + 1)
}


###################################################
# Plot estimated projection time vs true projection time

plot(res$est..proj..time, res$projection.time, pch = 20, col = 'blue', cex = .6,
     xlab = 'Estimated projection time', ylab = 'True projection time', main = 'Projection time')
grid <- seq(0,1,by=.01)
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')

###################################################
# Plot estimated voting time vs true voting time

plot(res$est..voting.time, res$voting.time, pch = 20, col = 'blue', cex = .6,
     xlab = 'Estimated voting time', ylab = 'True voting time', main = 'Voting time')
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')

###################################################
# Plot estimated exact time vs true exact time

plot(res$est..exact.time, res$exact.time, pch = 20, col = 'blue', cex = .6,
     xlab = 'Estimated exact time', ylab = 'True exact time', main = 'Exact time')
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')

###################################################
# Plot estimated query time vs true query time

plot(res$est..query.time, res$query.time, pch = 20, col = 'blue', cex = .6,
     xlab = 'Estimated query time', ylab = 'True query time', main = 'Query time')
lines(grid, grid, type = 'l', lty = 2, lwd = 2, col = 'red')



###################################################
# Plot recall vs. |S|

plot(k100$n_elected, k100$recall, pch = 20, col = k100$depth - 4, cex = .6)
legend("bottomright", legend = 5:9, title = "depth", col = (5:9) - 4, bty = 'n', lty = 1, lwd = 2)
