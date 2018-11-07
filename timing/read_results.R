source("~/git/rp_test/timing/tools/read_res.R")

res <- read_times("mrpt_times_new11")
t <- tail(res, 20)

k100 <- res[res$k == 100, ]
plot(k100$recall, log10(pmax(0.0001, k100$est..query.time)), type = 'l', col = 'blue')
lines(k100$recall, log10(k100$query.time), type = 'l', col = 'red')

###################################################
# Plot exact search time vs. |S| 

plot(k100$n_elected, k100$exact.time, pch = 20, col = 'red', cex = .6, ylim = c(0, 0.30))

beta <- fit_theil_sen(k100$n_elected, k100$exact.time)
grid <- 0:5000
lines(grid, beta[1] + beta[2] * grid, lwd = 2, col = 'blue')

plot(k100$n_elected, k100$recall, pch = 20, col = k100$depth - 4, cex = .6)
legend("bottomright", legend = 5:9, title = "depth", col = (5:9) - 4, bty = 'n', lty = 1, lwd = 2)
