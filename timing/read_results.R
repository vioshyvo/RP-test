source("~/git/rp_test/timing/tools/read_res.R")

res <- read_times("mrpt_times_new6")
t <- tail(res, 20)

k100 <- res[res$k == 100, ]
plot(k100$recall, log10(pmax(0.0001, k100$est..query.time)), type = 'l', col = 'blue')
lines(k100$recall, log10(k100$query.time), type = 'l', col = 'red')
