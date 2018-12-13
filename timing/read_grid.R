rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('tools/read_res.R')
source('tools/plot.R')

dir <- 'mnist'
res1 <- read_normal_times(filename=file.path(dir, 'mrpt_total2'))
res2 <- read_normal_times2(filename=file.path(dir, 'mrpt_total2_size'))
res3 <- read_normal_times3(filename=file.path(dir, 'mrpt_total2_size2'))
res4 <- read_normal_times3(filename=file.path(dir, 'mrpt_total2_size3'))

k <- 100
r1 <- pareto_frontier(res1[res1$k == k, ])
r2 <- pareto_frontier(res2[res2$k == k, ])
r3 <- pareto_frontier(res3[res3$k == k, ])
r4 <- pareto_frontier(res4[res4$k == k, ])
rlist = list(mrpt = r1, mrpt_size = r2, mrpt_size2 = r3, mrpt_size3 = r4)

plot_times(rlist, 'query_time', 'mnist, k = 100')
