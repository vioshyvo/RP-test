rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('tools/read_res.R')
source('tools/plot.R')

dir <- 'mnist'
res1 <- read_normal_times(filename=file.path(dir, 'mrpt_total2'))
res2 <- read_normal_times2(filename=file.path(dir, 'mrpt_total2_size'))
res3 <- read_normal_times3(filename=file.path(dir, 'mrpt_total2_size2'))
res4 <- read_normal_times3(filename=file.path(dir, 'mrpt_total2_size3'))

rr1 <- res1[res1$k == k, ]
rr2 <- res1[res2$k == k, ]
rr3 <- res1[res3$k == k, ]
rr4 <- res1[res4$k == k, ]

k <- 100
r1 <- pareto_frontier(res1[res1$k == k, ])
r2 <- pareto_frontier(res2[res2$k == k, ])
r3 <- pareto_frontier(res3[res3$k == k, ])
r4 <- pareto_frontier(res4[res4$k == k, ])
rlist = list(mrpt = r1, mrpt_size = r2, mrpt_size2 = r3, mrpt_size3 = r4)

plot_times(rlist, 'query_time', paste0('mnist, k = ', k))
plot_components(r1, main = paste0('mnist, k = ', k))
plot_components(r2, c('projection_time', 'voting_time', 'exact_time', 'sorting_time'),
                main = paste0('mnist, k = ', k))
plot_components(r3, c('projection_time', 'voting_time', 'exact_time', 'sorting_time',
                      'choosing_time'), main = paste0('mnist, k = ', k))
plot_components(r4, c('projection_time', 'voting_time', 'exact_time', 'sorting_time',
                      'choosing_time'), main = paste0('mnist, k = ', k))

