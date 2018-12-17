rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source('tools/read_res.R')
source('tools/plot.R')

dir <- 'gist'
res1 <- read_normal_times(filename=file.path(dir, 'mrpt_total4'))
res2 <- read_normal_times2(filename=file.path(dir, 'mrpt_total4_size'))
res3 <- read_normal_times3(filename=file.path(dir, 'mrpt_total4_size2'))
res4 <- read_normal_times3(filename=file.path(dir, 'mrpt_total4_size3'))

k <- 100
n_trees <- 1000
rr1 <- res1[res1$k == k & res1$n_trees == n_trees, ]
rr2 <- res2[res2$k == k & res2$n_trees == n_trees, ]
rr3 <- res3[res3$k == k & res3$n_trees == n_trees, ]
rr4 <- res4[res4$k == k & res4$n_trees == n_trees, ]

r1 <- pareto_frontier(res1[res1$k == k, ])
r2 <- pareto_frontier(res2[res2$k == k, ])
r3 <- pareto_frontier(res3[res3$k == k, ])
r4 <- pareto_frontier(res4[res4$k == k, ])
rlist = list(mrpt = r1, mrpt_size = r2, mrpt_size2 = r3, mrpt_size3 = r4)

plot_times(rlist, 'query_time', paste0(dir, ', k = ', k))
plot_components(r1, main = paste0(dir, ', k = ', k), ylim = c(-5, -1))
plot_components(r2, c('projection_time', 'voting_time', 'exact_time', 'sorting_time'),
                main = paste0(dir, ', k = ', k), ylim = c(-5, -1))
plot_components(r3, c('projection_time', 'voting_time', 'exact_time',
                      'choosing_time'), main = paste0(dir, ', k = ', k), ylim = c(-5, -1))
pdf('fig/all_components.pdf', width = 8, height = 6)
plot_components(r4, c('projection_time', 'voting_time', 'exact_time',
                      'choosing_time'), main = paste0(dir, ', k = ', k), ylim = c(-4, -1))
dev.off()


pdf('fig/all_components.pdf', width = 8, height = 8)
par(mfrow = c(2,2), mar = c(4,4,4,1))
plot_components(m4, c('projection_time', 'voting_time', 'exact_time',
                      'choosing_time'), main = paste0('mnist', ', k = ', k), ylim = c(-5, -2.5))
plot_components(g4, c('projection_time', 'voting_time', 'exact_time',
                      'choosing_time'), main = paste0('gist', ', k = ', k), ylim = c(-4, -1))
plot_components(s4, c('projection_time', 'voting_time', 'exact_time',
                      'choosing_time'), main = paste0('sift', ', k = ', k), ylim = c(-4.5, -2))
dev.off()
