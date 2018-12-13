rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("tools/read_res.R")

dir <- 'mnist'
res <- read_normal_times(filename=file.path(dir, 'mrpt_mmm9'))
