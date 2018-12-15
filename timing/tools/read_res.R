read_times <- function(filename) {
  dir <- "~/git/rp_test/timing/results/"
  res <- read.table(paste0(dir, filename), sep = ' ', header = FALSE, 
                    strip.white = TRUE, col.names = c(
                      "k",
                      "n_trees",
                      "depth",
                      "sparsity",
                      "votes",
                      "recall",
                      "sd. recall",
                      "query time",
                      "sd. query time",
                      "indexing time",
                      "recall 2",
                      "est. query time",
                      "est. proj. time",
                      "est. voting time",
                      "est. exact time",
                      "comp. query time",
                      "projection time",
                      "voting time",
                      "exact time",
                      "n_elected",
                      "Na"
                    ))
  
  res[, 1:20] 
} 

read_exact <- function(filename) {
  dir <- "~/git/rp_test/timing/results/times_mnist/"
  res <- read.table(paste0(dir, filename), sep = ' ', header = FALSE, 
                    strip.white = TRUE, col.names = c("k", "n_elected", "exact_time"))
  res
}
                      


fit_theil_sen <- function(x, y) {
  n <- length(x)
  c <- 0
  slopes <- numeric(n^2)
  for(i in 1:n) 
    for(j in 1:n) {
      c <- c + 1
      if(i != j) slopes[c] <- (y[j] - y[i]) / (x[j] - x[i])
    }
  
  slopes <- slopes[slopes > 0]
  
  beta1 <- median(slopes)
  beta0 <- median(y - beta1 * x)
  
  c(beta0, beta1)
}


read_normal_times <- function(filename) {
  dir <- "~/git/rp_test/timing/results"
  res <- read.table(file.path(dir, filename), sep = ' ', header = FALSE, 
                    strip.white = TRUE, col.names = c(
                      "k",
                      "n_trees",
                      "depth",
                      "sparsity",
                      "votes",
                      "recall",
                      "sd_recall",
                      "query_time",
                      "sd_query_time",
                      "projection_time",
                      "voting_time",
                      "exact_time",
                      "build_time",
                      "n_elected",
                      "na"
                    ))
  
  res[, ncol(res)] <- NULL
  res
} 

read_normal_times2 <- function(filename) {
  dir <- "~/git/rp_test/timing/results"
  res <- read.table(file.path(dir, filename), sep = ' ', header = FALSE, 
                    strip.white = TRUE, col.names = c(
                      "k",
                      "n_trees",
                      "depth",
                      "sparsity",
                      "votes",
                      "recall",
                      "sd_recall",
                      "query_time",
                      "sd_query_time",
                      "projection_time",
                      "voting_time",
                      "exact_time",
                      "build_time",
                      "n_elected",
                      "sorting_time",
                      "na"
                    ))
  
  res[, ncol(res)] <- NULL
  res
} 

read_normal_times3 <- function(filename) {
  dir <- "~/git/rp_test/timing/results"
  res <- read.table(file.path(dir, filename), sep = ' ', header = FALSE, 
                    strip.white = TRUE, col.names = c(
                      "k",
                      "n_trees",
                      "depth",
                      "sparsity",
                      "votes",
                      "recall",
                      "sd_recall",
                      "query_time",
                      "sd_query_time",
                      "projection_time",
                      "voting_time",
                      "exact_time",
                      "build_time",
                      "n_elected",
                      "sorting_time",
                      "choosing_time",
                      "na"
                    ))
  
  res[, ncol(res)] <- NULL
  res
} 

pareto_frontier <- function(df) {
  p <- rPref::low(query_time) * rPref::high(recall)
  res <- rPref::psel(df, p)
  res[order(res$recall), ]
}



read_exact <- function(filename) {
  dir <- "~/git/rp_test/timing/results/times_mnist/"
  res <- read.table(paste0(dir, filename), sep = ' ', header = FALSE, 
                    strip.white = TRUE, col.names = c("k", "n_elected", "exact_time"))
  res
}
