read_times <- function(filename) {
  dir <- "~/git/rp_test/timing/results/times_mnist/"
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
