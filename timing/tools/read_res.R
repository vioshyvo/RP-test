read_times <- function(filename) {
  dir <- "~/git/rp_test/timing/results/times_mnist/"
  res <- read_table(paste0(dir, filename), sep = ' ', header = FALSE, 
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
                      "Na"
                    ))
  
  res[, 1:19] 
} 
