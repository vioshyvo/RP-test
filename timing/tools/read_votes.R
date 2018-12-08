read_votes_file <- function(filename) {
  dir <- "~/git/rp_test/timing/results/"
  fn <- paste0(dir,filename)
  readLines(fn)
}

read_votes <- function(lines, n_test = 100) {
  k <- as.integer(strsplit(lines[1], " ")[[1]][1])
  split_lines <- strsplit(lines, " ")

  votes <- cs_sizes <- list()
  line <- lines[1]
  c <- 1
  par <- matrix(as.integer(strsplit(line, ' ')[[1]]), nrow = 1)
  while(!is.na(line)) {
    c <- c + 1
    votes_tmp <- matrix(integer(n_test * k), nrow = n_test, ncol = k)
    cs_sizes_tmp <- integer(n_test)
    for(i in 1:n_test) {
      split <- split_lines[[c]]
      votes_tmp[i, ] <- as.integer(split[-length(split)])
      cs_sizes_tmp[i] <- as.integer(split[length(split)])
      c <- c + 1
    }
    crnt_idx <- length(votes) + 1
    votes[[crnt_idx]] <- votes_tmp
    cs_sizes[[crnt_idx]] <- cs_sizes_tmp
    line <- lines[c]
    if(!is.na(line)) par <- rbind(par, as.integer(strsplit(line, ' ')[[1]]))
  }
  parameters <- as.data.frame(par)
  names(parameters) <- c('k', 'n_trees', 'depth', 'v')
  
  list(parameters = parameters, votes = votes, cs_sizes = cs_sizes) 
}