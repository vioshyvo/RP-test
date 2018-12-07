read_votes_file <- function(filename) {
  dir <- "~/git/rp_test/timing/results/"
  fn <- paste0(dir,filename)
  readLines(fn)
}

read_votes <- function(lines, n_test = 100) {
  k <- as.integer(strsplit(lines[1], " ")[[1]][1])
  split_lines <- strsplit(lines, " ")

  votes <- list()
  line <- lines[1]
  c <- 1
  par <- matrix(as.integer(strsplit(line, ' ')[[1]]), nrow = 1)
  while(!is.na(line)) {
    c <- c + 1
    votes_tmp <- matrix(integer(n_test * k), nrow = n_test, ncol = k)
    for(i in 1:n_test) {
      votes_tmp[i, ] <- as.integer(split_lines[[c]])
      c <- c + 1
    }
    votes[[length(votes) + 1]] <- votes_tmp
    line <- lines[c]
    if(!is.na(line)) par <- rbind(par, as.integer(strsplit(line, ' ')[[1]]))
  }
  parameters <- as.data.frame(par)
  names(parameters) <- c('k', 'n_trees', 'depth', 'v')
  
  list(parameters = parameters, votes = votes) 
}