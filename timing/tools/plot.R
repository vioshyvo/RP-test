plot_times <- function(rlist, colname, main = '') {
  r1 <- rlist[[1]]
  plot(r1$recall, log10(r1[[colname]]), col = 2, pch = 15, type = 'b', lwd = 2, cex = .85, 
       xlim = c(0.5, 1), xlab = 'recall', ylab = paste0('log(', colname, ')'), main = main, 
       bty = 'n')
  
  n <- length(rlist)
  if(n == 1) return()
  
  for(i in 2:n) {
    ri <- rlist[[i]]
    points(ri$recall, log10(ri[[colname]]), col = i + 1, pch = i + 14, type = 'b', lwd = 2, cex = .85)
  }
  
  legend('topleft', legend = names(rlist), col = (1:n) + 1, lwd = 2, bty = 'n')
}
