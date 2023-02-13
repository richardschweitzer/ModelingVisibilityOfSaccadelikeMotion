get_max_se <- function(x, n_reps = 5000, n_cores = 1, print_output = FALSE) {

    if (n_cores > 1) { 
      parallelize <- "multicore" 
    } else {
      parallelize <- "no"    
    }
    
    # compute max
    mx = max(x, na.rm = TRUE)
    m = mean(x[x>0], na.rm = TRUE)
    
    if (length(x)>0 & !is.na(mx)) {
      # for bootstrapping we need a sampling function
      samplemx <- function(x, d) {
        return(max(x[d], na.rm = TRUE))
      }
      # now we bootstrap to estimate the median
      require(boot)
      booty = boot(x, samplemx, 
                   R = n_reps, parallel = parallelize, ncpus = n_cores)  
      # compute 
      se <- sd(booty$t)
      # compute confidence interval based on Efron's confident limit
      upper <- quantile(booty$t, 0.975, names = FALSE)
      lower <- quantile(booty$t, 0.025, names = FALSE)
    } else {
      se <- NA
      upper <- NA
      lower <- NA
    }
    
    # return values
    res <- list(w = mx, w_se = se, w_lower = lower, w_upper = upper, m = m)
    if (print_output) {
      print(res)
    }
    return(res)
}
