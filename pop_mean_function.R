# weighted mean with NaN replacement (not used)
wmean <- function(x, w, replacement = 0, na.rem = TRUE) {
  res <- weighted.mean(x = x, w = w, na.rm = na.rem)
  res[is.na(res)] <- replacement
  return(res)
}

# this function creates by population mean, i.e., mean weighted by activity and distance from max
pop_mean <- function(x, y, # the definition of the 2D field 
                     z, # extra variable, say the activation
                     w, # the weights / evidence
                     do_compute_pop_mean = TRUE, # set to FALSE to skip population mean
                     gaussian_integration_sd = 2, # should be in the metric of x and y, or NaN to perform simple weighted average
                     replacement = 0, na.rem = TRUE, do_plot = FALSE) {
  # find the maximum activation
  w_max <- max(w, na.rm = na.rem)
  max_index <- which(!is.na(w) & w==w_max)
  x_max <- mean(x[max_index], na.rm = na.rem)
  y_max <- mean(y[max_index], na.rm = na.rem)
  z_max <- mean(z[max_index], na.rm = na.rem)
  # do we want to compute the population mean?
  if (do_compute_pop_mean) {
    # produce the Gaussian integration field, otherwise just perform the simple weighted mean
    if (!is.null(gaussian_integration_sd) && !is.na(gaussian_integration_sd) && gaussian_integration_sd>0) {
      dist_from_max <- sqrt((x-x_max)^2+(y-y_max)^2)
      gaussian_field <- dnorm(x = dist_from_max, mean = 0, sd = gaussian_integration_sd)
    } else {
      gaussian_field <- 1
    }
    # now perform the weighted mean based on the weights and spatial (gaussian) weights
    these_weights <- (w/max(w)) * (gaussian_field/max(gaussian_field))
    x_mean <- weighted.mean(x = x, w = these_weights, na.rm = na.rem)
    if (is.na(x_mean)) { x_mean <- replacement }
    y_mean <- weighted.mean(x = y, w = these_weights, na.rm = na.rem)
    if (is.na(y_mean)) { y_mean <- replacement }
    z_mean <- weighted.mean(x = z, w = these_weights, na.rm = na.rem)
    if (is.na(z_mean)) { z_mean <- replacement }
    # plot, maybe?
    if (do_plot) {
      # # to debug:
      # x = sim_sac_df$x[sim_sac_df$t==unique(sim_sac_df$t)[200]]
      # y = sim_sac_df$y[sim_sac_df$t==unique(sim_sac_df$t)[200]]
      # z = sim_sac_df$w_present[sim_sac_df$t==unique(sim_sac_df$t)[200]]
      # w = sim_sac_df$O_present[sim_sac_df$t==unique(sim_sac_df$t)[200]]
      # w = w + rev(w/2)
      require(ggplot2)
      require(viridis)
      # the field
      df <- data.frame(x = x, y = y, w = w, 
                       dist_from_max = dist_from_max, gaussian_field = gaussian_field)
      # the estimates
      x_mean_simple <- weighted.mean(x = x, w = (w/max(w)), na.rm = na.rem)
      y_mean_simple <- weighted.mean(x = y, w = (w/max(w)), na.rm = na.rem)
      df_res <- data.frame(x_max = x_max, y_max = y_max, x_mean = x_mean, y_mean = y_mean, 
                           x_mean_simple = x_mean_simple, y_mean_simple = y_mean_simple)
      # plot them
      pepe <- ggplot(df, aes(x = x, y = y, fill = w)) + geom_raster(interpolate = TRUE) + scale_fill_viridis_c() + 
        geom_point(data = df_res, aes(x = x_max, y = y_max), fill = NA, color = "black") + 
        geom_point(data = df_res, aes(x = x_mean, y = y_mean), fill = NA, color = "red") + 
        geom_point(data = df_res, aes(x = x_mean_simple, y = y_mean_simple), fill = NA, color = "orange") + 
        theme_classic() + coord_fixed(expand = FALSE)
      print(pepe)
    }
  } else { # no computation of weighted means
    x_mean <- NaN
    y_mean <- NaN
    z_mean <- NaN
  }
  return(list(w_max, x_max, y_max, z_max, x_mean, y_mean, z_mean))
}

