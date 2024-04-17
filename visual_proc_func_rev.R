noise_gen <- function(n, mu_x, sd_x, upper_cutoff = NaN) {
  mu_l = log(mu_x^2 / sqrt(mu_x^2 + sd_x^2))
  sd_l = sqrt(log(1+(sd_x^2 / mu_x^2)))
  return_this <- rlnorm(n, mu_l, sd_l)
  # now sample until we have no samples that are above cutoff.
  # this can take a looong time potentially.
  while(!is.na(upper_cutoff) & any(return_this>upper_cutoff)) {
    return_this[return_this>upper_cutoff] <- 
      rlnorm(length(return_this[return_this>upper_cutoff]), mu_l, sd_l)
  }
  return(return_this)
}

# local aux fun: heatmap
plot_heatmap <- function(mat, title="") {
  require(reshape2)
  require(ggplot2)
  require(viridis)
  mat_long <- melt(mat)
  colnames(mat_long) <- c("y", "x", "z")
  p <- ggplot(mat_long, aes(x = x, y = y, fill = z)) + 
    geom_tile() + 
    theme_classic() + 
    scale_fill_viridis_c() + coord_fixed(expand = FALSE) + 
    ggtitle(title)
  print(p)
}

# local aux fun: produce the temporal response kernel
gamma_fun <- function(x_dur = 200, x_step = 1, shape = 5, scale = 5, latency = 0, 
                      make_symmetric = FALSE, norm_amp = NaN) {
  # function to simulate a temporal response profile, using a gamma function
  # by Richard, 06/2021
  x_here <- seq(from = 0, to = x_dur, by = x_step)
  suppressWarnings( gamma_res <- dgamma(x = x_here, shape = shape, scale = scale) )
  # if either shape or scale are zero (or negative), then we have a special case,
  # i.e., a vector like [Inf, 0, 0, 0, ...], which we'll transform to [1, 0, 0, 0, ...]
  # this would be the zero dynamics model
  if (round(shape, 3) <= 0 | round(scale, 3) <= 0 | any(is.na(gamma_res))) {
    gamma_res <- dgamma(x = x_here, shape = 0, scale = 1) # produce the infinity vector
    assert_that(sum(is.infinite(gamma_res))==1) # there should only one infinite (at first place)
    gamma_res[is.infinite(gamma_res)] <- 1
  }
  if (latency>0) { # apply additional latency?
    latency_here <- seq(from = 0, to = latency, by = x_step)
    gamma_res <- c(rep(0, length(latency_here)), gamma_res)
    x_here <- c(latency_here, x_here+max(latency_here)+x_step)
  }
  if (!is.na(norm_amp)) { # normalize amplitude to one?
    gamma_res <- gamma_res / max(gamma_res)
    gamma_res <- gamma_res * norm_amp
  }
  if (make_symmetric) { # output center around 0?
    gamma_res <- c(rep(0, length(gamma_res)-1), gamma_res)
    x_here <- c(-1*rev(x_here[-1]), x_here)
  }
  return(list(gamma_res, x_here))
}


visual_proc_func <- function(signal_x, signal_y, signal_t, # properties of the signal (relative to saccade onset)
                             signal_x_range = NULL, signal_y_range = NULL, signal_t_range = NULL, # the min and max of temporal scale
                             do_on_GPU = FALSE, # set to TRUE to do the computations on the GPU
                             no_CUDA = FALSE, # set to TRUE if you want to use CPU instead of GPU (but torch)
                             use_half_on_GPU = FALSE, # if a GPU is available, use half-precision?
                             spatial_resolution = 0.05, # spatial resolution of processing function in dva
                             temporal_resolution = 1000/1440, # temporal resolution of processing function in milliseconds
                             gaussian_aperture_sd = 0.5, # size of gaussian blob stimulus in dva
                             gaussian_max_luminance = 1, # 0 .. 1 in normalized luminance space
                             spatial_field_sd = 0.15, # SD of receptive field, in dva (0.15 deg was chosen by Teichert and colleagues)
                             temporal_scale = 10, # scale parameter of Gamma function for temporal response
                             temporal_shape = 5, # shape parameter of Gamma function for temporal response
                             temporal_latency = 10, # latency of Gamma function for temporal response
                             temporal_duration = 300, # size [in ms] of the kernel for the temporal response function
                             contrast_thres = 0.5, # contrast sensitivity (0..1), that is, at what threshold is activity taken into account?
                             normalize_C = NaN, # use this value for normalization instead of max(activity)
                             check_for_outliers = FALSE, # if TRUE, then a check for outliers is performed on the weighted averages
                             debug_mode = FALSE, # set to TRUE for diagnostic output
                             show_result_plot = TRUE, # show the final results plot
                             skip_zeros = TRUE, # if TRUE, then we'll skip those convolutions where the input data consists only of zeros, 
                             use_polynomials = FALSE, # if TRUE, then instead of weighted averages, polynomials will be fitted
                             output_final_matrices = FALSE, # if TRUE, then the final matrices will be returned along with the weighted averages
                             use_CG = TRUE # TRUE: use center of gravity, FALSE: use maximum likelihood
) {
  # visual processing function, inspired by Teichert et al. (2010), JoV
  # by Richard Schweitzer
  
  if (debug_mode) { # or: https://stackoverflow.com/questions/11885207/get-all-parameters-as-list
    print(paste0("spatial_resolution = ", round(spatial_resolution, 3), "; ",
                 "temporal_resolution = ", round(temporal_resolution, 3), "; ",
                 "gaussian_aperture_sd = ", round(gaussian_aperture_sd, 3), "; ",
                 "spatial_field_sd = ", round(spatial_field_sd, 3), "; "))
    print(paste0("temporal_scale = ", round(temporal_scale, 3), "; ",
                 "temporal_shape = ", round(temporal_shape, 3), "; ",
                 "temporal_latency = ", round(temporal_latency, 3), "; ",
                 "temporal_duration = ", round(temporal_duration, 3), "; ",
                 "normalize_C = ", round(normalize_C, 3), "; ",
                 "contrast_thres = ", round(contrast_thres, 3), ". "))
  }
  # required stuff
  require(pracma)
  require(reshape2)
  require(data.table)
  require(ggplot2)
  require(assertthat)
  if (do_on_GPU) {
    require(torch)
  } else {
    require(foreach)
  }
  ## 0.1 compute properties of signal
  length_signal_x <- abs(max(signal_x)-min(signal_x)) # the spatial extent of the signal 
  length_signal_y <- abs(max(signal_y)-min(signal_y))
  length_signal_t <- abs(max(signal_t)-min(signal_t)) # the spatial extent of the signal 
  assert_that(length(signal_x)==length(signal_y))
  assert_that(length(signal_x)==length(signal_t))
  n_samples <- length(signal_t)
  signal_df <- data.frame(signal_x, signal_y, signal_t)
  signal_df$n_samples <- 1:n_samples
  if (debug_mode) {
    print("Input:")
    print(signal_df)
  }
  ## STEP 1: fill the position-over-time matrix
  # 1.1 create the gaussian stimulus aperture
  blob_size <- seq(-3*gaussian_aperture_sd, 3*gaussian_aperture_sd, spatial_resolution)
  meshlist <- meshgrid(x = blob_size, y = blob_size)
  blob <- exp(-(meshlist$X^2+meshlist$Y^2)/(2*gaussian_aperture_sd^2)) # compute gaussian aperture of stimulus
  blob <- blob * gaussian_max_luminance # rescale to maximum luminance
  dimnames(blob) <- list(blob_size, blob_size) # give the blob proper dimension names
  blob_length_x <- ncol(blob)
  blob_length_y <- nrow(blob)
  if (debug_mode) {
    print(paste("Created Gaussian blob stimulus with size", blob_length_y, blob_length_x))
  }
  # 1.2 get the dimensions for the cells of the matrix
  spatial_pad_size <- 4*gaussian_aperture_sd
  if (do_on_GPU) {
    # upscale when done on GPU?
    temporal_pad_size <- round(0.5*(temporal_duration+temporal_latency))
  } else {
    temporal_pad_size <- round(0.5*(temporal_duration+temporal_latency))
  }
  # X
  if (!is.null(signal_x_range)) {
    mat_size_x = seq(signal_x_range[1]-spatial_pad_size, signal_x_range[2]+spatial_pad_size, 
                     by = spatial_resolution) # columns, x coordinate
  } else {
    mat_size_x = seq(min(signal_x)-spatial_pad_size, max(signal_x)+spatial_pad_size, 
                     by = spatial_resolution) # columns, x coordinate
  }
  # Y
  if (!is.null(signal_y_range)) {
    mat_size_y = seq(signal_y_range[1]-spatial_pad_size, signal_y_range[2]+spatial_pad_size, 
                     by = spatial_resolution) # rows, y coordinate
  } else {
    mat_size_y = seq(min(signal_y)-spatial_pad_size, max(signal_y)+spatial_pad_size, 
                     by = spatial_resolution) # rows, y coordinate
  }
  # T
  if (!is.null(signal_t_range)) { # compute the temporal scale based on signal_t_range
    mat_size_t = seq(signal_t_range[1], signal_t_range[2]+temporal_pad_size, 
                     by = temporal_resolution)
  } else { # compute the temporal scale based on signal_t
    if (min(signal_t)>=0) {
      min_t <- 0
    } else {
      min_t <- min(signal_t)
    }
    mat_size_t = seq(min_t, max(signal_t)+temporal_pad_size, 
                     by = temporal_resolution) # third dimension, t coordinate (0 was previously: min(signal_t)-round(length_signal_t))
  }
  # 1.3 create large matrix of zeros according to the specified resolution
  mat_over_t <- array(data = 0, dim = c(length(mat_size_y), length(mat_size_x), length(mat_size_t)), 
                      dimnames = list(mat_size_y, mat_size_x, mat_size_t) )
  if (debug_mode) {
    print(paste("Allocated mat_over_t with size", paste(dim(mat_over_t), collapse = ",") )) 
  }
  # 1.4 fill the matrix with the blob where the position is present in the signal
  for (sample_i in (1:n_samples)) {
    put_sample_x_here <- which.min(abs(mat_size_x - signal_x[sample_i]))
    put_sample_y_here <- which.min(abs(mat_size_y - signal_y[sample_i]))
    put_sample_t_here <- which.min(abs(mat_size_t - signal_t[sample_i]))
    #print(paste(put_sample_x_here, put_sample_y_here, put_sample_t_here))
    mat_over_t[seq(put_sample_y_here-round(blob_length_y/2), 
                   put_sample_y_here-round(blob_length_y/2)+blob_length_y-1, 
                   by = 1), 
               seq(put_sample_x_here-round(blob_length_x/2), 
                   put_sample_x_here-round(blob_length_x/2)+blob_length_x-1, 
                   by = 1), 
               put_sample_t_here] <- blob
    #print(address(mat_over_t)) # this operation works in place
  }
  last_signal_t_index <- put_sample_t_here
  last_signal_x_index <- put_sample_x_here
  last_signal_y_index <- put_sample_y_here
  if (debug_mode) {
    print("Created and filled activation matrix with dimensions:")
    print(dim(mat_over_t))
  }
  # STEP 2: CONVOLUTION
  # 2.0 create spatial response function
  spat_fun_size <- seq(-3*spatial_field_sd, 3*spatial_field_sd, spatial_resolution)
  if (mod(length(spat_fun_size), 2) == 0) { # the spatial kernel needs to have odd dimensions for the 2D convolution
    zero_center <- seq(0, 3*spatial_field_sd, spatial_resolution)
    spat_fun_size <- c(-1*rev(zero_center)[-length(zero_center)], zero_center)
    assert_that(!(mod(length(spat_fun_size), 2) == 0))
  }
  spat_meshlist <- meshgrid(x = spat_fun_size, y = spat_fun_size)
  spat_kernel <- 1/((spatial_field_sd^2)*2*pi) * exp(-(spat_meshlist$X^2+spat_meshlist$Y^2)/(2*spatial_field_sd^2))
  spat_kernel_length_x <- ncol(spat_kernel)
  spat_kernel_length_y <- nrow(spat_kernel)
  if (debug_mode) {
    plot_heatmap(spat_kernel, paste0("Spatial RF kernel with SD=", spatial_field_sd))
    print("Created spatial RF kernel with dimensions:")
    print(dim(spat_kernel))
  }
  # 2.1 create temporal response function
  #assert_that(temporal_latency>=0 & temporal_latency<temporal_duration/2 & temporal_scale>=0 & temporal_shape>=0)
  resp_fun <- gamma_fun(x_dur = temporal_duration, x_step = temporal_resolution, 
                        scale = temporal_scale, latency = temporal_latency, 
                        shape = temporal_shape, # can be kept constant, as by Teichert and colleagues, i.e., 5
                        make_symmetric = FALSE)
  resp_fun_time <- resp_fun[[2]]
  resp_fun <- resp_fun[[1]]
  are_equal(length(resp_fun_time), length(resp_fun))
  if (mod(length(resp_fun), 2) == 0) { # make an asymmetric temporal 
    resp_fun_time <- resp_fun_time[-length(resp_fun_time)]
    resp_fun <- resp_fun[-length(resp_fun)]
    assert_that(mod(length(resp_fun), 2) == 1)
  }
  if (debug_mode) {
    plot(resp_fun_time, resp_fun, main = paste0("Temporal response function with:\n", 
                                                "scale=", round(temporal_scale, 2), 
                                                ", shape=", round(temporal_shape, 2), 
                                                ", latency=", round(temporal_latency, 2) ) )
    print(paste0("Created temporal response kernel with length of ", length(resp_fun), "."))
  }
  ########## PERFORM CONVOLUTIONS HERE ###########
  if (!do_on_GPU) { ################################################## PERFORM OPERATIONS ON CPU
    # 2.2 perform column-wise convolution with spatial function
    if (debug_mode) {
      print("Performing convolutions with spatial RF kernel...")
    }
    mat_over_t_spatial <- foreach(i = 1:(dim(mat_over_t)[3]), .packages = c("smoothie")) %do% {
      if (skip_zeros==TRUE && all(mat_over_t[ , , i]==0)) {
        convoluted_image <- mat_over_t[ , , i]
      } else {
        convoluted_image <- kernel2dsmooth(x = mat_over_t[ , , i], K = spat_kernel) 
      }
      return(convoluted_image)
    } 
    # ... and convert to array
    mat_over_t_spatial <- array(data = unlist(mat_over_t_spatial), 
                                dim = c(length(mat_size_y), length(mat_size_x), length(mat_size_t)), 
                                dimnames = list(mat_size_y, mat_size_x, mat_size_t)  )
    assert_that(all(dim(mat_over_t)==dim(mat_over_t_spatial)))
    if (debug_mode) {
      plot_heatmap(mat_over_t[ , , last_signal_t_index], paste0("Retinal input at t=", 
                                                                mat_size_t[last_signal_t_index]))
      plot_heatmap(mat_over_t_spatial[ , , last_signal_t_index], paste0("Output of spatial RF at t=", 
                                                                        mat_size_t[last_signal_t_index]))
      print("Done. Dimensions of new array:")
      print(dim(mat_over_t_spatial))
    }
    if (!debug_mode) {
      rm(mat_over_t) # to free working memory
    }
    # 2.3 perform convolution with temporal function
    # ideally, use parApply with the SNOW backend for that, which works well!
    if (debug_mode) {
      print("Performing convolutions with temporal response kernel...")
    }
    mat_over_t_temporal <- apply(X = mat_over_t_spatial, MARGIN = c(1,2),
                                 FUN = function(x, resp_f, do_skip_zeros) {
                                   if (do_skip_zeros==TRUE && all(x==0)) {
                                     convoluted <- x
                                   } else {
                                     convoluted <- zapsmall(convolve(x, rev(resp_f), type = "open"))
                                     convoluted <- convoluted[1:length(x)]
                                   }
                                   return(convoluted)
                                 }, resp_fun, skip_zeros)
    mat_over_t_temporal <- aperm(mat_over_t_temporal, c(2,3,1)) # rearrange dimensions
    mat_over_t_temporal <- array(data = unlist(mat_over_t_temporal), 
                                 dim = c(length(mat_size_y), length(mat_size_x), length(mat_size_t)), 
                                 dimnames = list(mat_size_y, mat_size_x, mat_size_t)  )
    assert_that(all(dim(mat_over_t_temporal)==dim(mat_over_t_spatial)))
    if (debug_mode) {
      plot(mat_size_t, mat_over_t_temporal[last_signal_y_index, last_signal_x_index, ], 
           main = paste0("Temporal response over time at\n", 
                         "x=", mat_size_x[last_signal_x_index], 
                         ", y=", mat_size_y[last_signal_y_index]) )
      plot_heatmap(mat_over_t_temporal[ , , last_signal_t_index], 
                   paste0("Spatial and temporal response at time t=", 
                          mat_size_t[last_signal_t_index]))
      print("Done. Dimensions of new array:")
      print(dim(mat_over_t_temporal))
    }
    if (!debug_mode) {
      rm(mat_over_t_spatial) # to free working memory
    }
    # STEP 3: Compute the neural signal
    # 3.1 normalize the activation, so that we can set a comparable threshold, and set all values below it to zero
    mat_over_t_temporal_normalized <- copy(mat_over_t_temporal)
    if (debug_mode) {
      print(paste("Maximum activity is:", max(mat_over_t_temporal_normalized), "."))
    }
    # Naka-Rushton transform here to normalize between 0 and 1
    if (is.na(normalize_C)) { # if no C is specified, use a relative C
      relative_C <- max(mat_over_t_temporal_normalized) * 0.5
      # simple activity normalization
      mat_over_t_temporal_normalized <- mat_over_t_temporal_normalized^2 / (mat_over_t_temporal_normalized^2 + relative_C^2)
    } else { # use a pre-specified value to normalize and Naka-Rushton transform
      mat_over_t_temporal_normalized <- mat_over_t_temporal_normalized^2 / (mat_over_t_temporal_normalized^2 + normalize_C^2)
    }
    # now perform contrast thres cutoff
    mat_over_t_temporal_normalized[mat_over_t_temporal_normalized < contrast_thres] <- 0
    assert_that(all(dim(mat_over_t_temporal)==dim(mat_over_t_temporal_normalized)))
    if (debug_mode) {
      t_strongest_response <- which(mat_over_t_temporal[last_signal_y_index, last_signal_x_index, ]==
                                      max(mat_over_t_temporal[last_signal_y_index, last_signal_x_index, ]))[1]
      plot_heatmap(mat_over_t_temporal[ , , t_strongest_response], 
                   paste0("Spatial and temporal activity\nat time of strongest response t=", mat_size_t[t_strongest_response]))
      plot_heatmap(mat_over_t_temporal_normalized[ , , t_strongest_response], 
                   paste0("Spatial and temporal activity above threshold (c=", contrast_thres, 
                          ")\nat time of strongest response t=", mat_size_t[t_strongest_response]))
      print("Performed normalization on spatial and temporal activity.")
    }
    # 3.2 computed the (weighted averages) represented positions
    weighted_averages <- vector(mode = "list", length = dim(mat_over_t_temporal_normalized)[3])
    for (i in 1:(dim(mat_over_t_temporal_normalized)[3])) {
      mat_normalized_long <- reshape2::melt(mat_over_t_temporal_normalized[ , , i], varnames = c("y", "x"), value.name = "w")
      n_nonzero_cells <- sum(as.numeric(mat_normalized_long$w>0))
      if (n_nonzero_cells > 2) {
        mean_x <- weighted.mean(x = mat_normalized_long$x, w = mat_normalized_long$w)
        mean_y <- weighted.mean(x = mat_normalized_long$y, w = mat_normalized_long$w)
        mean_w <- mean(mat_normalized_long$w)
      } else {
        mean_x <- NaN
        mean_y <- NaN
        mean_w <- 0
      }
      weighted_averages[[i]] <- data.frame(x = mean_x, y = mean_y, #w = mean_w,
                                           t = as.numeric(names(mat_over_t_temporal_normalized[1,1, ]))[i] )
    }
    weighted_averages <- rbindlist(weighted_averages)
    if (debug_mode) {
      print("Computed averages weighted by activity.")
    }
    if (use_polynomials) { # compute weighted averages using polynomials? (is extremely slow and memory-expensive)
      apply_padding <- TRUE
      # transform matrix into long format
      mat_over_t_temp_long <- reshape2::melt(mat_over_t_temporal_normalized, varnames = c("y", "x", "t"), value.name = "w")
      setDT(mat_over_t_temp_long)
      mat_over_t_temp_long[ , all_zeros := all(w==0), by = .(t)] # mark those time points that are all zero
      if (apply_padding) {
        first_w <- mat_over_t_temp_long[all_zeros==FALSE][t==min(t)][ , c("t", "w")]
        last_w <- mat_over_t_temp_long[all_zeros==FALSE][t==max(t)][ , c("t", "w")]
        mat_over_t_temp_long[all_zeros==TRUE & t<unique(first_w$t), w := first_w$w, 
                             by = .(t)] 
        mat_over_t_temp_long[all_zeros==TRUE & t>unique(last_w$t), w := last_w$w, 
                             by = .(t)] 
        mat_over_t_temp_long <- mat_over_t_temp_long[t>=unique(first_w$t)-10 & t<=unique(last_w$t)+10]
      } else {
        mat_over_t_temp_long <- mat_over_t_temp_long[all_zeros==FALSE]
      }
      # fit polynomial for x
      m_poly_x <- lm(data = mat_over_t_temp_long, 
                     formula = x ~ poly(t, 3), weights = w)
      weighted_averages$x_poly <- predict(object = m_poly_x, newdata = weighted_averages)
      weighted_averages$x_poly[is.na(weighted_averages$x)] <- NaN
      # fit polynomial for y
      m_poly_y <- lm(data = mat_over_t_temp_long, 
                     formula = y ~ poly(t, 3), weights = w)
      weighted_averages$y_poly <- predict(object = m_poly_y, newdata = weighted_averages)
      weighted_averages$y_poly[is.na(weighted_averages$y)] <- NaN
      # save working mem
      rm(mat_over_t_temp_long, m_poly_x, m_poly_y)
      # diagnostic plot?
      if (debug_mode) {
        print("Also computed averages by weighted polynomial fits.")
        plot(weighted_averages$t, weighted_averages$x, main = "Weighted averages / polynomials X")
        points(weighted_averages$t, weighted_averages$x_poly, col = "red")
        plot(weighted_averages$t, weighted_averages$y, main = "Weighted averages / polynomials Y")
        points(weighted_averages$t, weighted_averages$y_poly, col = "red")
      }
    }
    
  } else { ################################################## PERFORM OPERATIONS ON GPU
    
    # is cuda available?
    # ... and what should be the preferred data type? (there are torch_half, torch_float, torch_double)
    if (cuda_is_available()) {
      if (no_CUDA) {
        use_device <- torch_device('cpu')
      } else {
        use_device <- torch_device('cuda') # 'cuda'
      }
      if (use_half_on_GPU) {
        preferred_dtype <- torch_float16() # or half as mixed/half computing is optimized?
      } else {
        preferred_dtype <- torch_float32()
      }
    } else {
      use_device <- torch_device('cpu')
      preferred_dtype <- torch_float32()
    }
    
    # 0) reshape the mat_over_t to match the 4D requirements of the 2D convolution
    mat_over_t_gpu <- torch_tensor(mat_over_t, device = use_device, dtype = preferred_dtype)
    mat_over_t_gpu <- mat_over_t_gpu$permute(c(3, 1, 2))$unsqueeze(1)
    if (debug_mode) {
      print("Converted matrix to tensor and reshaped to dimensions:")
      print(mat_over_t_gpu$size())
    }
    # 1) reshape spatial convolution kernel to (B, C, H, W)
    spat_kernel_gpu <- torch_tensor(spat_kernel, device = use_device, dtype = preferred_dtype)
    spat_kernel_gpu <- spat_kernel_gpu$reshape(c(1, 1, spat_kernel_gpu$size(1), spat_kernel_gpu$size(2)))
    spat_kernel_gpu <- spat_kernel_gpu$expand(c(mat_over_t_gpu$size(2), 1, spat_kernel_gpu$size(3), spat_kernel_gpu$size(4)))
    if (debug_mode) {
      print("Created spatial response tensor of dimensions for 2D convolution:")
      print(spat_kernel_gpu$size())
    }
    # 2) run the spatial convolution, according to https://discuss.pytorch.org/t/manual-2d-convolution-per-channel/83907
    mat_over_t_spatial_gpu <- torch_conv2d(input = mat_over_t_gpu, weight = spat_kernel_gpu, 
                                           padding = c(spat_kernel_gpu$size(3)%/%2, spat_kernel_gpu$size(4)%/%2), # make sure the padding retains the overall size of the image
                                           groups = mat_over_t_gpu$size(2) # number of time points
    )
    assert_that(all(mat_over_t_gpu$size()==mat_over_t_spatial_gpu$size()))
    if (debug_mode) {
      print("Performed 2D spatial convolution, thereby achieving matrix with size:")
      print(mat_over_t_spatial_gpu$size())
      plot_heatmap(as_array(mat_over_t_gpu[1, last_signal_t_index-3, , ]$cpu()), title = paste("index =", i), ref_scale = NaN)
      plot_heatmap(as_array(mat_over_t_spatial_gpu[1, last_signal_t_index-3, , ]$cpu()), title = paste("index =", i), ref_scale = NaN)
    } else {
      rm(spat_kernel_gpu, spat_kernel, mat_over_t_gpu, mat_over_t)  # save memory!
    }
    # 3) prepare the temporal response kernel (is has odd dimension, we made sure of that before)
    resp_fun_gpu <- torch_tensor(resp_fun, device = use_device, dtype = preferred_dtype)
    resp_fun_gpu <- resp_fun_gpu$expand(c(mat_over_t_spatial_gpu$size(4), 1, resp_fun_gpu$size())) # expand to match matrix dimensions
    resp_fun_gpu <- torch_cat(list(torch_zeros(c(resp_fun_gpu$size(1), resp_fun_gpu$size(2), resp_fun_gpu$size(3)-1), device = use_device, dtype = preferred_dtype), # create the zero pad to center the TRF
                                   resp_fun_gpu), dim = 3)
    if (debug_mode) {
      print("Created temporal response tensor of dimensions for 1D convolution:")
      print(resp_fun_gpu$size())
    }
    # 4) run the temporal convolution
    mat_over_t_spatial_gpu <- mat_over_t_spatial_gpu$permute(c(1, 3, 4, 2))$squeeze() # new dim: y, x, t
    if (debug_mode) {
      print("Reshaped main tensor to dimensions to prepare for 1D temporal convolution:")
      print(mat_over_t_spatial_gpu$size())
    }
    # 5) run the temporal convolution
    mat_over_t_temporal_gpu <- torch_conv1d(input = mat_over_t_spatial_gpu, 
                                            weight = resp_fun_gpu$flip(3), 
                                            padding = resp_fun_gpu$size(3)%/%2, 
                                            groups = mat_over_t_spatial_gpu$size(2) # is the x dimension
    )
    are_equal(mat_over_t_temporal_gpu$size(), mat_over_t_spatial_gpu$size())
    if (debug_mode) {
      print("Performed convolution through time, resulting in matrix of dim:")
      print(mat_over_t_temporal_gpu$size())
      # # for debug: initial cpu-based version
      # plot_heatmap(mat_over_t_temporal[ , , last_signal_t_index], 
      #              title = paste0("Spatial and temporal response at time t=", 
      #                     mat_size_t[last_signal_t_index]), 
      #              ref_scale = NaN)
      # version with torch
      plot_heatmap(as_array(mat_over_t_temporal_gpu[ , , last_signal_t_index]$cpu()), 
                   title = paste0("Spatial and temporal response with torch"), 
                   ref_scale = NaN)
    } else {
      rm(mat_over_t_spatial_gpu, resp_fun_gpu, resp_fun) # save memory!
    }
    # 6) perform the Naka-Rushton transformation
    if (is.na(normalize_C)) { # if no C is specified, use a relative C
      relative_C <- mat_over_t_temporal_gpu$max() * 0.5
      # simple activity normalization
      mat_over_t_temporal_gpu_normalized <- torch_div(mat_over_t_temporal_gpu$square(), 
                                                      mat_over_t_temporal_gpu$square()$add(relative_C$square()) )
    } else { # use a pre-specified value to normalize and Naka-Rushton transform
      normalize_C_gpu <- torch_tensor(normalize_C, device = use_device, dtype = preferred_dtype)
      mat_over_t_temporal_gpu_normalized <- torch_div(mat_over_t_temporal_gpu$square(), 
                                                      mat_over_t_temporal_gpu$square()$add(normalize_C_gpu$square()) )
    }
    # 7) set all values below the specified threshold to zero
    contrast_thres_gpu <- torch_tensor(contrast_thres, device = use_device, dtype = preferred_dtype)
    mat_over_t_temporal_gpu_normalized[mat_over_t_temporal_gpu_normalized < contrast_thres_gpu] = 0 # this is an in-place operation
    if (debug_mode) {
      print("Smallest non-zero number in normalized matrix:")
      print(as.numeric(mat_over_t_temporal_gpu_normalized[mat_over_t_temporal_gpu_normalized!=0]$min()$cpu()))
    }
    # 8) extract the weighted averages
    # 8.1) get the coordinates
    mat_size_x_gpu <- torch_tensor(mat_size_x, device = use_device, dtype = preferred_dtype)
    mat_size_x_gpu$unsqueeze_(c(1))
    mat_size_x_gpu$unsqueeze_(c(3))
    mat_size_y_gpu <- torch_tensor(mat_size_y, device = use_device, dtype = preferred_dtype)
    mat_size_y_gpu$unsqueeze_(c(2))
    mat_size_y_gpu$unsqueeze_(c(3))
    # 8.2) multiply those with weights and compute weighted average (and have them ready on CPU!)
    sum_of_weights <- torch_sum(self = mat_over_t_temporal_gpu_normalized, dim = c(1,2), 
                                dtype = torch_float32() )
    weighted_avg_x <- as_array((torch_sum(self = mat_over_t_temporal_gpu_normalized$mul(mat_size_x_gpu), dim = c(1,2), 
                                          dtype = torch_float32() ) / 
                                  sum_of_weights)$cpu() )
    weighted_avg_y <- as_array((torch_sum(self = mat_over_t_temporal_gpu_normalized$mul(mat_size_y_gpu), dim = c(1,2), 
                                          dtype = torch_float32() ) / 
                                  sum_of_weights)$cpu() )
    rm(sum_of_weights)
    assert_that(length(weighted_avg_x)==length(weighted_avg_y) & length(weighted_avg_x)==length(mat_size_t))
    weighted_averages <- data.table(x = weighted_avg_x, y = weighted_avg_y, t = mat_size_t)
    # 9) finally, trim down the matrices and transfer them back from GPU (TO DO: trim?)
    mat_over_t_temporal_normalized <- as_array(mat_over_t_temporal_gpu_normalized$to(torch_float32())$cpu())
    dimnames(mat_over_t_temporal_normalized) <- list(mat_size_y, mat_size_x, mat_size_t)
    mat_over_t_temporal <- as_array(mat_over_t_temporal_gpu$to(torch_float32())$cpu())
    dimnames(mat_over_t_temporal) <- list(mat_size_y, mat_size_x, mat_size_t)
    # done.
  }
  
  ## Check for outliers in the sequence? is the time sequence uniformly sampled?
  # TO DO: do this better
  if (check_for_outliers) {
    weighted_averages_check <- copy(weighted_averages)
    weighted_averages_check <- weighted_averages_check[complete.cases(weighted_averages_check)]
    # compute the mean distance in space and time between points
    weighted_averages_check[ , x_diff := c(diff(x), median(diff(x)))]
    weighted_averages_check[ , y_diff := c(diff(y), median(diff(y)))]
    weighted_averages_check[ , t_diff := c(diff(t), median(diff(t)))]
    # check whether the sequence is uniformly sampled
    time_of_outlier <- NaN
    if (length(unique(round(weighted_averages_check$t_diff,2)))>1) { # if not uniformly sampled, there's likely an outlier
      weighted_averages_check[ , all_diff := sqrt(x_diff^2 + y_diff^2 + t_diff^2)]
      weighted_averages_check[ , all_diff_cutoff := median(all_diff)+2*sd(all_diff)]
      weighted_averages_check[ , all_diff_extreme := (all_diff > all_diff_cutoff)]
      if (any(weighted_averages_check$all_diff_extreme)) { # check whether any outlier could be detected
        time_of_outlier <- max(weighted_averages_check[all_diff_extreme==TRUE, t])
      }
    }
    # have we found an outlier?
    if (!is.na(time_of_outlier)) {
      weighted_averages[t<=time_of_outlier, x := NaN]
      weighted_averages[t<=time_of_outlier, y := NaN]
    }
    rm(weighted_averages_check)
  } # end of checking for outliers in the weighted averages
  
  # 3.3 Finally, diagnostic plots
  if (debug_mode | show_result_plot) {
    # x over t (need to aggregate across y)
    mat_x_over_t <- apply(X = mat_over_t_temporal, MARGIN = c(2,3), FUN = mean) # aggregate on activity
    mat_x_over_t_long <- reshape2::melt(mat_x_over_t, varnames = c("x", "t"), value.name = "z")
    mat_x_over_t.norm <- apply(X = mat_over_t_temporal_normalized, MARGIN = c(2,3), FUN = mean) # aggregate on normalized activity
    mat_x_over_t_long.norm <- reshape2::melt(mat_x_over_t.norm, varnames = c("x", "t"), value.name = "z")
    p_x <- ggplot(mat_x_over_t_long.norm, aes(x = t, y = x)) + 
      geom_tile(aes(fill = z)) + 
      # geom_rug(data = mat_x_over_t_long.norm[mat_x_over_t_long.norm$z>0, ], 
      #          aes(x = t, y = x), color = "white", alpha = 0.1) + 
      geom_point(data = signal_df, aes(x = signal_t, y = signal_x), color = "orange", alpha = 0.6) + 
      geom_point(data = weighted_averages, aes(x = t, y = x), color = "red", alpha = 0.6) + 
      theme_classic() + coord_cartesian(expand = FALSE) + 
      scale_fill_viridis_c() + 
      labs(x = "Time [ms]", y = "X coordinates (averaged over Y)", 
           fill = "Activity")
    if (use_polynomials) { 
      p_x <- p_x + geom_point(data = weighted_averages, aes(x = t, y = x_poly), 
                              color = "orange")
    }
    if (exists("StrkTheme")) {
      p_x <- p_x + StrkTheme()
    }
    suppressWarnings(print(p_x))
    # # y over t
    # mat_y_over_t <- apply(X = mat_over_t_temporal, MARGIN = c(1,3), FUN = mean)
    # mat_y_over_t_long <- reshape2::melt(mat_y_over_t, varnames = c("y", "t"), value.name = "z")
    # mat_y_over_t.norm <- apply(X = mat_over_t_temporal_normalized, MARGIN = c(1,3), FUN = mean)
    # mat_y_over_t_long.norm <- reshape2::melt(mat_y_over_t.norm, varnames = c("y", "t"), value.name = "z")
    # p_y <- ggplot(mat_y_over_t_long, aes(x = t, y = y)) + 
    #   geom_tile(aes(fill = z)) + 
    #   geom_rug(data = mat_y_over_t_long.norm[mat_y_over_t_long.norm$z>0, ], 
    #            aes(x = t, y = y), color = "white", alpha = 0.1) + 
    #   geom_point(data = signal_df, aes(x = signal_t, y = signal_y), color = "red") + 
    #   geom_point(data = weighted_averages, aes(x = t, y = y), color = "red") + 
    #   theme_classic() + coord_cartesian(expand = FALSE) + 
    #   scale_fill_viridis_c() + 
    #   labs(x = "Time [ms]", y = "Y coordinates [pix] (averaged over X dimension)", 
    #        fill = "Activity")
    # if (use_polynomials) { 
    #   p_y <- p_y + geom_point(data = weighted_averages, aes(x = t, y = y_poly), 
    #                           color = "orange")
    # }
    # if (exists("StrkTheme")) {
    #   p_y <- p_y + StrkTheme()
    # }
    # suppressWarnings(print(p_y))
    if (debug_mode) {
      print("Showing results of visual processing.")
    }
  }
  # shall the final matrices be returned??
  if (output_final_matrices) {
    return_this <- list(weighted_averages, mat_over_t_temporal, mat_over_t_temporal_normalized)
  } else {
    return_this <- weighted_averages
  }
  return(return_this)
}
