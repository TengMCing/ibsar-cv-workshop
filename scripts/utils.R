#' Convert an RGB image array into a tidy data frame of pixel colors.
#'
#' @param x Array/Numpy array. A 3D R array or numpy array with three color channels (RGB).
#' @return A data frame with pixel positions (`height`, `width`) and their hex color values.
process_rgb_array <- function(x) {
  if (length(dim(x)) != 3) cli::cli_abort("Argument `x` must be a 3D array.")
  if (dim(x)[3] != 3) cli::cli_abort("Argument `x` must have exactly three color channels (RGB).")
  
  purrr::map_df(1:3, function(i) {
    x[, , i] |>
      as.data.frame() |>
      dplyr::mutate(height = 1:32) |>
      tidyr::pivot_longer(V1:V32) |>
      dplyr::mutate(name = gsub("V", "", name)) |>
      dplyr::mutate(name = as.integer(name)) |>
      dplyr::rename(width = name) |>
      dplyr::mutate(channel = c("R", "G", "B")[i])
  }) |>
    dplyr::group_by(height, width) |>
    dplyr::summarise(color = rgb(value[channel == "R"], 
                                 value[channel == "G"],
                                 value[channel == "B"],
                                 maxColorValue = 255))
                          
}

#' Plot an RGB image using ggplot2
#'
#' @param x Array/Numpy array. A 3D array (R array or NumPy array) representing an RGB image.
#' @param title Character. The plot title.
#' @param axis Logical. Whether to display axis ticks, labels, and titles.
#' @return A ggplot object visualizing the RGB image.
plot_rgb <- function(x, title = NULL, axis = TRUE) {
  
  np <- reticulate::import("numpy", convert = FALSE)
  
  if (reticulate::is_py_object(x)) {
    if (!py_to_r(scrubwren::py_builtins$isinstance(x, np$ndarray))) {
      cli::cli_abort("Argument `x` is not a Numpy array!")
    }
    
    x <- py_to_r(x)
  }
  
  x <- process_rgb_array(x)
  
  p <- ggplot2::ggplot(x) +
    ggplot2::geom_tile(ggplot2::aes(width, height, fill = color)) +
    ggplot2::scale_y_reverse(expand = c(0,0)) +
    ggplot2::scale_x_continuous(position = "top", expand = c(0,0)) +
    ggplot2::scale_fill_identity() +
    ggplot2::coord_fixed()
  
  if (!axis) p <- p + ggplot2::theme(axis.line = ggplot2::element_blank(), 
                                     axis.ticks = ggplot2::element_blank(), 
                                     axis.text.x = ggplot2::element_blank(), 
                                     axis.text.y = ggplot2::element_blank(), 
                                     axis.title.x = ggplot2::element_blank(), 
                                     axis.title.y = ggplot2::element_blank())
  
  if (!is.null(title)) p <- p + ggplot2::ggtitle(title)
  
  return(p)
}

process_history <- function(history, model_name = "") {
  reticulate::py_to_r(history$history) |>
    as.data.frame() |>
    dplyr::mutate(epoch = 1:n()) |>
    tidyr::pivot_longer(-epoch) |>
    dplyr::mutate(set = ifelse(grepl("val", name), "Validation", "Train")) |>
    dplyr::mutate(metric = gsub("val_", "", name)) |>
    dplyr::mutate(model = model_name)
}

plot_total_history <- function(total_history) {
  total_history <- purrr::map_df(names(total_history), function(model_name) {
    total_history[[model_name]] |>
      dplyr::mutate(model = model_name)
  })
  
  total_history |>
    ggplot2::ggplot(ggplot2::aes(epoch, value, col = set)) +
    ggplot2::geom_line() +
    ggplot2::facet_grid(metric ~ model, scales = "free") +
    ggplot2::theme(legend.position = "bottom")
}
