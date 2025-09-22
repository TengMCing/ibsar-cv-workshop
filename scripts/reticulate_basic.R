library(reticulate)
library(scrubwren)

# Use the conda environment -----------------------------------------------
use_condaenv("ibsar-cv-workshop")

# Convert R objects to Python ---------------------------------------------

# R integer scalar -> Python int
r_to_py(1L)

# R double scalar -> Python float
r_to_py(1.0)

# R string scalar -> Python str
r_to_py("abc")

# R boolean scalar -> Python bool
r_to_py(TRUE)

# R NULL -> Python None
r_to_py(NULL)

# R NaN -> Python nan
r_to_py(NaN)

# R NA -> Python (depends on internal R representation)
r_to_py(NA)
r_to_py(NA_integer_)
r_to_py(NA_real_)       # Always nan
r_to_py(NA_character_)  # Always "NA"

# R vector -> Python list
r_to_py(c(1, 2, 3))

# R list -> Python list
r_to_py(list(1, 2, 3))

# R named list -> Python dict
r_to_py(list(a = 1, b = 2))

# R array -> NumPy array
r_to_py(matrix(1:10, ncol = 2))

# R function -> Python proxy (still executed in R)
r_to_py(function(x) print(x))

# Convert Python objects back to R ----------------------------------------

# Python list/tuple -> R vector or list (simplified when possible)
py_to_r(py_builtins$list(list(1, 2, 3)))
py_to_r(py_builtins$list(list(1, 2, list(3, 4))))

py_to_r(py_builtins$tuple(c(1, 2, 3)))
py_to_r(py_builtins$tuple(list(1, 2, list(3, 4))))

# Python dict -> R named list
py_to_r(py_builtins$dict(list(a = 1, b = 2)))

# Python str -> R string
py_to_r(py_builtins$str("123"))

# Python bool -> R logical
py_to_r(py_builtins$bool(TRUE))

# Some Python objects stay as Python (e.g., sets)
py_to_r(py_builtins$set(c(1,2,3))) |> class()

# Import Python modules ---------------------------------------------------

# With convert = TRUE, returned Python objects are automatically converted
# to R. For robustness, it’s usually better to keep convert = FALSE.
np <- import("numpy", convert = FALSE)

# Working with NumPy arrays -----------------------------------------------

x <- np$array(matrix(1:10, ncol = 2))

# Basic operations (similar to R)
x + x
x - x
x * x
x^2
t(x) %*% x
np$sum(x)

# Array metadata
x$shape   # Dimensions
x$dtype   # Data type

# Indexing (Python is 0-based)
x[0]              # First element along the first dimension
x[0, 0]           # First element of the first dimension, first element of the second dimension
x[, 0]            # First element along the second dimension
x[, -1]           # Last element along the second dimension
x[0:2]            # First two elements along the first dimension
x[0:-1]           # All elements along the first dimension except the last
x[0] <- c(1L, 2L) # Replace the first element along the first dimension with (1, 2)

# Convert back to R
py_to_r(x)


# Working with Torch tensors ----------------------------------------------

# Torch tensors are similar to NumPy arrays, but they can be stored on different devices (CPU or GPU)
torch <- import("torch", convert = FALSE)

# A tensor is the fundamental unit in a computation graph.
# If required, it can store gradients for backpropagation.
x <- torch$tensor(matrix(1:10, ncol = 2))

# Check properties of the tensor (currently on CPU)
x$dtype   # Data type
x$device  # Device where the tensor is stored
x$shape   # Shape (dimensions)

# If a GPU is available, move the tensor to that device
device <- torch$accelerator$current_accelerator(check_available = TRUE)
x_copy <- x$to(device)
x_copy$device  # Verify the tensor is now on GPU

# Or create a tensor directly on the GPU
y <- torch$tensor(matrix(1:10, ncol = 2), device = device)
y

# Converting tensors back to R won't work
py_to_r(x)

# Safest approach: make a copy via NumPy without affecting the computation graph
# Note: This can be cumbersome and inefficient if done repeatedly
# This is also why we recommend keeping Python objects as Python objects
# as much as possible in your machine learning pipeline
y_np <- y$detach()$cpu()$numpy()$copy()
y_np |> class()

py_to_r(y_np)
