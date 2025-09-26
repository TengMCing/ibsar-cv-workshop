
# Install R packages for this workshop ------------------------------------

install.packages(c("tidyverse", "reticulate", "patchwork", 
                   "cli", "glue", "yardstick", "randomForest",
                   "plotly"))
remotes::install_github("TengMCing/scrubwren")


# Install Conda -----------------------------------------------------------

# Install `miniconda`
# Skip if `conda` exists in the system
if (is.null(reticulate:::find_conda()[[1]])) {
  reticulate::install_miniconda()
}

# You could use `options(reticulate.conda_binary = "/path/to/conda")` to
# force `reticulate` to use a particular `conda` binary

# Install Python libraries ------------------------------------------------

# Create an environment
if (reticulate::condaenv_exists("ibsar-cv-workshop")) {
  reticulate::conda_remove("ibsar-cv-workshop")
}

reticulate::conda_create("ibsar-cv-workshop",
                         python_version = "3.11.9")

# Install libraries
reticulate::conda_install("ibsar-cv-workshop",
                          pip = TRUE,
                          packages = c("torch", "torchvision", 
                                       "torchaudio", "keras"))
