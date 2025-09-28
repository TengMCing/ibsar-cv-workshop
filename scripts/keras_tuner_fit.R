# Setup -------------------------------------------------------------------

library(reticulate)
library(scrubwren)
library(tidyverse)

source(here::here("scripts/utils.R"))

Sys.setenv(KERAS_BACKEND = "torch")

use_condaenv("ibsar-cv-workshop")

keras <- import("keras", convert = FALSE)
torch <- import("torch", convert = FALSE)

device <- torch$accelerator$current_accelerator()

# Load training data ------------------------------------------------------

cat_and_dog <- readRDS(here::here("data/cat0_and_dog1_train.rds"))

# Convert arrays to tensors -----------------------------------------------

x <- torch$tensor(cat_and_dog$x_train, device = device, dtype = torch$float32)

x_mean <- torch$mean(x)
x_sd <- torch$std(x)
x <- (x - x_mean) / x_sd

y <- torch$tensor(cat_and_dog$y_train, device = device, dtype = torch$float32)

# Initialize tuner --------------------------------------------------------

keras_tuner <- import("keras_tuner", convert = FALSE)

# Build model -------------------------------------------------------------


HYPER_MODEL <- py_class("HYPER_MODEL", inherit = keras_tuner$HyperModel,
                        build = function(self, hp) {
                          # Define hyperparameters
                          layers <- hp$Int("layers", min_value = 1L, max_value = 3L, step = 1L)
                          units <- hp$Choice("units", list(8L, 16L, 32L, 64L, 128L, 256L))
                          dropout <- hp$Boolean("dropout")
                          dropout_rate <- hp$Float("dropout_rate", min_value = 0.1, max_value = 0.6, step = 0.1)
                          lr <- hp$Float("lr", min_value = 1e-4, max_value = 1e-2, sampling = "log")
                          
                          # Build model based on hyperparameters
                          inputs <- keras$layers$Input(dim(cat_and_dog$x_train)[-1])
                          l <- keras$layers$Flatten()(inputs)
                          
                          py_for(i ~ py_builtins$range(layers), {
                            l <- keras$layers$Dense(units, activation = "relu")(l)
                            if (py_to_r(dropout)) {
                              l <- keras$layers$Dropout(dropout_rate)(l)
                            }
                          })
                          
                          outputs <- keras$layers$Dense(1, activation = "sigmoid")(l)
                          
                          keras_model <- keras$Model(inputs, outputs)
                          
                          keras_model$compile(loss = "binary_crossentropy",
                                              optimizer = keras$optimizers$Adam(learning_rate = lr),
                                              metrics = list("accuracy"))
                          
                          keras_model$summary()
                          return(keras_model)
                        },
                        
                        fit = function(self, hp, model, x, y, ...) {
                          # Define model training hyperparameters
                          epochs <- hp$Int("epochs", min_value = 10L, max_value = 100L, step = 10L)
                          model$fit(x, y, 
                                    epochs = epochs, 
                                    ...)
                        })


# Configure search strategy -----------------------------------------------

tuner <- keras_tuner$BayesianOptimization(
  hypermodel = HYPER_MODEL(),           # Hyper model instance
  objective = "val_accuracy",           # Metric to optimize
  max_trials = 10L,                     # Total number of trials
  num_initial_points = 3L,              # Initial random samples
  executions_per_trial = 2L,            # Repeated runs per trial
  overwrite = TRUE,                     # Overwrite old results
  directory = "keras_tuner",            # Base output directory
  project_name = "cat_and_dog_bayesian" # Subfolder name
)

tuner$search_space_summary()


# Run hyperparameter search -----------------------------------------------

tuner$search(x, y, batch_size = 512L, validation_split = 0.2)

tuner$results_summary()


# Extract best configuration ----------------------------------------------

# Best hyperparameters
tuner$get_best_hyperparameters()[0]

# Best checkpoint
tuner$get_best_models()

# Retrain best model ------------------------------------------------------

hyper_model <- HYPER_MODEL()
best_hp <- tuner$get_best_hyperparameters()[0]
best_model <- hyper_model$build(best_hp)
hyper_model$fit(best_hp, 
                best_model, 
                x, 
                y,
                batch_size = 512L)

