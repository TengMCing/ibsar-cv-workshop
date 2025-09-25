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

# 10,000 training images stored as a 4D R array: (n_images, height, width, channels)
dim(cat_and_dog$x_train)
# 10,000 labels stored as an R vector (0 = cat, 1 = dog)
dim(cat_and_dog$y_train)
# Dataset is balanced
table(cat_and_dog$y_train)


# Load testing data -------------------------------------------------------

cat_and_dog_test <- readRDS(here::here("data/cat0_and_dog1_test.rds"))

# Convert array to tensor -------------------------------------------------

x <- torch$tensor(cat_and_dog$x_train, device = device, dtype = torch$float32)

# Standardize inputs:
# 1. Keeps gradients at a consistent scale across layers, avoiding exploding/vanishing gradients
# 2. Matches assumptions of many initialization schemes (zero mean, unit variance)
x_mean <- torch$mean(x)
x_sd <- torch$std(x)
x <- (x - x_mean) / x_sd

y <- torch$tensor(cat_and_dog$y_train, device = device, dtype = torch$float32)

x_test <- torch$tensor(cat_and_dog_test$x_test, device = device, dtype = torch$float32)
y_test <- torch$tensor(cat_and_dog_test$y_test, device = device, dtype = torch$float32)
x_test <- (x_test - x_mean) / x_sd


# Build a simple model ----------------------------------------------------

# Input layer with shape (32, 32, 3)
inputs <- keras$layers$Input(shape = dim(cat_and_dog$x_train)[-1])

# Flatten pixels into a 1D vector
flatten <- keras$layers$Flatten()(inputs)

# Hidden layer with 64 units and ReLU activation
hidden <- keras$layers$Dense(64L, activation = "relu")(flatten)

# Output layer with 1 unit and sigmoid activation for binary classification
outputs <- keras$layers$Dense(1L, activation = "sigmoid")(hidden)

# Define Keras model
keras_model <- keras$Model(inputs, outputs)

# Model summary: 4 layers, 196,737 parameters
# Calculation: (3072 * 64 + 64) + (64 * 1 + 1) = 196,737
keras_model$summary()


# Compile model -----------------------------------------------------------

# After defining the architecture, specify:
# - Loss function: how the model’s predictions are evaluated
# - Optimizer: how parameters are updated
# - Metrics: what to track during training
#
# Here we use:
# - Binary cross-entropy loss (standard for binary classification)
# - Stochastic Gradient Descent (SGD)
# - Accuracy as an evaluation metric
keras_model$compile(loss = "binary_crossentropy",
                    optimizer = "sgd",
                    metrics = list("accuracy"))


# Train model -------------------------------------------------------------

# Get predictions (returns a NumPy array)
keras_model$predict(x)

# Evaluate performance on training data
# High loss and ~50% accuracy indicate random guessing
keras_model$evaluate(x, y)

# Train the model using `fit()`
# - Batch size: number of samples per update.
#   Larger batches yield more stable gradient estimates and run faster,
#   but require more memory.
#   Smaller batches use less memory, produce noisier gradients,
#   and can sometimes escape local minima.
#   In practice, batch size is increased until memory is exhausted.
# - Epochs: number of passes over the full dataset.
train_history <- keras_model$fit(x, y, 
                                 batch_size = 512L, 
                                 epochs = 100L,
                                 validation_data = list(x_test, y_test))

total_history <- process_history(train_history, "base_64")
plot_total_history(total_history)

# Predictions and performance on train/test sets
train_pred <- tibble(y = factor(cat_and_dog$y_train),
                     base_64_prob = py_to_r(keras_model$predict(x))[, 1],
                     base_64_class = factor(round(base_64_prob)))

test_pred <- tibble(y = factor(cat_and_dog_test$y_test),
                    base_64_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    base_64_class = factor(round(base_64_prob)))

yardstick::accuracy(y, base_64_class, data = train_pred)
yardstick::accuracy(y, base_64_class, data = test_pred)


# Model with dropout ------------------------------------------------------

# Add a dropout layer after the hidden layer
# Randomly drops 30% of activations during training,
# forcing the model to learn more general features.
inputs <- keras$layers$Input(shape = dim(cat_and_dog$x_train)[-1])
flatten <- keras$layers$Flatten()(inputs)
hidden <- keras$layers$Dense(64L, activation = "relu")(flatten)
dropout <- keras$layers$Dropout(0.3)(hidden)
outputs <- keras$layers$Dense(1L, activation = "sigmoid")(dropout)

keras_model <- keras$Model(inputs, outputs)

# Dropout does not add trainable parameters
keras_model$summary()

keras_model$compile(loss = "binary_crossentropy",
                    optimizer = "sgd",
                    metrics = list("accuracy"))

train_history <- keras_model$fit(x, y, 
                                 batch_size = 512L, 
                                 epochs = 100L,
                                 validation_data = list(x_test, y_test))

total_history <- process_history(train_history, "dropout_64") |>
  bind_rows(total_history)

# Overfitting should be less severe
plot_total_history(total_history)

# Predictions and performance
train_pred <- mutate(train_pred,
                     dropout_64_prob = py_to_r(keras_model$predict(x))[, 1],
                     dropout_64_class = factor(round(dropout_64_prob)))

test_pred <- mutate(test_pred,
                    dropout_64_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    dropout_64_class = factor(round(dropout_64_prob)))

yardstick::accuracy(y, dropout_64_class, data = train_pred)
yardstick::accuracy(y, dropout_64_class, data = test_pred)


# Increase the number of units --------------------------------------------

# Increase hidden layer size to 256 units
inputs <- keras$layers$Input(shape = dim(cat_and_dog$x_train)[-1])
flatten <- keras$layers$Flatten()(inputs)
hidden <- keras$layers$Dense(256L, activation = "relu")(flatten)
dropout <- keras$layers$Dropout(0.3)(hidden)
outputs <- keras$layers$Dense(1L, activation = "sigmoid")(dropout)
keras_model <- keras$Model(inputs, outputs)

keras_model$summary()

keras_model$compile(loss = "binary_crossentropy",
                    optimizer = "sgd",
                    metrics = list("accuracy"))

train_history <- keras_model$fit(x, y, 
                                 batch_size = 512L, 
                                 epochs = 100L,
                                 validation_data = list(x_test, y_test))

total_history <- process_history(train_history, "dropout_256") |>
  bind_rows(total_history)

# Increasing the hidden layer size allows the model to capture more 
# complex patterns.
plot_total_history(total_history)

train_pred <- mutate(train_pred,
                     dropout_256_prob = py_to_r(keras_model$predict(x))[, 1],
                     dropout_256_class = factor(round(dropout_256_prob)))

test_pred <- mutate(test_pred,
                    dropout_256_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    dropout_256_class = factor(round(dropout_256_prob)))

yardstick::accuracy(y, dropout_256_class, data = train_pred)
yardstick::accuracy(y, dropout_256_class, data = test_pred)


# Increase the number of layers -------------------------------------------

# Two hidden layers, each with 256 units and dropout
inputs <- keras$layers$Input(shape = dim(cat_and_dog$x_train)[-1])
flatten <- keras$layers$Flatten()(inputs)
hidden <- keras$layers$Dense(256L, activation = "relu")(flatten)
dropout <- keras$layers$Dropout(0.3)(hidden)
hidden_2 <- keras$layers$Dense(256L, activation = "relu")(dropout)
dropout_2 <- keras$layers$Dropout(0.3)(hidden_2)
outputs <- keras$layers$Dense(1L, activation = "sigmoid")(dropout_2)
keras_model <- keras$Model(inputs, outputs)

keras_model$summary()

keras_model$compile(loss = "binary_crossentropy",
                    optimizer = "sgd",
                    metrics = list("accuracy"))

train_history <- keras_model$fit(x, y, 
                                 batch_size = 512L, 
                                 epochs = 100L,
                                 validation_data = list(x_test, y_test))

total_history <- process_history(train_history, "dropout_256_256") |>
  bind_rows(total_history)

# Adding a second hidden layer increases model depth, 
# enabling it to learn hierarchical features,
# though this also raises capacity and potential overfitting.
plot_total_history(total_history)

train_pred <- mutate(train_pred,
                     dropout_256_256_prob = py_to_r(keras_model$predict(x))[, 1],
                     dropout_256_256_class = factor(round(dropout_256_256_prob)))

test_pred <- mutate(test_pred,
                    dropout_256_256_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    dropout_256_256_class = factor(round(dropout_256_256_prob)))

yardstick::accuracy(y, dropout_256_256_class, data = train_pred)
yardstick::accuracy(y, dropout_256_256_class, data = test_pred)


# Three hidden layers -----------------------------------------------------

# Three hidden layers, each with 256 units and dropout
inputs <- keras$layers$Input(shape = dim(cat_and_dog$x_train)[-1])
flatten <- keras$layers$Flatten()(inputs)
hidden <- keras$layers$Dense(256L, activation = "relu")(flatten)
dropout <- keras$layers$Dropout(0.3)(hidden)
hidden_2 <- keras$layers$Dense(256L, activation = "relu")(dropout)
dropout_2 <- keras$layers$Dropout(0.3)(hidden_2)
hidden_3 <- keras$layers$Dense(256L, activation = "relu")(dropout_2)
dropout_3 <- keras$layers$Dropout(0.3)(hidden_3)
outputs <- keras$layers$Dense(1L, activation = "sigmoid")(dropout_3)
keras_model <- keras$Model(inputs, outputs)

keras_model$summary()

keras_model$compile(loss = "binary_crossentropy",
                    optimizer = "sgd",
                    metrics = list("accuracy"))

train_history <- keras_model$fit(x, y, 
                                 batch_size = 512L, 
                                 epochs = 100L,
                                 validation_data = list(x_test, y_test))

total_history <- process_history(train_history, "dropout_256_256_256") |>
  bind_rows(total_history)

# With three hidden layers, the model can learn richer representations,
# but training may become harder
plot_total_history(total_history)

train_pred <- mutate(train_pred,
                     dropout_256_256_256_prob = py_to_r(keras_model$predict(x))[, 1],
                     dropout_256_256_256_class = factor(round(dropout_256_256_256_prob)))

test_pred <- mutate(test_pred,
                    dropout_256_256_256_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    dropout_256_256_256_class = factor(round(dropout_256_256_256_prob)))

yardstick::accuracy(y, dropout_256_256_256_class, data = train_pred)
yardstick::accuracy(y, dropout_256_256_256_class, data = test_pred)


# Final comparison --------------------------------------------------------

# Overall, the performance of these models should be quite similar.
# If you retrain them many times, the more complex models will generally
# outperform the simpler ones on average, provided overfitting is controlled.
# That said, neural networks are not guaranteed to outperform other flexible
# methods such as random forests. As we know in model fitting, the more
# meaningful constraints you impose, the better you can estimate coefficients
# and make predictions. Pure neural networks are extremely flexible and impose
# almost no constraints, which is why in many Kaggle competitions they are
# often outperformed by methods like boosting trees.
test_pred |>
  pivot_longer(contains("class")) |>
  group_by(name) |>
  yardstick::accuracy(y, value)

bind_rows(train_pred |> mutate(set = "train"),
          test_pred  |> mutate(set = "test")) |>
  pivot_longer(contains("prob")) |>
  group_by(set, name) |>
  yardstick::roc_curve(y, value, event_level = "second") |>
  ggplot() +
  geom_line(aes(1 - specificity, sensitivity, col = name)) +
  geom_abline(slope = 1, linetype = 3) +
  facet_wrap(~set) +
  theme_light() +
  coord_fixed()
