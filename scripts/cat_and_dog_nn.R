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

# Training set:
# - 10,000 images stored as a 4D array: (n_images, height, width, channels)
# - 10,000 labels stored as a vector (0 = cat, 1 = dog)
# - Dataset is balanced
dim(cat_and_dog$x_train)
dim(cat_and_dog$y_train)
table(cat_and_dog$y_train)


# Load testing data -------------------------------------------------------

cat_and_dog_test <- readRDS(here::here("data/cat0_and_dog1_test.rds"))


# Convert arrays to tensors ------------------------------------------------

x <- torch$tensor(cat_and_dog$x_train, device = device, dtype = torch$float32)

# Standardize inputs:
# - Ensures gradients remain well-scaled, avoiding exploding/vanishing issues
# - Matches assumptions of common initialization schemes (zero mean, unit variance)
x_mean <- torch$mean(x)
x_sd <- torch$std(x)
x <- (x - x_mean) / x_sd

y <- torch$tensor(cat_and_dog$y_train, device = device, dtype = torch$float32)

x_test <- torch$tensor(cat_and_dog_test$x_test, device = device, dtype = torch$float32)
y_test <- torch$tensor(cat_and_dog_test$y_test, device = device, dtype = torch$float32)
x_test <- (x_test - x_mean) / x_sd


# PCA preprocessing -------------------------------------------------------

# Flatten each image and compute 100 principal components
x_pca <- cat_and_dog$x_train
dim(x_pca) <- c(dim(x_pca)[1], prod(dim(x_pca)[2:4]))
x_pca <- as.data.frame(x_pca)

x_test_pca <- cat_and_dog_test$x_test
dim(x_test_pca) <- c(dim(x_test_pca)[1], prod(dim(x_test_pca)[2:4]))
x_test_pca <- as.data.frame(x_test_pca)

if (!file.exists(here::here("cached_data/cat_and_dog_train_pca.rds"))) {
  train_pca <- prcomp(x_pca, scale. = TRUE, center = TRUE, rank. = 100L)  
  saveRDS(train_pca, here::here("cached_data/cat_and_dog_train_pca.rds"))
} else {
  train_pca <- readRDS(here::here("cached_data/cat_and_dog_train_pca.rds"))
}

x_pca <- predict(train_pca, x_pca)[, 1:100]
x_test_pca <- predict(train_pca, x_test_pca)[, 1:100]

x_pca <- torch$tensor(x_pca, device = device, dtype = torch$float32)
x_pca_mean <- torch$mean(x_pca)
x_pca_sd <- torch$std(x_pca)
x_pca <- (x_pca - x_pca_mean) / x_pca_sd

x_test_pca <- torch$tensor(x_test_pca, device = device, dtype = torch$float32)
x_test_pca <- (x_test_pca - x_pca_mean) / x_pca_sd


# Baseline model ----------------------------------------------------------

# Architecture:
# - Input: (32, 32, 3)
# - Flatten to 1D vector
# - Dense hidden layer (64 units, ReLU)
# - Output: 1 unit, sigmoid activation (binary classification)
inputs <- keras$layers$Input(shape = dim(cat_and_dog$x_train)[-1])
flatten <- keras$layers$Flatten()(inputs)
hidden <- keras$layers$Dense(64L, activation = "relu")(flatten)
outputs <- keras$layers$Dense(1L, activation = "sigmoid")(hidden)
keras_model <- keras$Model(inputs, outputs)

# Parameter count: (3072 * 64 + 64) + (64 * 1 + 1) = 196,737
keras_model$summary()


# Compile model -----------------------------------------------------------

# Specify:
# - Loss: binary cross-entropy (standard for binary classification)
# - Optimizer: stochastic gradient descent (SGD)
# - Metric: accuracy
keras_model$compile(loss = "binary_crossentropy",
                    optimizer = "sgd",
                    metrics = list("accuracy"))


# Train model -------------------------------------------------------------

# Quick check before training
keras_model$predict(x)
keras_model$evaluate(x, y)  # ~50% accuracy = random guessing

# Fit model
# - Batch size: number of samples per update (trade-off between speed and noise)
# - Epochs: number of full passes through the dataset
train_history <- keras_model$fit(x, y, 
                                 batch_size = 512L, 
                                 epochs = 100L,
                                 validation_data = list(x_test, y_test))

total_history <- list()
total_history$base_64 <- process_history(train_history)
plot_total_history(total_history)

# Evaluate predictions on train and test sets
train_pred <- tibble(y = factor(cat_and_dog$y_train),
                     base_64_prob = py_to_r(keras_model$predict(x))[, 1],
                     base_64_class = factor(round(base_64_prob)))

test_pred <- tibble(y = factor(cat_and_dog_test$y_test),
                    base_64_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    base_64_class = factor(round(base_64_prob)))

yardstick::accuracy(y, base_64_class, data = train_pred)
yardstick::accuracy(y, base_64_class, data = test_pred)


# Model with dropout ------------------------------------------------------

# Add dropout after the hidden layer (rate = 0.3)
# - Randomly disables 30% of activations during training
# - Reduces overfitting by encouraging robust feature learning
inputs <- keras$layers$Input(shape = dim(cat_and_dog$x_train)[-1])
flatten <- keras$layers$Flatten()(inputs)
hidden <- keras$layers$Dense(64L, activation = "relu")(flatten)
dropout <- keras$layers$Dropout(0.3)(hidden)
outputs <- keras$layers$Dense(1L, activation = "sigmoid")(dropout)
keras_model <- keras$Model(inputs, outputs)

keras_model$summary()  # dropout adds no trainable parameters

keras_model$compile(loss = "binary_crossentropy",
                    optimizer = "sgd",
                    metrics = list("accuracy"))

train_history <- keras_model$fit(x, y, 
                                 batch_size = 512L, 
                                 epochs = 100L,
                                 validation_data = list(x_test, y_test))

total_history$dropout_64 <- process_history(train_history)
plot_total_history(total_history)

# Evaluate
train_pred <- mutate(train_pred,
                     dropout_64_prob = py_to_r(keras_model$predict(x))[, 1],
                     dropout_64_class = factor(round(dropout_64_prob)))

test_pred <- mutate(test_pred,
                    dropout_64_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    dropout_64_class = factor(round(dropout_64_prob)))

yardstick::accuracy(y, dropout_64_class, data = train_pred)
yardstick::accuracy(y, dropout_64_class, data = test_pred)


# Larger hidden layer -----------------------------------------------------

# Increase hidden size from 64 → 256 units
# - Higher capacity to capture complex patterns
# - Greater risk of overfitting
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

total_history$dropout_256 <- process_history(train_history)
plot_total_history(total_history)

train_pred <- mutate(train_pred,
                     dropout_256_prob = py_to_r(keras_model$predict(x))[, 1],
                     dropout_256_class = factor(round(dropout_256_prob)))

test_pred <- mutate(test_pred,
                    dropout_256_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    dropout_256_class = factor(round(dropout_256_prob)))

yardstick::accuracy(y, dropout_256_class, data = train_pred)
yardstick::accuracy(y, dropout_256_class, data = test_pred)


# Deeper model: 2 layers --------------------------------------------------

# Two hidden layers (256 units each, with dropout)
# - Increases model depth
# - Enables learning hierarchical features
# - Further increases risk of overfitting
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

total_history$dropout_256_256 <- process_history(train_history)
plot_total_history(total_history)

train_pred <- mutate(train_pred,
                     dropout_256_256_prob = py_to_r(keras_model$predict(x))[, 1],
                     dropout_256_256_class = factor(round(dropout_256_256_prob)))

test_pred <- mutate(test_pred,
                    dropout_256_256_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    dropout_256_256_class = factor(round(dropout_256_256_prob)))

yardstick::accuracy(y, dropout_256_256_class, data = train_pred)
yardstick::accuracy(y, dropout_256_256_class, data = test_pred)


# Deeper model: 3 layers --------------------------------------------------

# Three hidden layers (256 units each, with dropout)
# - Greater representational power
# - Harder to train and tune
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

total_history$dropout_256_256_256 <- process_history(train_history)
plot_total_history(total_history)

train_pred <- mutate(train_pred,
                     dropout_256_256_256_prob = py_to_r(keras_model$predict(x))[, 1],
                     dropout_256_256_256_class = factor(round(dropout_256_256_256_prob)))

test_pred <- mutate(test_pred,
                    dropout_256_256_256_prob = py_to_r(keras_model$predict(x_test))[, 1],
                    dropout_256_256_256_class = factor(round(dropout_256_256_256_prob)))

yardstick::accuracy(y, dropout_256_256_256_class, data = train_pred)
yardstick::accuracy(y, dropout_256_256_256_class, data = test_pred)


# PCA-based model ---------------------------------------------------------

# Train on top 100 PCs instead of raw pixels
# - Reduces dimensionality
# - Speeds up training
# - May converge more slowly or limit accuracy if information is lost
inputs <- keras$layers$Input(shape = list(100L))
hidden <- keras$layers$Dense(256L, activation = "relu")(inputs)
outputs <- keras$layers$Dense(1L, activation = "sigmoid")(hidden)
keras_model <- keras$Model(inputs, outputs)

keras_model$summary()
keras_model$compile(loss = "binary_crossentropy",
                    optimizer = "sgd",
                    metrics = list("accuracy"))

train_history <- keras_model$fit(x_pca, y, 
                                 batch_size = 512L, 
                                 epochs = 100L,
                                 validation_data = list(x_test_pca, y_test))

total_history$pca_256 <- process_history(train_history)
plot_total_history(total_history)

train_pred <- mutate(train_pred,
                     pca_256_prob = py_to_r(keras_model$predict(x_pca))[, 1],
                     pca_256_class = factor(round(pca_256_prob)))

test_pred <- mutate(test_pred,
                    pca_256_prob = py_to_r(keras_model$predict(x_test_pca))[, 1],
                    pca_256_class = factor(round(pca_256_prob)))

yardstick::accuracy(y, pca_256_class, data = train_pred)
yardstick::accuracy(y, pca_256_class, data = test_pred)


# Final comparison --------------------------------------------------------

# Overall:
# - All variants achieve broadly similar performance
# - Larger and deeper models tend to outperform smaller ones on average,
#   provided overfitting is controlled
# - Neural nets are not always superior to classical ML methods 
#   (e.g. random forests, boosting)
# - Models with structural constraints often generalize better than very
#   flexible unconstrained networks
#
# Hyperparameter tuning (learning rate, batch size, epochs, etc.) is critical
# for extracting the last few percentage points of accuracy. Optimal settings
# are both data- and architecture-dependent.
test_pred |>
  pivot_longer(contains("class")) |>
  group_by(name) |>
  yardstick::accuracy(y, value) |>
  arrange(desc(.estimate))

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

bind_rows(train_pred |> mutate(set = "train"),
          test_pred  |> mutate(set = "test")) |>
  saveRDS(here::here("cached_data/cat_and_dog_nn_pred.rds"))


# Compared with ML --------------------------------------------------------

# Compare neural nets with alternative ML baselines
ml_pred <- readRDS(here::here("cached_data/cat_and_dog_ml_pred.rds"))

test_pred |>
  bind_rows(filter(ml_pred, set == "test")) |>
  pivot_longer(contains("class")) |>
  group_by(name) |>
  yardstick::accuracy(y, value) |>
  arrange(desc(.estimate))

bind_rows(train_pred |> mutate(set = "train"),
          test_pred  |> mutate(set = "test")) |>
  bind_rows(ml_pred) |>
  pivot_longer(contains("prob")) |>
  group_by(set, name) |>
  yardstick::roc_curve(y, value, event_level = "second") |>
  ggplot() +
  geom_line(aes(1 - specificity, sensitivity, col = name)) +
  geom_abline(slope = 1, linetype = 3) +
  facet_wrap(~set) +
  theme_light() +
  coord_fixed()
