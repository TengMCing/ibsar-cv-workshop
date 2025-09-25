# Load training data ------------------------------------------------------

library(tidyverse)

cat_and_dog <- readRDS(here::here("data/cat0_and_dog1_train.rds"))

# 10,000 training images stored as a 4D R array: (n_images, height, width, RGB channels)
dim(cat_and_dog$x_train)
# 10,000 labels stored as an R vector (0 = cat, 1 = dog)
dim(cat_and_dog$y_train)
# The dataset is balanced
table(cat_and_dog$y_train)

# Image indexing ----------------------------------------------------------

# Access the first image (pixel values range from 0 to 255)
cat_and_dog$x_train[1, , , ]

# Access the red channel of the top-left pixel in the first image
cat_and_dog$x_train[1, 1, 1, 1]

# Visualize the images ----------------------------------------------------

source(here::here("scripts/utils.R"))
map(1:12, function(i) {
  plot_rgb(cat_and_dog$x_train[i, , , ], 
           title = cat_and_dog$y_train[i], 
           axis = FALSE) 
}) |>
  patchwork::wrap_plots()


# Transform X -------------------------------------------------------------

# For standard machine learning/statistical modelling, we first need to
# transform the data into a matrix form.

# One approach is to flatten each image into a single row, with columns
# representing pixel values across all channels and positions.

x <- cat_and_dog$x_train
dim(x) <- c(dim(cat_and_dog$x_train)[1], prod(dim(cat_and_dog$x_train)[2:4]))

# This gives a matrix with 10,000 rows and 3,072 columns.
# Each pixel value across the three channels is treated as a feature.
x <- as.data.frame(x)
dim(x)


# Apply principal component analysis --------------------------------------

# With 3,072 predictors, even logistic regression could be slow.
# We reduce the dimension using PCA.

if (!file.exists(here::here("cached_data/cat_and_dog_train_pca.rds"))) {
  train_pca <- prcomp(x, scale. = TRUE, center = TRUE, rank. = 100L)  
  saveRDS(train_pca, here::here("cached_data/cat_and_dog_train_pca.rds"))
} else {
  train_pca <- readRDS(here::here("cached_data/cat_and_dog_train_pca.rds"))
}

# The first 100 PCs explain ~91% of the total variance
((cumsum(train_pca$sdev^2)) / ncol(x))[100]

x_100 <- predict(train_pca, x)[, 1:100] |> as.data.frame()
train_data <- mutate(x_100, y = factor(cat_and_dog$y_train))


# Logistic regression -----------------------------------------------------

glm_model <- glm(y ~ ., data = train_data, family = binomial)

train_pred <- data.frame(y = factor(cat_and_dog$y_train),
                         glm_prob = predict(glm_model, train_data, type = "response"),
                         glm_class = factor(round(glm_prob)))

# Inspect the first 12 predictions against the actual labels
map(1:12, function(i) {
  plot_rgb(cat_and_dog$x_train[i, , , ], 
           title = glue::glue("Actual: {train_pred$y[i]}, Pred: {train_pred$glm_class[i]}"), 
           axis = FALSE) 
}) |>
  patchwork::wrap_plots()

# Training accuracy: ~61.6%
yardstick::accuracy(y, glm_class, data = train_pred)
# Confusion matrix
yardstick::conf_mat(y, glm_class, data = train_pred)

# Linear discriminant analysis --------------------------------------------

lda_model <- MASS::lda(y ~ ., train_data)
train_pred <- train_pred |>
  mutate(lda_prob = predict(lda_model, train_data)$posterior[, 2],
         lda_class = factor(round(lda_prob)))

# Training accuracy: ~61.5%
yardstick::accuracy(y, lda_class, data = train_pred)
yardstick::conf_mat(y, lda_class, data = train_pred)


# Quadratic discriminant analysis -----------------------------------------

qda_model <- MASS::qda(y ~ ., train_data)
train_pred <- train_pred |>
  mutate(qda_prob = predict(qda_model, train_data)$posterior[, 2],
         qda_class = factor(round(qda_prob)))

# Training accuracy: ~75.5%
# QDA is more flexible and fits the training data better
yardstick::accuracy(y, qda_class, data = train_pred)
yardstick::conf_mat(y, qda_class, data = train_pred)


# Random forest -----------------------------------------------------------

set.seed(10086)
rf_model <- randomForest::randomForest(y ~ ., train_data)
train_pred <- train_pred |>
  mutate(rf_prob = predict(rf_model, train_data, type = "prob")[, 2],
         rf_class = factor(round(rf_prob)))

# Training accuracy: 100%
# Random forest is very flexible and strongly overfits
yardstick::accuracy(y, rf_class, data = train_pred)
yardstick::conf_mat(y, rf_class, data = train_pred)


# Evaluate the models on test data ----------------------------------------

cat_and_dog_test <- readRDS(here::here("data/cat0_and_dog1_test.rds"))
x <- cat_and_dog_test$x_test
dim(x) <- c(dim(cat_and_dog_test$x_test)[1], prod(dim(cat_and_dog_test$x_test)[2:4]))
x <- as.data.frame(x)
x_100 <- predict(train_pca, x)[, 1:100] |> as.data.frame()
test_data <- mutate(x_100, y = factor(cat_and_dog_test$y_test))

test_pred <- data.frame(y = factor(test_data$y),
                        glm_prob = predict(glm_model, test_data, type = "response"),
                        lda_prob = predict(lda_model, test_data)$posterior[, 2],
                        qda_prob = predict(qda_model, test_data)$posterior[, 2],
                        rf_prob = predict(rf_model, test_data, type = "prob")[, 2]) |>
  mutate(glm_class = factor(round(glm_prob)),
         lda_class = factor(round(lda_prob)),
         qda_class = factor(round(qda_prob)),
         rf_class = factor(round(rf_prob)))

# glm and lda: similar train/test performance
# qda: slightly overfits but still best overall
# rf: severe overfitting
test_pred |>
  pivot_longer(glm_class:rf_class) |>
  group_by(name) |>
  yardstick::accuracy(y, value)

# ROC curves tell the same story:
# for vision tasks, traditional linear or inflexible models cannot
# effectively capture visual patterns.
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

# Takeaways:
# 1. Flexible models are needed for vision tasks, as the problem is too
#    complex for linear models to handle.
# 2. Flexible models can easily overfit, so regularization or parameter
#    control is necessary to balance fit and generalization.
# 3. Baseline models such as QDA and random forest achieve 63–66% accuracy;
#    our model should aim to outperform them.

bind_rows(train_pred |> mutate(set = "train"),
          test_pred  |> mutate(set = "test")) |>
  saveRDS(here::here("cached_data/cat_and_dog_ml_pred.rds"))