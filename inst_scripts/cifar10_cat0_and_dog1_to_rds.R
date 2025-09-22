library(reticulate)
library(scrubwren)

Sys.setenv(KERAS_BACKEND = "torch")
use_condaenv("ibsar-cv-workshop")
keras <- import("keras", convert = FALSE)

py_tuple_unpack(c(c(x_train, y_train),
                  c(x_test, y_test)), 
                keras$datasets$cifar10$load_data())

y_train <- y_train$reshape(y_train$shape[0])
y_test <- y_test$reshape(y_test$shape[0])

dog_cat_train_idx <- y_train == 3 | y_train == 5 
dog_cat_test_idx <- y_test == 3 | y_test == 5

x_train <- x_train[dog_cat_train_idx]
y_train <- y_train[dog_cat_train_idx]

x_test <- x_test[dog_cat_test_idx]
y_test <- y_test[dog_cat_test_idx]

x_train <- py_to_r(x_train)
y_train <- py_to_r(y_train)
y_train <- ifelse(y_train == 3, 0L, 1L)

x_test <- py_to_r(x_test)
y_test <- py_to_r(y_test)
y_test <- ifelse(y_test == 3, 0L, 1L)

list(x_train = x_train,
     y_train = y_train) |>
  saveRDS(here::here("data/cat0_and_dog1_train.rds"))

list(x_test = x_test,
     y_test = y_test) |>
  saveRDS(here::here("data/cat0_and_dog1_test.rds"))


# 
# 
# x_train_array <- x_train |> py_to_r()
# 
# first_channel <- x_train_array[3, , , 1]
# second_channel <- x_train_array[3, , , 2]
# third_channel <- x_train_array[3, , , 3]
# 
# process_channel <- function(x, channel_name) {
#   x |>
#     as.data.frame() |>
#     mutate(height = 1:32) |>
#     pivot_longer(V1:V32) |>
#     mutate(name = gsub("V", "", name)) |>
#     mutate(name = as.integer(name)) |>
#     rename(width = name) |>
#     mutate(channel = channel_name)
# }
# 
# bind_rows(process_channel(first_channel, "R"),
#           process_channel(second_channel, "G"),
#           process_channel(third_channel, "B")) |>
#   group_by(height, width) |>
#   summarise(color = rgb(value[channel == "R"], 
#                         value[channel == "G"], 
#                         value[channel == "B"], 
#                         maxColorValue = 255)) |>
#   ggplot() +
#   geom_tile(aes(width, height, fill = color)) +
#   scale_y_reverse(expand = c(0,0)) +
#   scale_x_continuous(position = "top", expand = c(0,0)) +
#   scale_fill_identity() +
#   coord_fixed()
