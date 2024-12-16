library(caret)
library(DT)
library(Metrics) 
library(dplyr)
library(furrr)
library(progressr)
library(ggplot2)
library(progressr)
library(e1071)

type_mode <- tolower(names(sort(table(wine_type_location$type), decreasing = TRUE))[1])
location_mode <- tolower(names(sort(table(wine_type_location$location), decreasing = TRUE))[1])

test_data = read.csv(
  './Data/Wine_Test_Set.csv',
  header = TRUE
)

train_data = read.csv(
  './Data/Wine_Train.csv',
  header = TRUE
)

wine_type_location <- read.csv(
  './Data/Wine_Types_and_Locations.csv',
  header = TRUE
) 

wine_type_location <- wine_type_location |> mutate(
    type = ifelse(is.na(type) | type == "", type_mode, tolower(type)),
    location = ifelse(is.na(location) | location == "", location_mode, tools::toTitleCase(location))
  ) |>
  mutate(location = ifelse(location == "Califormia", "California", location)) |>
  mutate(
    type = factor(type),
    location = factor(location)
  )


train_data <- left_join(train_data, wine_type_location, by = "ID")

str(train_data)
type_dummies <- model.matrix(~ type - 1, data = train_data)
location_dummies <- model.matrix(~ location - 1, data = train_data)

train_data_encoded <- cbind(
  train_data,
  type_dummies,
  location_dummies
) |> dplyr::select(-all_of(c("ID")))


head(train_data_encoded)
# Parallel Execution Setup
n_cores <- detectCores()
max_memory <- 16 * 1024 ^ 3
plan(multisession, workers = max(1, n_cores - 2))
options(future.globals.maxSize = max_memory)


evaluateModels <- function(target, features, data_train) {
  train_index <- createDataPartition(data_train[[target]], p = 0.7, list = FALSE)
  train_data <- data_train[train_index, ]
  test_data <- data_train[-train_index, ]
  
  train_x <- train_data[, features, drop = FALSE]
  test_x <- test_data[, features, drop = FALSE]
  train_labels <- factor(train_data[[target]])
  test_labels <- factor(test_data[[target]], levels = levels(train_labels))
  
  ## NB
  formulaNb <- paste(target, "~", paste(features, collapse = " + "))
  nbModel <- naiveBayes(as.formula(formulaNb), data = train_data)
  
  nbPredictions <- predict(nbModel, test_data)
  nbMatrix <- confusionMatrix(factor(nbPredictions, levels = levels(test_labels)), test_labels, mode = "everything")
  
  return(
    list(
      nb_formula = formulaNb,
      nb_accuracy = nbMatrix$overall["Accuracy"],
      nb_sensitivity = mean(nbMatrix$byClass[, "Sensitivity"], na.rm = TRUE),
      nb_specificity = mean(nbMatrix$byClass[, "Specificity"], na.rm = TRUE)
    )
  )
}

evaluateRegression <- function(target, features, data) {
  model <- paste(target, " ~ ", paste(features, collapse = " + "))
  train_indices <- createDataPartition(data[[target]], p = 0.75, list = FALSE)
  train_data <- data[train_indices, ]
  validation_data <- data[-train_indices, ]
  
  fit <- lm(as.formula(model), data = train_data)
  predictions <- predict(fit, newdata = validation_data)
  actual_values <- validation_data[[target]]
  
  residuals <- actual_values - predictions
  mae_value <- mean(abs(residuals))
  
  # Metrics::mae(validation_data[[target]], predictions)
  # mean(abs(residuals))
  # print(validation_data[[target]])
  
  
  return(
    list(
      features = model,
      AIC = AIC(fit),
      BIC = BIC(fit),
      MAE = mae_value,
      RMSE = sqrt(mean(residuals ^ 2)),
      ADJUSTED_R2 = summary(fit)$adj.r.squared,
      R2_VALIDATION = 1 - sum(residuals ^ 2) / sum((actual_values - mean(actual_values)) ^ 2)
    )
  )
}

permuteModels <- function(target, data) {
  all_features <- colnames(data)[!colnames(data) %in% target]
  total_features <- length(all_features)
  
  all_combinations <- unlist(lapply(1:total_features, function(size) {
    combn(all_features, size, simplify = FALSE)
  }), recursive = FALSE)
  
  chunk_size <- ceiling(length(all_combinations) / future::nbrOfWorkers())
  chunks <- split(all_combinations, ceiling(seq_along(all_combinations) / chunk_size))
  
  total_steps <- length(all_combinations)
  p <- progressr::progressor(steps = total_steps)
  
  print(paste("Will Run --- ", total_steps, " --- Combinations", collapse = "\n"))
  results <- future_map_dfr(seq_along(chunks), ~ {
    chunk <- chunks[[.x]]
    results_list <- list()
    
    for (i in seq_along(chunk)) {
      features_subset <- chunk[[i]]
      # message(paste("Starting: ", target, " ~ ", paste(features_subset, collapse = " + "), collapse = "\n"))
      knn_nb_results <- data.frame(evaluateModels(target, features_subset, data))
      regression_results <- data.frame(evaluateRegression(target, features_subset, data))
      
      #regression_results <- regression_results[regression_results$adjusted_r2 > 0.7, , drop = FALSE]
      combined_results <- data.frame(
        iteration = paste(.x, "-", i),
        features = paste(features_subset, collapse = ", "),
        nb_accuracy = knn_nb_results$nb_accuracy,
        nb_sensitivity = knn_nb_results$nb_sensitivity,
        nb_specificity = knn_nb_results$nb_specificity,
        AIC = regression_results$AIC,
        BIC = regression_results$BIC,
        MAE = regression_results$MAE,
        RMSE = regression_results$RMSE,
        ADJUSTED_R2 = regression_results$ADJUSTED_R2,
        R2_VALIDATION = regression_results$R2_VALIDATION
      )
      
      results_list[[i]] <- combined_results
      p(paste("Chunk", .x, "complete"))
    }
    
    
    do.call(rbind, results_list)
  }, .options = furrr_options(seed = TRUE))
  
  return(results)
}

runTests <- function(target, data) {
  progressr::with_progress({
    all_results <- permuteModels(target, data)
  })
  
  return(all_results |>
           arrange(MAE) |>
           mutate(across(where(is.numeric), ~ round(.x, 6))))
}

## Select best feature set based without categorical
results <- runTests("quality", train_data |> dplyr::select(-ID, -type, -location))

datatable(
  results,
  caption = "Performance Comparison for NB, KNN, and Regression Models",
  options = list(pageLength = 20, autoWidth = TRUE),
  rownames = FALSE
)

## Visual analysis for some of the best models
test_polynomials <- function(data, base_formula, features) {
  results <- data.frame()
  
  for (feature in features) {
    formula_quad <- paste(base_formula, " + I(", feature, "^2)", sep = "")
    formula_cubic <- paste(base_formula, "+ I(", feature, "^2) + I(", feature, "^3)", sep = "")
    
    fit_quad <- lm(as.formula(formula_quad), data = data)
    fit_cubic <- lm(as.formula(formula_cubic), data = data)
    
    mse_quad <- mean(residuals(fit_quad)^2)
    adj_r2_quad <- summary(fit_quad)$adj.r.squared
    
    mse_cubic <- mean(residuals(fit_cubic)^2)
    adj_r2_cubic <- summary(fit_cubic)$adj.r.squared
    
    analysis <- data.frame(
      Feature = feature,
      AIC_QUAD = AIC(fit_quad),
      AIC_CUBIC = AIC(fit_cubic),
      MSE_QUAD = mse_quad,
      MSE_CUBIC = mse_cubic,
      ADJ_R2_QUAD = adj_r2_quad,
      ADJ_R2_CUBIC = adj_r2_cubic,
      FORMULA_CUBIC = formula_cubic,
      FORMULA_QUAD = formula_quad
    )
    results <- rbind(results, analysis)
  }
  
  return(results)
}


fitO <- lm(quality ~ ., data = train_data)
summary(fitO)
vif(fitO)

train_index <- createDataPartition(train_data[["quality"]], p = 0.7, list = FALSE)
train_set <- train_data[train_index, ]
test_set <- train_data[-train_index, ]

train_set$quality <- factor(train_set$quality)
test_set$quality <- factor(test_set$quality, levels = levels(train_set$quality))
nbModel <- naiveBayes(quality ~ ., data = train_set)

predictions <- predict(nbModel, test_set)
predictions <- factor(predictions, levels = levels(train_set$quality))
actual <- factor(test_set[["quality"]], levels = levels(train_set$quality))

conf_matrix <- confusionMatrix(predictions, actual, mode = "everything")
print(conf_matrix)

###

fitO <- lm(quality ~ ., data = train_data)
summary(fitO)
vif(fitO)

train_index <- createDataPartition(train_data[["quality"]], p = 0.7, list = FALSE)
train_set <- train_data[train_index, ]
test_set <- train_data[-train_index, ]

train_set$quality <- factor(train_set$quality)
test_set$quality <- factor(test_set$quality, levels = levels(train_set$quality))
nbModel <- naiveBayes(quality ~ ., data = train_set)

predictions <- predict(nbModel, test_set)
predictions <- factor(predictions, levels = levels(train_set$quality))
actual <- factor(test_set[["quality"]], levels = levels(train_set$quality))

conf_matrix <- confusionMatrix(predictions, actual, mode = "everything")
print(conf_matrix)

# 
# wines$qlevel <- cut(
#   wines$quality,
#   breaks = c(0, 3, 6, 8, 10),
#   labels = c("Very Low", "Medium", "High", "Very High"),
#   right = TRUE
# )
# wines$qlevelN <- as.numeric(wines$qlevel)
# 
# table(wines$qlevel)
# table(wines$qlevelN)

train_index <- createDataPartition(train_data$quality, p = 0.7, list = FALSE)
train_set <- train_data[train_index, ]
test_set <- train_data[-train_index, ]

model_base <- "quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + total.sulfur.dioxide + pH + alcohol + sulphates"

features <- c(
  "fixed.acidity",
  "volatile.acidity",
  "citric.acid",
  "residual.sugar",
  "total.sulfur.dioxide",
  "pH",
  "alcohol",
  "sulphates",
  "type",
  "location"
)


# results <- test_polynomials(train_data, model_base, features)
# results <- results[order(results$MSE_QUAD), ]
# results

train_index <- createDataPartition(train_data[["quality"]], p = 0.7, list = FALSE)
train_set <- train_data[train_index, ]
test_set <- train_data[-train_index, ]

model <- quality ~ fixed.acidity + citric.acid + chlorides + residual.sugar + pH + alcohol + sulphates + type + location + type * location
fit <- lm(model, data = train_set)

summary_fit <- summary(fit)
vif_values <- vif(fit)

sprintf("MSE: %.6f -- ADJ RSQR: %.6f -- AIC: %.6f", mean(residuals(fit)^2), summary_fit$adj.r.squared, AIC(fit))
test_set$predicted_quality <- predict(fit, newdata = test_set)
head(test_set[, c("quality", "predicted_quality")])

ggplot(test_set, aes(x = predicted_quality, y = quality)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Prices",
    x = "Predicted Prices",
    y = "Actual Prices"
  ) +
  theme_minimal()


train_set$quality <- factor(train_set$quality)
test_set$quality <- factor(test_set$quality, levels = levels(train_set$quality))
nbModel <- naiveBayes(quality ~ citric.acid + residual.sugar + pH + alcohol + sulphates + type + location, data = train_set)

predictions_prob <- predict(nbModel, test_set, type = "raw")
quality_levels <- as.numeric(colnames(predictions_prob))
predicted_numeric <- apply(predictions_prob, 1, function(x) sum(x * quality_levels))

actual_numeric <- as.numeric(as.character(test_set$quality))
mse <- mean((actual_numeric - predicted_numeric)^2)
sprintf("Mean Squared Error (MSE): %.6f", mse)


# 
# 
# fit <- lm(as.formula(model_base), train_data)
# 
# preds <- predict(fit, newdata = test_data)
# 
# model_linear <- lm(quality ~ residual.sugar, data = train_data)
# model_quad <- lm(quality ~ residual.sugar + I(residual.sugar^2), data = train_data)
# model_cubic <- lm(quality ~ residual.sugar+ I(residual.sugar^2) + I(residual.sugar^3), data = train_data)
# 
# AIC(model_linear, model_quad, model_cubic)
# summary(model_linear)$adj.r.squared
# summary(model_quad)$adj.r.squared
# summary(model_cubic)$adj.r.squared
# 
# par(mfrow = c(1, 3))

predictions_prob <- predict(nbModel, test_set)

ggplot(test_set, aes(x = quality, y = predictions_prob)) +
  geom_point(color = "blue") +
  geom_abline(
    slope = 1,
    intercept = 0,
    color = "red",
    linetype = "dashed"
  ) +
  labs(title = "Actual vs Predicted Quality", x = "Actual Quality", y = "Predicted Quality")


ggplot(train_data, aes(x = alcohol, y = quality)) +
  geom_point() +
  geom_smooth(method = "loess", color = "green") +
  labs(title = "Quality vs Alcohol")

ggplot(train_data, aes(x = residual.sugar, y = quality)) +
  geom_point() +
  geom_smooth(method = "loess", color = "blue") +
  labs(title = "Quality vs Residual Sugar")
# 
# ggplot(train_data, aes(x = volatile.acidity, y = quality)) +
#   geom_point() +
#   geom_smooth(method = "loess", color = "blue") +
#   labs(title = "Quality vs Volatile Acidity")
# 
# ggplot(train_data, aes(x = citric.acid, y = quality)) +
#   geom_point() +
#   geom_smooth(method = "loess", color = "blue") +
#   labs(title = "Quality vs Citric Acid")

