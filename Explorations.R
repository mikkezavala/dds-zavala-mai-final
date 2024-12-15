library(e1071)
library(parallel)
library(progressr)
library(furrr)
library(caret)
library(DT)
library(Metrics) 
library(dplyr)
library(furrr)
library(progressr)
library(ggplot2)

train_data = read.csv(
  './Wine_Train.csv',
  header = TRUE
)

train_data = read.csv(
  './Wine_Test_Set.csv',
  header = TRUE
)


# Parallel Execution Setup
n_cores <- detectCores()
max_memory <- 16 * 1024 ^ 3
plan(multisession, workers = max(1, n_cores - 2))
options(future.globals.maxSize = max_memory)


evaluateModels <- function(target, features, data_train) {
  ### KNN
  train_index <- createDataPartition(data_train[[target]], p = 0.7, list = FALSE)
  train_data <- data_train[train_index, ]
  test_data <- data_train[-train_index, ]
  
  train_x <- train_data[, features, drop = FALSE]
  test_x <- test_data[, features, drop = FALSE]
  train_labels <- factor(train_data[[target]])
  test_labels <- factor(test_data[[target]], levels = levels(train_labels))
  
  knnModel <- FNN::knn(
    train = train_x,
    test = test_x,
    cl = train_labels,
    k = ceiling(sqrt(nrow(test_data)))
  )
  
  knnMatrix <- confusionMatrix(factor(knnModel, levels = levels(test_labels)), test_labels, mode = "everything")
  
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
      nb_specificity = mean(nbMatrix$byClass[, "Specificity"], na.rm = TRUE),
      knn_accuracy = knnMatrix$overall["Accuracy"],
      knn_sensitivity = mean(knnMatrix$byClass[, "Sensitivity"], na.rm = TRUE),
      knn_specificity = mean(knnMatrix$byClass[, "Specificity"], na.rm = TRUE)
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
        knn_accuracy = knn_nb_results$knn_accuracy,
        knn_sensitivity = knn_nb_results$knn_sensitivity,
        knn_specificity = knn_nb_results$knn_specificity,
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

results <- runTests("quality", wines |> dplyr::select(-all_of(c("ID"))))

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

wines$qlevel <- cut(
  wines$quality,
  breaks = c(0, 3, 6, 8, 10),
  labels = c("Very Low", "Medium", "High", "Very High"),
  right = TRUE
)
wines$qlevelN <- as.numeric(wines$qlevel)

table(wines$qlevel)
table(wines$qlevelN)

train_index <- createDataPartition(wines$quality, p = 0.7, list = FALSE)
train_data <- wines[train_index, ]
test_data <- wines[-train_index, ]

model_base <- "quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + total.sulfur.dioxide + pH + alcohol + sulphates"

features <- c(
  "fixed.acidity",
  "volatile.acidity",
  "citric.acid",
  "residual.sugar",
  "total.sulfur.dioxide",
  "pH",
  "alcohol",
  "sulphates"
)


# results <- test_polynomials(train_data, model_base, features)
# results <- results[order(results$MSE_QUAD), ]
# results
set.seed(1233)
model <- qlevelN ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + total.sulfur.dioxide + pH + alcohol + sulphates + I(alcohol^2) + I(alcohol^3) + I(residual.sugar ^2) + I(residual.sugar ^3) + I(total.sulfur.dioxide ^2) + I(total.sulfur.dioxide ^3) + I(pH ^2) + I(pH ^3) + I(volatile.acidity ^2) + I(volatile.acidity ^3) + I(fixed.acidity ^2) + I(fixed.acidity ^3) + I(sulphates^2) + I(sulphates^3) + I(citric.acid ^2) + I(citric.acid ^3)

fit <- lm(model, data = train_data)
summary <- summary(fit)
sprintf("MSE: %.6f -- ADJ RSQR: %.6f -- AIC: %.6f", mean(residuals(fit) ^ 2), summary$adj.r.squared, AIC(fit))


# 
# 
# fit <- lm(as.formula(model_base), train_data)
# 
preds <- predict(fit, newdata = test_data)
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
plot(fit, which = 1)  # Residuals for Linear
# plot(model_quad, which = 1)    # Residuals for Quadratic
# plot(model_cubic, which = 1)   # Residuals for Cubic
# 
ggplot(test_data, aes(x = quality, y = preds)) +
  geom_point(color = "blue") +                  # Scatter plot
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

ggplot(train_data, aes(x = residual.sugar, y = qlevelN)) +
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

