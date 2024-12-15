# Load libraries 
library(tidyverse)
library(caret)
library(class)
library(e1071)
library(corrplot)
library(FNN)
library(car)
library(GGally)
library(naivebayes)
library(ggcorrplot)

# Load data
wine_train = read.csv("Wine_Train.csv")
wine_test = read.csv("Wine_Test_Set.csv")

# Data Structure
str(wine_train)
str(wine_test)

# Missing Data
print(colSums(is.na(wine_train)))
print(colSums(is.na(wine_test)))

# Histogram of the quality
ggplot(wine_train, aes(x = quality)) +
  geom_histogram(binwidth = 1, fill = "maroon", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Wine Quality", x = "Quality", y = "Frequency") +
  theme_minimal()

# Correlation Matrix
correlation_matrix = cor(wine_train %>% select_if(is.numeric))
corrplot(correlation_matrix, 
         method = "color",           
         type = "upper",            
         tl.col = "black",          
         tl.srt = 45,               
         addCoef.col = "black",     
         number.cex = 0.7,          
         col = colorRampPalette(c("red", "white", "blue"))(200),  
         diag = FALSE,              
         tl.cex = 0.8)

# Variance Inflation Factor (VIF)
vif_data = lm(quality ~ ., data = wine_train %>% select(-ID))
vif_values = vif(vif_data)
print(vif_values)

# Remove features with high VIF
high_vif_threshold = 10
features_to_remove = names(vif_values[vif_values > high_vif_threshold])
model_cols = setdiff(names(wine_train %>% select_if(is.numeric)), c("quality", features_to_remove))
cat("Removed features with high VIF:\n", features_to_remove, "\n")

# Feature Engineering Function
create_engineered_features = function(data) {
  data %>%
    mutate(
      acid_ratio = fixed.acidity / volatile.acidity,
      sugar_acid_ratio = residual.sugar / fixed.acidity,
      alcohol_density_ratio = alcohol / density,
      alcohol_squared = alcohol^2,
      pH_squared = pH^2,
      alcohol_acid = alcohol * fixed.acidity,
      sugar_acid = residual.sugar * fixed.acidity,
      log_sulfur = log1p(total.sulfur.dioxide),
      log_sugar = log1p(residual.sugar)
    )
}
wine_train_engineered = create_engineered_features(wine_train)
wine_test_engineered = create_engineered_features(wine_test)

# Scaling
preprocess_params = preProcess(wine_train_engineered[model_cols], method = c("center", "scale"))
wine_train_scaled = predict(preprocess_params, wine_train_engineered)
wine_test_scaled = predict(preprocess_params, wine_test_engineered)

# Validation Split
set.seed(123)
trainIndex = createDataPartition(wine_train$quality, p = 0.8, list = FALSE)
train_set = wine_train_scaled[trainIndex, ]
valid_set = wine_train_scaled[-trainIndex, ]

# quality_cat column to train_set and valid_set
train_set$quality_cat = cut(
  train_set$quality,
  breaks = c(-Inf, 4, 5, 6, 7, Inf),
  labels = c("Very Low", "Low", "Medium", "High", "Very High")
)

valid_set$quality_cat = cut(
  valid_set$quality,
  breaks = c(-Inf, 4, 5, 6, 7, Inf),
  labels = c("Very Low", "Low", "Medium", "High", "Very High")
)

# Cross-Validation Setup
cv_control = trainControl(method = "cv", number = 5)



# Linear Regression 
lm_model = train(quality ~ ., data = train_set[, c(model_cols, "quality")], 
                 method = "lm", trControl = cv_control)
lm_predictions = predict(lm_model, valid_set[, model_cols])
lm_mae = mean(abs(lm_predictions - valid_set$quality))
cat("MAE for Linear Regression:", lm_mae, "\n")

# Naive Bayes 
nb_model = train(
  quality_cat ~ .,
  data = train_set[, c(model_cols, "quality_cat")],
  method = "naive_bayes",
  trControl = cv_control
)

# Predictions and MAE for Naive Bayes
nb_predictions_cat = predict(nb_model, valid_set[, model_cols])
nb_predictions = as.numeric(as.character(factor(
  nb_predictions_cat,
  levels = c("Very Low", "Low", "Medium", "High", "Very High"),
  labels = c(3, 4.5, 5.5, 6.5, 7.5)
)))
nb_mae = mean(abs(nb_predictions - valid_set$quality))
cat("MAE for Naive Bayes:", nb_mae, "\n")

# k-NN Model with Cross-Validation
cv_control = trainControl(method = "cv", number = 5) # 5-fold cross-validation

evaluate_knn = function(k, train_data, train_labels, folds) {
  fold_mae = vector("numeric", length = length(folds))
  
  for (i in seq_along(folds)) {
    valid_idx = folds[[i]]
    train_idx = setdiff(seq_len(nrow(train_data)), valid_idx)
    
    knn_model = knn.reg(
      train = train_data[train_idx, ],
      test = train_data[valid_idx, ],
      y = train_labels[train_idx],
      k = k
    )
    predictions = knn_model$pred
    fold_mae[i] = mean(abs(predictions - train_labels[valid_idx]))
  }
  
  return(mean(fold_mae)) # Average MAE across all folds
}

set.seed(123)
folds = createFolds(train_set$quality, k = 5, list = TRUE)

k_values = seq(1, 30, by = 1)
knn_mae_values = sapply(k_values, function(k) {
  evaluate_knn(k, as.matrix(train_set[, model_cols]), train_set$quality, folds)
})

best_k = k_values[which.min(knn_mae_values)]
cat("Best k for k-NN:", best_k, "\n")

final_knn_model = knn.reg(
  train = as.matrix(train_set[, model_cols]),
  test = as.matrix(valid_set[, model_cols]),
  y = train_set$quality,
  k = best_k
)

# Calculate MAE 
knn_mae = mean(abs(final_knn_model$pred - valid_set$quality))
cat("MAE for k-NN:", knn_mae, "\n")

# Compare MAE of Models
model_comparison = data.frame(
  Model = c("Linear Regression", "Naive Bayes", "k-NN"),
  MAE = c(lm_mae, nb_mae, knn_mae)
)
print(model_comparison)

# Final Predictions for Test Set (k-NN)
final_knn_predictions = knn.reg(
  train = as.matrix(wine_train_scaled[, model_cols]),
  test = as.matrix(wine_test_scaled[, model_cols]),
  y = wine_train_scaled$quality,
  k = best_k
)$pred

# Combine MAE information with the data
mae_data = data.frame(
  Model = c("Linear Regression", "Naive Bayes", "k-NN"),
  MAE = c(lm_mae, nb_mae, knn_mae)
)

# Bar plot of MAE for each model with annotations
ggplot(model_comparison, aes(x = Model, y = MAE, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(MAE, 3)), vjust = -0.5, size = 4, color = "black") +
  labs(
    title = "Model Comparison by MAE",
    x = "Model",
    y = "Mean Absolute Error"
  ) +
  theme_minimal()

# Define model_cols without 'ID' and 'quality'
model_cols = setdiff(names(wine_train %>% select_if(is.numeric)), c("ID", "quality"))

# Compute correlations between features and wine quality
feature_correlations_knn = cor(wine_train[, model_cols], wine_train$quality, use = "complete.obs")

# Convert correlations into a data frame
feature_importance_knn = data.frame(
  Feature = model_cols,
  Correlation = feature_correlations_knn
) %>% 
  arrange(desc(abs(Correlation)))  # Sort by absolute correlation

# Create the plot without 'ID'
ggplot(feature_importance_knn, aes(x = reorder(Feature, Correlation), y = Correlation, fill = Correlation > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("red", "blue"), labels = c("Negative", "Positive")) +
  labs(
    title = "Feature Importance for k-NN (Correlation with Quality)",
    x = "Feature",
    y = "Correlation",
    fill = "Direction"
  ) +
  theme_minimal()




# Create submission file with ID and predicted Quality
create_submission_file = function(test_data, predictions, file_path) {
  submission = data.frame(
    ID = test_data$ID,
    Quality = predictions
  )
  write.csv(submission, file_path, row.names = FALSE)
  cat("Submission file created successfully at", file_path, "\n")
}

# Define the file path and call the function
file_path = "~/Desktop/Wine_Test_Predictions.csv"
create_submission_file(wine_test, final_knn_predictions, file_path)

# Libraries


