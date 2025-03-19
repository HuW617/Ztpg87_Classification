# Define required packages
list_of_packages <- c("tidyverse", "caret", "randomForest", "xgboost", 
                      "pROC", "MASS", "ggplot2")

# Install missing packages
for (pkg in list_of_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

# Load required libraries
library(tidyverse)  # Includes dplyr, ggplot2, etc.
library(caret)  # Machine learning framework
library(randomForest)  # Random Forest algorithm
library(xgboost)  # XGBoost algorithm
library(pROC)  # AUC calculation
library(MASS)  # LDA (Linear Discriminant Analysis)
library(ggplot2)  # Visualization


# Load dataset
data_url <- "https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv"
df <- read.csv(data_url, header = TRUE)

# Check data structure
str(df)

# Summary statistics
summary(df)

# Check missing values
sum(is.na(df))

# Check target variable distribution
table(df$Personal.Loan)
prop.table(table(df$Personal.Loan))  # Show proportion

# Visualize target variable
ggplot(df, aes(x = as.factor(Personal.Loan), fill = as.factor(Personal.Loan))) +
  geom_bar(aes(y = after_stat(count) / sum(after_stat(count))), stat = "count") + 
  scale_y_continuous(labels = scales::percent) +  
  scale_fill_manual(values = c("red", "blue"), labels = c("No Loan", "Has Loan")) +
  labs(title = "Loan Approval Distribution", x = "Personal Loan", y = "Percentage", fill = "Loan Status") +
  theme_minimal()

# Loan vs Income Distribution
ggplot(df, aes(x = Income, fill = as.factor(Personal.Loan))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("red", "blue"), labels = c("No Loan", "Has Loan")) +
  facet_wrap(~Personal.Loan, scales = "free_y") +  
  labs(title = "Income Distribution by Loan Approval", x = "Income", y = "Density", fill = "Loan Status") +
  theme_minimal()

# Loan vs Age Distribution
ggplot(df, aes(x = as.factor(Personal.Loan), y = Age, fill = as.factor(Personal.Loan))) +
  geom_boxplot() +
  scale_fill_manual(values = c("red", "blue"), labels = c("No Loan", "Has Loan")) +
  labs(title = "Age Distribution by Loan Approval", x = "Personal Loan", y = "Age", fill = "Loan Status") +
  theme_minimal()

# Loan vs CreditCard
ggplot(df, aes(x = as.factor(CreditCard), fill = as.factor(Personal.Loan))) +
  geom_bar(aes(y = after_stat(count) / sum(after_stat(count))), stat = "count", position = "dodge") + 
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("red", "blue"), labels = c("No Loan", "Has Loan")) +
  labs(title = "Loan Approval by Credit Card Ownership", x = "Credit Card", y = "Percentage", fill = "Loan Status") +
  theme_minimal()




# Data Preprocessing

# Replace negative Experience with median
df$Experience[df$Experience < 0] <- median(df$Experience[df$Experience >= 0], na.rm = TRUE)

# Remove ZIP Code column
df <- df[, !(names(df) %in% "ZIP.Code")]

# Convert categorical variables to factors
df <- df %>%
  mutate(across(c(Education, Family, Securities.Account, CD.Account, Online, CreditCard, Personal.Loan), as.factor))

# Keep Personal.Loan as 0/1
df$Personal.Loan <- as.factor(df$Personal.Loan)

# Standardize numeric variables
standardize <- function(x) { (x - mean(x)) / sd(x) }
df_norm <- df %>%
  mutate(across(c(Age, Experience, Income, CCAvg, Mortgage), standardize))

# Split data (70% train, 15% validation, 15% test)
trainIndex <- createDataPartition(df_norm$Personal.Loan, p = 0.7, list = FALSE)
train_data <- df_norm[trainIndex, ]
temp_data <- df_norm[-trainIndex, ]

validIndex <- createDataPartition(temp_data$Personal.Loan, p = 0.5, list = FALSE)
valid_data <- temp_data[validIndex, ]
test_data <- temp_data[-validIndex, ]

# Check class distribution
prop.table(table(train_data$Personal.Loan))
prop.table(table(valid_data$Personal.Loan))
prop.table(table(test_data$Personal.Loan))

# Check data structure and summary
str(train_data)
summary(train_data)








#====================================================#
# 1. Setup: Set seed, load packages, convert target to factor
#====================================================#
set.seed(2323)

# Convert target variable (0/1) to factor ("No"/"Yes") for caret
train_data <- train_data %>%
  mutate(Personal.Loan = factor(Personal.Loan, levels = c(0, 1), labels = c("No", "Yes")))
valid_data <- valid_data %>%
  mutate(Personal.Loan = factor(Personal.Loan, levels = c(0, 1), labels = c("No", "Yes")))
test_data <- test_data %>%
  mutate(Personal.Loan = factor(Personal.Loan, levels = c(0, 1), labels = c("No", "Yes")))

#====================================================#
# 2. Cross-validation setup (caret)
#====================================================#

# 5-fold cross-validation, repeated 3 times
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

#====================================================#
# 3. Model 1: Logistic Regression
#====================================================#

# Use "glm" with binomial family, optimize using AUC
set.seed(2323)
model_logit <- caret::train(
  Personal.Loan ~ .,
  data = train_data,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = train_control
)
model_logit

#====================================================#
# 4. Model 2: LDA
#====================================================#
set.seed(2323)
model_lda <- caret::train(
  Personal.Loan ~ .,
  data = train_data,
  method = "lda",
  metric = "ROC",
  trControl = train_control
)
model_lda

#====================================================#
# 5. Model 3: Random Forest
#====================================================#

# Auto-tune mtry with tuneLength = 3
set.seed(2323)
model_rf <- caret::train(
  Personal.Loan ~ .,
  data = train_data,
  method = "rf",
  metric = "ROC",
  tuneLength = 3,
  trControl = train_control
)
model_rf
model_rf$bestTune

#====================================================#
# 6. Model 4: XGBoost
#====================================================#
# Auto-tune hyperparameters with tuneLength = 3

set.seed(2323)
model_xgb <- caret::train(
  Personal.Loan ~ .,
  data = train_data,
  method = "xgbTree",
  metric = "ROC",
  tuneLength = 3,
  trControl = train_control
)
model_xgb
model_xgb$bestTune

#====================================================#
# 7. Evaluate models on validation set
#====================================================#

# Function to compute performance metrics
evaluate_model <- function(model, data, positive_class = "Yes") {
  pred_prob <- predict(model, newdata = data, type = "prob")[, positive_class]
  pred_class <- predict(model, newdata = data)
  
  # Calculate the confusion matrix
  cm <- confusionMatrix(
    data = pred_class,
    reference = data$Personal.Loan,
    positive = positive_class
  )
  
  # Calculate AUC
  auc_val <- roc(response = data$Personal.Loan, predictor = pred_prob, levels = c("No","Yes"))$auc

  acc  <- cm$overall["Accuracy"]
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  f1   <- cm$byClass["F1"]
  return(list(Accuracy = acc, AUC = auc_val, Sensitivity = sens, Specificity = spec, F1 = f1))
}

# Evaluate models
res_logit <- evaluate_model(model_logit, valid_data)
res_lda   <- evaluate_model(model_lda,   valid_data)
res_rf    <- evaluate_model(model_rf,    valid_data)
res_xgb   <- evaluate_model(model_xgb,   valid_data)

# Compare models
model_compare <- data.frame(
  Model       = c("LogisticRegression", "LDA", "RandomForest", "XGBoost"),
  Accuracy    = c(res_logit$Accuracy,    res_lda$Accuracy,    res_rf$Accuracy,    res_xgb$Accuracy),
  AUC         = c(res_logit$AUC,         res_lda$AUC,         res_rf$AUC,         res_xgb$AUC),
  Sensitivity = c(res_logit$Sensitivity, res_lda$Sensitivity, res_rf$Sensitivity, res_xgb$Sensitivity),
  Specificity = c(res_logit$Specificity, res_lda$Specificity, res_rf$Specificity, res_xgb$Specificity),
  F1          = c(res_logit$F1,          res_lda$F1,          res_rf$F1,          res_xgb$F1)
)
print(model_compare)

# Convert to long format for visualization
model_long <- model_compare %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

# Plot grouped bar chart
ggplot(model_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Model Performance Comparison", x = "Model", y = "Metric Value") +
  theme_minimal() +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_fill_brewer(palette = "Set2")

#====================================================#
# 8. XGBoost Hyperparameter Tuning (Class Weighting)
#====================================================#

# Compute class weights (give more weight to class "Yes")
class_0 <- sum(train_data$Personal.Loan == "No")  # 0 class
class_1 <- sum(train_data$Personal.Loan == "Yes")  # 1 class
class_weight <- class_0 / class_1  # Calculate weight ratio

# Check best parameters from previous tuning
print(model_xgb$bestTune)

# Refined hyperparameter grid search
xgb_grid_refined <- expand.grid(
  nrounds = c(80, 100, 120, 150),
  max_depth = c(2, 3, 4),
  eta = c(0.25, 0.3, 0.35),
  gamma = c(0, 1),
  colsample_bytree = c(0.75, 0.8, 0.85),
  min_child_weight = c(1, 3),
  subsample = c(0.9, 1.0)
)


set.seed(2323)
model_xgb_refined <- caret::train(
  Personal.Loan ~ .,
  data = train_data,
  method = "xgbTree",
  metric = "ROC",
  tuneGrid = xgb_grid_refined,
  trControl = train_control,
  weights = ifelse(train_data$Personal.Loan == "Yes", class_weight, 1)  # Give class 1 a higher weight
)

# Best parameters after tuning
print(model_xgb_refined$bestTune)

# Get best parameter combination
best_index <- which.max(model_xgb_refined$results$ROC)
best_params <- model_xgb_refined$results[best_index, ]

cat("\n=== Best Parameters for Final XGBoost Model ===\n")
print(best_params)

#====================================================#
# 9. Evaluate Final Model on Test Set (Threshold=0.5)
#====================================================#

res_xgb_test <- evaluate_model(model_xgb_refined, test_data)
cat("\n=== Final XGBoost Model Performance on Test Data (Threshold=0.5) ===\n")
print(res_xgb_test)

# Get predicted probabilities
pred_xgb_prob <- predict(model_xgb_refined, newdata = test_data, type = "prob")[, "Yes"]

#====================================================#
# 10. Find Best Precision-Recall Threshold
#====================================================#

# Convert to numeric vector
best_precision_threshold_numeric <- as.numeric(best_precision_threshold)

# Show all thresholds (rounded)
cat("\nAll thresholds (rounded):", 
    paste(round(best_precision_threshold_numeric, 3), collapse = ", "), "\n")

# Best threshold for precision-recall balance
cat("Best threshold (Minimizing False Positives, maximizing Precision & Recall):", 
    round(best_precision_threshold_numeric[1], 3), "\n")



#====================================================#
# 11. Predict on Test Set with Best Threshold
#====================================================#

# Convert best threshold to numeric
best_precision_threshold_numeric <- as.numeric(best_precision_threshold)

# Show all thresholds
cat("\nAll thresholds (rounded):", paste(round(best_precision_threshold_numeric, 3), collapse = ", "), "\n")

# Select best threshold
best_threshold <- best_precision_threshold_numeric[1]

# Apply threshold to classify predictions
pred_xgb_class_best <- ifelse(pred_xgb_prob > best_threshold, "Yes", "No")

# Compute confusion matrix
cm_xgb_test_best <- confusionMatrix(
  reference = test_data$Personal.Loan, 
  data = factor(pred_xgb_class_best, levels = levels(test_data$Personal.Loan)), 
  positive = "Yes"
)

cat("\n=== Confusion Matrix on Test Data (Best Precision-Recall Threshold =", round(best_threshold, 3), ") ===\n")
print(cm_xgb_test_best)

# Compute Precision, Recall, F1-score
precision_best <- cm_xgb_test_best$byClass["Precision"]
recall_best <- cm_xgb_test_best$byClass["Sensitivity"]
f1_score_best <- 2 * (precision_best * recall_best) / (precision_best + recall_best)

cat("\n=== Model Performance at Best Threshold =", round(best_threshold, 3), " ===\n")
cat("Precision:", round(precision_best, 4), "\n")
cat("Recall (Sensitivity):", round(recall_best, 4), "\n")
cat("F1-score:", round(f1_score_best, 4), "\n")

#====================================================#
# 12. Compare XGBoost Before & After Optimization
#====================================================#

metrics <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1"),
  Pre_Optimization = c(0.9827, 0.9444, 0.9867, 0.8827, 0.9128),
  Post_Optimization = c(0.9760, 0.9861, 0.9749, 0.8068, 0.8875)
)

# Convert to long format for plotting
metrics_long <- pivot_longer(
  data = metrics,
  cols = c("Pre_Optimization", "Post_Optimization"),
  names_to = "Model",
  values_to = "Value"
)

# Plot comparison bar chart
ggplot(metrics_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  ylim(0, 1) +
  labs(
    title = "XGBoost Performance Before vs After Optimization",
    x = "Metric",
    y = "Value"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c(
    "Pre_Optimization" = "#1f77b4",
    "Post_Optimization" = "#ff7f0e"
  ))

#====================================================#
# 13. Compare ROC Curves Before & After Optimization
#====================================================#

# Get predicted probabilities
pred_xgb_prob_pre <- predict(model_xgb, newdata = test_data, type = "prob")[, "Yes"]
pred_xgb_prob_post <- predict(model_xgb_refined, newdata = test_data, type = "prob")[, "Yes"]

# Compute ROC curves
roc_pre <- roc(response = test_data$Personal.Loan, predictor = pred_xgb_prob_pre, levels = c("No", "Yes"))
roc_post <- roc(response = test_data$Personal.Loan, predictor = pred_xgb_prob_post, levels = c("No", "Yes"))

# Plot ROC curves
plot(roc_pre, col = "#1f77b4", lwd = 2, main = "ROC Curve Comparison: Pre vs Post Optimization")
lines(roc_post, col = "#ff7f0e", lwd = 2)

# Add legend with AUC values
legend("bottomright",
       legend = c(paste("Pre Optimization (AUC =", round(auc(roc_pre), 3), ")"),
                  paste("Post Optimization (AUC =", round(auc(roc_post), 3), ")")),
       col = c("#1f77b4", "#ff7f0e"),
       lwd = 2)


