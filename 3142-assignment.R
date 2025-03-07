#load the required libraries
library(tidyverse)
library(ROSE)
library(smotefamily)
library(glmnet)
library(glm2)
library(caret)
library(pROC)
library(MASS)
library(randomForest)

accidents_data <- read.csv("VicRoadFatalData.csv")

#TASK 1: Exploratory Data Analysis
head(accidents_data)           
summary(accidents_data)   

#Age summaries
fatal_summary <- accidents_data %>%
  filter(fatal == 1) %>%
  summarise(
    MEAN = mean(AGE, na.rm = TRUE),
    MEDIAN = median(AGE, na.rm = TRUE),
    SD = sd(AGE, na.rm = TRUE),
    MIN = min(AGE, na.rm = TRUE),
    MAX = max(AGE, na.rm = TRUE)
  )

#Summary statistics for age in non-fatal accidents
non_fatal_summary <- accidents_data %>%
  filter(fatal == 0) %>%
  summarise(
    mean_age = mean(AGE, na.rm = TRUE),
    median_age = median(AGE, na.rm = TRUE),
    sd_age = sd(AGE, na.rm = TRUE),
    min_age = min(AGE, na.rm = TRUE),
    max_age = max(AGE, na.rm = TRUE)
  )

print(fatal_summary)
print(non_fatal_summary)

#Exploratory data analysis for Sex
accidents_data_sex <- accidents_data %>%
  count(SEX) %>%
  mutate(prop = n / sum(n))
##PIE CHART##
ggplot(accidents_data_sex, aes(x = "", y = prop, fill = SEX)) +
  geom_bar(width = 1, stat = "identity", color = "black") +
  coord_polar("y") +
  theme_minimal() +
  labs(title = "Distribution of Drivers' Sex",
       x = NULL, 
       y = NULL) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.title = element_blank()) +
  ggtitle("Distribution of Drivers' Sex") +
  theme(plot.title = element_text(hjust = 0.5))

##TABLE OF PROPORTIONS##
proportions_table <- accidents_data %>%
  group_by(SEX, fatal) %>%
  summarise(count = n()) %>%
  mutate(prop = count / sum(count) * 100) %>%
  ungroup()

proportions_table

## Characteristics of the vehicles ##

# VEHICLE_TYPE and VEHICLE_YEAR_MANUF
table(accidents_data$VEHICLE_TYPE)

ggplot(accidents_data, aes(x = VEHICLE_YEAR_MANUF)) +
  geom_bar(fill = 'blue', color = 'black') +
  theme_minimal() +
  labs(title = "Distribution of Vehicle Manufacturing Year",
       x = "Year of Manufacturing", 
       y = "Count")+
  theme(plot.title = element_text(hjust = 0.5))

###VEHICLE TYPE AND ACCIDENT TYPE###
summary_data <- accidents_data %>%
  group_by(VEHICLE_TYPE, ACCIDENT_TYPE) %>%
  summarise(count = n()) %>%
  mutate(prop = count / sum(count))

accident_colors <- c("Collision with a fixed object" = "red",
                     "Collision with vehicle" = "blue",
                     "Struck Pedestrian" = "green",
                     "Vehicle overturned (no collision)" = "purple")

# Plot the data
ggplot(summary_data, aes(x = VEHICLE_TYPE, y = prop, fill = ACCIDENT_TYPE)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Proportion of Accident Type for Different Vehicle Types",
       x = "Vehicle Type", 
       y = "Proportion") +
  scale_fill_manual(values = accident_colors) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#TASK 2: Modelling the relationship between fatal accidents and the different variables

# Split data into training and test sets
# We use set.seed() to ensure that the random numbers used in the split 
# are reproducible in the future runs.
set.seed(123) 

# We use caret's createDataPartition() function to create indices for 
# training data. We specify p=0.7 to use 70% of the data for training.
Victrainindex <- createDataPartition(accidents_data$fatal, p = 0.7, list = FALSE)

# We use these indices to get the actual training data
Victrain <- accidents_data[Victrainindex, ]

# The rest 30% data is used for testing
Victest <- accidents_data[-Victrainindex, ]

logistic.regression <- glm(fatal~SEX+AGE+LICENCE_STATE+HELMET_BELT_WORN+VEHICLE_YEAR_MANUF+VEHICLE_BODY_STYLE+VEHICLE_MAKE+VEHICLE_TYPE+FUEL_TYPE+VEHICLE_COLOUR+TOTAL_NO_OCCUPANTS+DAY_OF_WEEK+ACCIDENT_TYPE+LIGHT_CONDITION+ROAD_GEOMETRY+SPEED_ZONE+SURFACE_COND+ATMOSPH_COND+ROAD_SURFACE_TYPE,data=Victrain,family=binomial)

stepwise_model <- step(logistic.regression, direction = "both")

# Print summary of final model
# The summary() function provides a detailed summary of the final model, 
# including the estimates of the coefficients, standard errors, and significance levels.
summary(stepwise_model)

# Predict on test data
# We use the predict() function to make predictions on the test data.
# The 'type="response"' argument tells the function to output probabilities.
predicted_probs <- predict(stepwise_model, newdata = Victest, type = "response")

# Convert probabilities to binary predictions
# We convert the predicted probabilities to binary predictions. We use a 
# threshold of 0.5, classifying observations with a predicted probability 
# of 0.5 or higher as 1 and below 0.5 as 0.
predicted_class <- ifelse(predicted_probs >= 0.5, 1, 0)

# Evaluate model
# We create a confusion matrix to evaluate the performance of the model. 
# The confusion matrix shows the counts of true positives, true negatives, 
# false positives, and false negatives. 
conf_mat <- table(predicted_class, Victest$fatal)

# We calculate the accuracy of the model by summing the diagonal elements 
# (true positives and true negatives) and dividing by the total number of predictions.
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
sensitivity <- conf_mat[2, 2] / sum(conf_mat[2, ])
specificity <- conf_mat[1, 1] / sum(conf_mat[1, ])
precision <- conf_mat[2, 2] / sum(conf_mat[, 2])
f1_score <- 2 * precision * sensitivity / (precision + sensitivity)

cat(paste(
  "Accuracy of Predictive Model One:", accuracy, "\n",
  "Sensitivity (Recall) of Predictive Model One:", sensitivity, "\n",
  "Specificity of Predictive Model One:", specificity, "\n",
  "Precision of Predictive Model One:", precision, "\n",
  "F1-score of Predictive Model One:", f1_score, "\n",
  sep = ""
))

model_summary <- summary(stepwise_model)
coeffs <- model_summary$coefficients
significant_vars <- coefs[coefs[, 4] < 0.05, , drop = FALSE]

# Printing the significant variables and their p-values
cat("Significant Variables and Their P-values:\n")
for (var in rownames(significant_vars)) {
  cat(var, "p-value:", significant_vars[var, 4], "\n")
}
###INTERACTIONS BETWEEN SEX AND AGE###
interaction_model <- glm(fatal ~ SEX * AGE, data=Victrain, family=binomial)
summary(interaction_model)


#TASK 3: Predictive modelling of drivers and fatal accidents
vehicle_demographics_data <- accidents_data %>%
  dplyr::select(fatal, SEX, AGE, LICENCE_STATE, HELMET_BELT_WORN, VEHICLE_ID, VEHICLE_YEAR_MANUF, VEHICLE_BODY_STYLE, 
         VEHICLE_MAKE, VEHICLE_TYPE, FUEL_TYPE, VEHICLE_COLOUR, OWNER_POSTCODE, TOTAL_NO_OCCUPANTS)

# Convert 'fatal' to a factor if it isn't already
vehicle_demographics_data$fatal <- as.factor(vehicle_demographics_data$fatal)

# Check the balance before sampling
table(vehicle_demographics_data$fatal)

# Apply over- and under-sampling without specifying N
balanced_data <- ovun.sample(fatal ~ ., data = vehicle_demographics_data, method = "both")$data

# Check the balance after sampling
table(balanced_data$fatal)

# Continue with existing code for training and evaluation
set.seed(123)
trainIndex <- createDataPartition(vehicle_demographics_data$fatal, p = 0.7, list = FALSE)
train <- balanced_data[trainIndex,]
test <- balanced_data[-trainIndex,]

####PREDICTIVE MODEL 1: LOGISTIC REGRESSION####
model <- glm(fatal ~ SEX + AGE + LICENCE_STATE + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF + 
                VEHICLE_BODY_STYLE + VEHICLE_MAKE + VEHICLE_TYPE + VEHICLE_COLOUR + TOTAL_NO_OCCUPANTS, data = train, family = binomial)

pred1 <- predict(model, newdata = test, type = "response")

conf1 <- table(pred1 > 0.5 , test$fatal)
print(conf1)

accuracy1 <- sum(diag(conf1)) / sum(conf1)
sensitivity1 <- conf1[2, 2] / sum(conf1[2, ])
specificity1 <- conf1[1, 1] / sum(conf1[1, ])
precision1 <- conf1[2, 2] / sum(conf1[, 2])
f1_score1 <- 2 * precision1 * sensitivity1 / (precision1 + sensitivity1)
roc1 <- roc(test$fatal,pred1)
AUC_value1 <- auc(roc1)
plot(roc1, main = "ROC Curve for Logistic Regression", col = "red", lwd = 2)

cat(paste(
  "Accuracy of Predictive Model One:", accuracy1, "\n",
  "Sensitivity (Recall) of Predictive Model One:", sensitivity1, "\n",
  "Specificity of Predictive Model One:", specificity1, "\n",
  "Precision of Predictive Model One:", precision1, "\n",
  "F1-score of Predictive Model One:", f1_score1, "\n",
  "AUC for Predictive Model One:", AUC_value1, "\n",
  sep = ""
))

####PREDICTIVE MODEL 2: LASSO AND RIDGE####
# Function to fit either LASSO or Ridge model using cross-validation
# 'alpha' parameter differentiates between LASSO (alpha = 1) and Ridge (alpha = 0)
fit_model <- function(alpha, X, y) {
  # Perform cross-validation to find the best lambda for the given alpha
  cv_model <- cv.glmnet(X, y, alpha = alpha)
  # Fit the final model using the optimal lambda
  glmnet(X, y, alpha = alpha, lambda = cv_model$lambda.min)
}

# Function to calculate the root mean squared error (RMSE) between predictions and actual values
calculate_rmse <- function(predicted, actual) {
  sqrt(mean((actual - predicted)^2))
}

# Define the formula representing the relationship between the response variable 'fatal' and predictors
formula.var <- fatal ~ SEX + AGE + LICENCE_STATE + HELMET_BELT_WORN + VEHICLE_YEAR_MANUF +
  VEHICLE_BODY_STYLE + VEHICLE_MAKE + VEHICLE_TYPE + VEHICLE_COLOUR +
  TOTAL_NO_OCCUPANTS

# Create design matrices for training and testing data using the defined formula
X <- model.matrix(formula.var, data = train)
y <- as.numeric(train$fatal) - 1
X_test <- model.matrix(formula.var, data = test)
y_test <- as.numeric(test$fatal) - 1

# Fit Lasso and Ridge models using the training data
lasso_final <- fit_model(alpha = 1, X, y)
ridge_final <- fit_model(alpha = 0, X, y)

# Predict the response for the test data using both LASSO and Ridge models
lasso_predictions <- predict(lasso_final, newx = X_test)
ridge_predictions <- predict(ridge_final, newx = X_test)

# Calculate RMSE for both LASSO and Ridge to evaluate their performance
actual_numeric <- as.numeric(test$fatal) - 1
rmse_lasso <- sqrt(mean((actual_numeric - lasso_predictions)^2))
rmse_ridge <- sqrt(mean((actual_numeric - ridge_predictions)^2))

# Select the best model based on the smallest RMSE
if (rmse_lasso < rmse_ridge) {
  best_model <- "Lasso"
  predictions2 <- lasso_predictions
} else {
  best_model <- "Ridge"
  predictions2 <- ridge_predictions
}

# Print the name of the best model
print(best_model)

# Convert continuous predictions into binary classification using a threshold
threshold <- 0.5
binary_predictions <- ifelse(predictions2 >= threshold, 1, 0)

# Create and print a confusion matrix to summarize the performance of the selected model
conf2 <- table(Actual = test$fatal, Predicted = binary_predictions)
print(conf2)

accuracy2 <- sum(diag(conf2)) / sum(conf2)
sensitivity2 <- conf2[2, 2] / sum(conf2[2, ])
specificity2 <- conf2[1, 1] / sum(conf2[1, ])
precision2 <- conf2[2, 2] / sum(conf2[, 2])
f1_score2 <- 2 * precision2 * sensitivity2 / (precision2 + sensitivity2)
roc2 <- roc(test$fatal, predictions2)
AUC_value2 <- auc(roc2)
plot(roc2, main = "ROC Curve for Lasso Regression", col = "red", lwd = 2)

cat(paste(
  "Accuracy of Predictive Model One:", accuracy2, "\n",
  "Sensitivity (Recall) of Predictive Model One:", sensitivity2, "\n",
  "Specificity of Predictive Model One:", specificity2, "\n",
  "Precision of Predictive Model One:", precision2, "\n",
  "F1-score of Predictive Model One:", f1_score2, "\n",
  "AUC for Predictive Model One:", AUC_value2, "\n",
  sep = ""
))

####PREDICTIVE MODEL 3: RANDOM FOREST####
mtryValues <- seq(2, 9, by = 1)

cvError <- rep(NA, length(mtryValues))

# Define the grid of mtry values to search. This creates a grid of potential hyperparameters to try.
# The mtry parameter controls the number of variables randomly sampled as candidates at each split, and we're
# exploring values from 2 to 9.
grid <- expand.grid(mtry = seq(2, 9, by = 1))

for (i in 1:length(mtryValues)) {
  randomForestModel <- randomForest(factor(fatal) ~ SEX + AGE + LICENCE_STATE + HELMET_BELT_WORN + 
                                      VEHICLE_YEAR_MANUF + VEHICLE_BODY_STYLE + VEHICLE_MAKE + VEHICLE_TYPE + 
                                      VEHICLE_COLOUR + TOTAL_NO_OCCUPANTS, data = train, ntree = 100, mtry = mtryValues[i])
  cvError[i] <- mean(randomForestModel$err.rate[, 2])
}

idealMtry <- mtryValues[which.min(cvError)]

predmodel3 <- randomForest(factor(fatal) ~ SEX + AGE + LICENCE_STATE + HELMET_BELT_WORN + 
                             VEHICLE_YEAR_MANUF + VEHICLE_BODY_STYLE + VEHICLE_MAKE + VEHICLE_TYPE + 
                             VEHICLE_COLOUR + TOTAL_NO_OCCUPANTS, data = train, ntree = 100, mtry = idealMtry)

predictions3 <- predict(predmodel3, newdata = test, type = "response")

conf3<- table(predictions3, test$fatal)
print(conf3)

accuracy3 <- sum(diag(conf3)) / sum(conf3)
sensitivity3 <- conf3[2, 2] / sum(conf3[2, ])
specificity3 <- conf3[1, 1] / sum(conf3[1, ])
precision3 <- conf3[2, 2] / sum(conf3[, 2])
f1_score3 <- 2 * precision3 * sensitivity3 / (precision3 + sensitivity3)

cat(paste(
  "Accuracy of Predictive Model One:", accuracy3, "\n",
  "Sensitivity (Recall) of Predictive Model One:", sensitivity3, "\n",
  "Specificity of Predictive Model One:", specificity3, "\n",
  "Precision of Predictive Model One:", precision3, "\n",
  "F1-score of Predictive Model One:", f1_score3, "\n",
  sep = ""
))

###EVALUATING 2500 DRIVERS###
Drivers_Eval <- read.csv("Drivers_Eval.csv")

# Make predictions using the previously trained model, predmodel_two.
# The "type = 'prob'" argument specifies that the prediction probabilities should be returned.
# The predictions are then stored in a variable called final_predictions.
final_predictions <- predict(predmodel3, newdata = Drivers_Eval, type = "prob")

predicted_data <- cbind(Drivers_Eval, Predicted_Probability = final_predictions[, 1])

# Order the combined data by Predicted_Probability in descending order.
# This allows us to easily select the top 2500 drivers based on their predicted probability.
predicted_data <- predicted_data[order(-predicted_data$Predicted_Probability), ]

# Select the top 2500 drivers.
# This takes the first 2500 rows of the ordered data, representing the drivers with the highest predicted probabilities.
top_2500_drivers <- predicted_data[1:2500, ]

# Print the first few rows of the top 2500 drivers to get a quick look at the data.
head(top_2500_drivers)

# Write the top 2500 drivers to a CSV file
write.csv(top_2500_drivers, file = "/Users/vishnuyannam/Downloads/top_2500_drivers.csv")

summary(top_2500_drivers[c("AGE", "VEHICLE_YEAR_MANUF", "TOTAL_NO_OCCUPANTS")]

###EDA FOR TOP 2500 Drivers###
proportions_table_2500 <- top_2500_drivers %>%
  group_by(SEX) %>%
  summarise(count = n()) %>%
  mutate(prop = count / sum(count) * 100) %>%
  ungroup()

proportions_table_2500

# Calculate the median age
median_age <- median(top_2500_drivers$AGE, na.rm = TRUE)
print(median_age)

# Calculate the median manufacturing year
median_manufacturing_year <- median(top_2500_drivers$VEHICLE_YEAR_MANUF, na.rm = TRUE)
print(median_manufacturing_year)

# Calculate the median number of occupants
median_occupants <- median(top_2500_drivers$TOTAL_NO_OCCUPANTS, na.rm = TRUE)
print(median_occupants)

# Calculate the percentage of seat-belt adherence
seat_belt_adherence <- mean(top_2500_drivers$HELMET_BELT_WORN == "Yes", na.rm = TRUE) * 100
print(seat_belt_adherence)

mean_age <- mean(top_2500_drivers$AGE, na.rm = TRUE)
print(mean_age)
