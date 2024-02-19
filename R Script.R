library(psych)
library(caret)
library(corrplot)
library(glmnet)
library(Metrics)
library(stargazer)
library(imputeTS)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(caTools)
library(knitr)

# Reading the file
df <- read.csv("crimes.csv")
df
# Summary
psych::describe(df)
summary(df)

#checking null values
is.na(df)
sum(is.na(df))

# Checking the number of null values in each column
null_counts <- colSums(is.na(df))
print(null_counts)

# Replace missing values in 'REPORTING_AREA' with 'Unknown'
df_cleaned <- df %>%
  mutate(REPORTING_AREA = ifelse(is.na(REPORTING_AREA), "Unknown", REPORTING_AREA))

# Drop rows with missing values in 'Lat' and 'Long'
df_cleaned <- df_cleaned %>%
  drop_na(Lat, Long)

# Remove 'OFFENSE_CODE_GROUP' and 'UCR_PART' columns
df_cleaned <- df_cleaned %>%
  select(-c(OFFENSE_CODE_GROUP, UCR_PART))

# Checking the number of null values after cleaning
null_counts_cleaned <- colSums(is.na(df_cleaned))
print(null_counts_cleaned)
colnames(df_cleaned)

# Subset data
df_subset <- subset(df_cleaned, select = c(INCIDENT_NUMBER, OFFENSE_CODE, YEAR, MONTH, DAY_OF_WEEK, HOUR, SHOOTING))
df_subset
stargazer(df_subset, type="text")

# Scatterplot
ggplot(df_cleaned, aes(x = DISTRICT, y = INCIDENT_NUMBER)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "blue") +
  labs(title = "Total Incidents by District", x = "DISTRICT", y = "INCIDENT_YEAR")


# df is my data frame
top_offenses <- df %>%
  count(OFFENSE_DESCRIPTION) %>%
  arrange(desc(n)) %>%
  slice_head(n = 10)

# Plotting bar plot
ggplot(top_offenses, aes(x = n, y = fct_reorder(OFFENSE_DESCRIPTION, n))) +
  geom_col(fill = "skyblue") +
  labs(title = "Top 10 Most Frequent Offense Descriptions",
       x = "Count",
       y = "Offense Description") +
  theme_minimal()

# Create a pie chart for the SHOOTING column
shootings <- table(df$SHOOTING)
shootings <- shootings[order(names(shootings))]

# Plotting pie chart
ggplot(data.frame(category = names(shootings), count = as.integer(shootings)),
       aes(x = "", y = count, fill = category)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Distribution of Shooting Incidents") +
  theme_void()

# Line Plot

# Assuming df is your data frame
monthly_counts <- table(df$MONTH)
monthly_counts <- monthly_counts[order(names(monthly_counts))]

# Plotting
ggplot(data.frame(month = as.integer(names(monthly_counts)), count = as.integer(monthly_counts)),
       aes(x = month, y = count)) +
  geom_point(shape = 16, size = 3, color = "blue") +
  geom_line(size = 1, color = "blue") +
  labs(title = "Incidents by Month",
       x = "Month",
       y = "Count") +
  scale_x_continuous(breaks = 1:12) +
  theme_minimal()

# Regression
reg <- lm(SHOOTING ~ OFFENSE_CODE + MONTH + DISTRICT + HOUR + YEAR, data = df_cleaned)
reg
summary(reg)
# Add regression line
plot(reg, data = df_cleaned)
abline(lm(SHOOTING ~ OFFENSE_CODE +  MONTH + DISTRICT + HOUR + YEAR, data = df_cleaned), col = "blue")


# Logistic Regression

# Convert DAY_OF_WEEK to a factor with an order
df_cleaned$DAY_OF_WEEK <- factor(df_cleaned$DAY_OF_WEEK, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

# Convert categorical variables to factors (if not already)
df_cleaned$OFFENSE_CODE <- as.factor(df_cleaned$OFFENSE_CODE)
df_cleaned$STREET <- as.factor(df_cleaned$STREET)

df_cleaned$DISTRICT <- as.factor(df_cleaned$DISTRICT)
df_cleaned$REPORTING_AREA <- as.factor(df_cleaned$REPORTING_AREA)

# Apply label encoding
df_cleaned$OFFENSE_CODE <- as.numeric(df_cleaned$OFFENSE_CODE)
df_cleaned$STREET <- as.numeric(df_cleaned$STREET)
df_cleaned$DISTRICT <- as.numeric(df_cleaned$DISTRICT)
df_cleaned$REPORTING_AREA <- as.numeric(df_cleaned$REPORTING_AREA)

# Data splitting 
set.seed(123)
sample <- sample.split(Y = df_cleaned$SHOOTING, SplitRatio = 0.7)
train_df <- df_cleaned[sample, ]
dim(train_df)
test_df <- df_cleaned[!sample, ]
dim(test_df)
# Logistic regression
Log_Mod <- glm(SHOOTING ~ OFFENSE_CODE + STREET + REPORTING_AREA + DISTRICT + YEAR, data = train_df, family = binomial(link="logit"))
summary(Log_Mod)

# Confusion Matrix for train data
prob_train <- predict(Log_Mod, newdata = train_df, type = "response")
pred_clas <- as.factor(ifelse(prob_train >= 0.5, 1, 0))
m <- as.factor(train_df$SHOOTING)
confusionMatrix(pred_clas, m)

# Check the structure of df_subset
str(df_subset)
df_subset$DAY_OF_WEEK <- factor(df_subset$DAY_OF_WEEK, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
df_subset$INCIDENT_NUMBER <- as.factor(df_subset$INCIDENT_NUMBER)


# Calculate correlation matrix
cor_matrix <- cor(df_subset[, c("OFFENSE_CODE", "YEAR", "MONTH", "HOUR", "SHOOTING")])

# Plot correlation matrix
corrplot(cor_matrix, method = "color", tl.pos = "n")


# ANOVA
two.way <- aov(SHOOTING ~ OFFENSE_CODE + STREET + REPORTING_AREA + DISTRICT + YEAR, data = df_cleaned)
summary(two.way)

# LASSO Regression
train_x <- model.matrix(SHOOTING ~ OFFENSE_CODE + YEAR + MONTH + DAY_OF_WEEK + HOUR, train_df)[,-1]
head(train_x)
test_x <- model.matrix(SHOOTING ~ OFFENSE_CODE + YEAR + MONTH + DAY_OF_WEEK + HOUR, test_df)[,-1]
head(test_x)
train_y <- as.numeric(train_df$SHOOTING)
head(train_y)
test_y <- as.numeric(test_df$SHOOTING)
head(test_y)

# Implementing LASSO Regression.
lasso_reggmodel <- cv.glmnet(x = as.matrix(train_df[-1]), y = train_df$SHOOTING, alpha = 1, nfolds = 10)
best_lambda_lasso <- lasso_reggmodel$lambda.min
lasso_predictions <- predict(lasso_reggmodel, s = best_lambda_lasso, newx = as.matrix(test_df[-1]))

# Evaluating Model
lasso_mse <- mse(lasso_predictions, test_df$SHOOTING)

# Fitting LASSO regression model with the cross-validation.
lasso_cv_model <- cv.glmnet(x = as.matrix(train_df[-1]), y = train_df$SHOOTING, alpha = 1, nfolds = 10)
lambda_min_lasso <- lasso_cv_model$lambda.min
lambda_1se_lasso <- lasso_cv_model$lambda.1se
lambda_min_lasso
lambda_1se_lasso

# Plotting results from cv.glmnet for LASSO.
plot(lasso_cv_model)

# Adding a title.
title("LASSO Regression: Cross-Validated Mean Squared Error vs. log(Lambda)")
# Adding labels to axes.
xlabel <- expression(log(lambda))
ylabel <- expression(CV~MSE)
axis(1)

# Fit LASSO regression model on training set with optimal lambda.
lasso_reggmodel <- glmnet(x = as.matrix(train_df[-1]), y = train_df$SHOOTING, alpha = 1, lambda = lambda_min_lasso)
lasso_reggmodel

# Extracting the coefficients.
coeff_lasso <- coef(lasso_reggmodel)
coeff_lasso

# Predicting Grad.Rate on training set using LASSO model.
lasso_train_predictions <- predict(lasso_reggmodel, s = lambda_min_lasso, newx = as.matrix(train_df[-1]))

# Calculating RMSE.
rmse_lasso_train <- sqrt(mean((train_df$SHOOTING - lasso_train_predictions)^2))
rmse_lasso_train

# Predicting the Grad.Rate on test set using LASSO model.
lasso_test_predictions <- predict(lasso_reggmodel, s = lambda_min_lasso, newx = as.matrix(test_df[-1]))

# Calculating RMSE on test set.
rmse_lasso_test <- sqrt(mean((test_df$SHOOTING - lasso_test_predictions)^2))

rmse_lasso_test


