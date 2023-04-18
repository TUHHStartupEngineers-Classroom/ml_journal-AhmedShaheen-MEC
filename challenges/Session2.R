library(tidymodels)  # for the parsnip package, along with the rest of tidymodels

# Helper packages
library(broom.mixed) # for converting bayesian models to tidy tibbles

library(rstanarm)
library(nycflights13)
library(rsample)
library(yardstick)
library(parsnip)
library(dials)
library(workflows)
library(modeldata)  # for the cells data
library(vip)         # for variable importance plots


# Data set
bike_data_tbl <- readRDS("raw_data/bike_orderlines.rds")

ggplot(bike_data_tbl,
       aes(x = price, 
           y = weight, 
           group = category_1, 
           col = category_1)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  scale_color_manual(values=c("#2dc6d6", "#d65a2d", "#d6af2d", "#8a2dd6", "5"))

linear_reg()
## Linear Regression Model Specification (regression)

lm_mod <- linear_reg() %>% 
  set_engine("lm")

lm_fit <- lm_mod %>% 
  fit(weight ~ price * category_1, 
      data = bike_data_tbl)

tidy(lm_fit)

new_points <- expand.grid(price = 20000, 
                          category_1 = c("E-Bikes", "Hybrid / City", "Mountain", "Road"))
new_points

mean_pred <- predict(lm_fit, new_data = new_points)
mean_pred

conf_int_pred <- predict(lm_fit, 
                         new_data = new_points, 
                         type = "conf_int")
conf_int_pred

plot_data <- new_points %>% 
  bind_cols(mean_pred) %>% 
  bind_cols(conf_int_pred)


ggplot(plot_data, aes(x = category_1)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, 
                    ymax = .pred_upper),
                width = .2) + 
  labs(y = "Bike weight", x = "Category") 


#############################Bayesian analysis. #################
# set the prior distribution
prior_dist <- rstanarm::student_t(df = 1)

set.seed(123)

# make the parsnip model
bayes_mod <- linear_reg() %>% 
             set_engine("stan",
             prior_intercept = prior_dist, 
             prior = prior_dist) 

# train the model
bayes_fit <-  bayes_mod %>% 
  fit(weight ~ price * category_1, 
      data = bike_data_tbl)

print(bayes_fit, digits = 5)

##################3 Pre-processing
set.seed(123)

flights <-
  flights %>% 
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = as.Date(time_hour)
  ) %>% 
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>% 
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance, 
         carrier, date, arr_delay, time_hour) %>% 
  # Exclude missing data
  na.omit() %>% 
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor)

flights %>% 
  count(arr_delay) %>% 
  mutate(prop = n/sum(n))

flights %>% 
  skimr::skim(dest, carrier) 

# Fix the random numbers by setting the seed 
# This enables the analysis to be reproducible when random numbers are used 
set.seed(555)
# Put 3/4 of the data into the training set 
data_split <- initial_split(flights, prop = 3/4)

# Create data frames for the two sets:
trasssin_data <- training(data_split)
test_data  <- testing(data_split)

flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID")

summary(flights_rec)

flights %>% 
  distinct(date) %>% 
  mutate(numeric_date = as.numeric(date))

flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal(), -all_outcomes())

trasssin_data %>% 
  distinct(dest) %>% 
  anti_join(trasssin_data)

lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")

flights_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(flights_rec)
flights_wflow

flights_fit <- 
  flights_wflow %>% 
  fit(data = train_data)

flights_fit %>% 
  pull_workflow_fit() %>% 
  tidy()

predict(flights_fit, test_data)

flights_pred <- 
  predict(flights_fit, test_data, type = "prob") %>% 
  bind_cols(test_data %>% select(arr_delay, time_hour, flight)) 

flights_pred %>% 
  roc_curve(truth = arr_delay, .pred_late) %>% 
  autoplot()

#Hopefully, it should be clear that the more values we have in 
#the upper left hand corner (high true positives and low false positives) 
#the better our model is doing. It follows that the area under 
#this ROC curve (AUC) will reflect the ability of the model to correctly catergorise the data, 
#with a maximum value of 1 and minimum of 0. A value of around 0.5 is the case 
#where the model output is doing no better than change (the dotted line), 
#while values under 0.5 suggest that the model is performing worse than chance. 
#Here, from visual inspection we can see that the AUC is 1 - 0.4 x 0.2 = 0.92.

flights_pred %>% 
  roc_auc(truth = arr_delay, .pred_late)


###### Evaluation ############

data(cells, package = "modeldata")
cells

cells %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

#Split data

set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

nrow(cell_train)
## 1515
nrow(cell_train)/nrow(cells)


# training set proportions by class
cell_train %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

#test set proportions by class
cell_test %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

#### Modeling using Trees

rf_mod <- 
  rand_forest(trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

set.seed(234)

rf_fit <- 
  rf_mod %>% 
  fit(class ~ ., data = cell_train)
rf_fit

rf_training_pred <- 
  predict(rf_fit, cell_train) %>% 
  bind_cols(predict(rf_fit, cell_train, type = "prob")) %>% 
  # Add the true outcome data back in
  bind_cols(cell_train %>% 
              select(class))

##### Cross validation

et.seed(345)
folds <- vfold_cv(cell_train, v = 10)
folds

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(class ~ .)

set.seed(456)
rf_fit_rs <- 
  rf_wf %>% 
  fit_resamples(folds)

rf_fit_rs


collect_metrics(rf_fit_rs)

################# Tuning ###############

set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

tune_spec <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

tune_spec


tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5)
tree_grid


set.seed(234)
cell_folds <- vfold_cv(cell_train)

set.seed(345)

tree_wf <- workflow() %>%
  add_model(tune_spec) %>%
  add_formula(class ~ .)

tree_res <- 
  tree_wf %>% 
  tune_grid(
    resamples = cell_folds,
    grid = tree_grid
  )

tree_res

tree_res %>% 
  collect_metrics()

tree_res %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

tree_res %>%
  show_best("roc_auc")
## # A tibble: 5 x 8
##   cost_complexity tree_depth .metric .estimator  mean     n std_err .config
##             <dbl>      <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>  
## 1    0.0000000001          4 roc_auc binary     0.865    10 0.00965 Model06
## 2    0.0000000178          4 roc_auc binary     0.865    10 0.00965 Model07
## 3    0.00000316            4 roc_auc binary     0.865    10 0.00965 Model08
## 4    0.000562              4 roc_auc binary     0.865    10 0.00965 Model09
## 5    0.0000000001          8 roc_auc binary     0.859    10 0.0104  Model11

best_tree <- tree_res %>%
  select_best("roc_auc")

best_tree
## # A tibble: 1 x 3
##   cost_complexity tree_depth .config
##             <dbl>      <int> <chr>  
## 1    0.0000000001          4 Model06

final_wf <- 
  tree_wf %>% 
  finalize_workflow(best_tree)

final_wf

final_tree <- 
  final_wf %>%
  fit(data = cell_train) 

final_tree

final_tree %>% 
  pull_workflow_fit() %>% 
  vip()

final_fit <- 
  final_wf %>%
  last_fit(cell_split) 

final_fit %>%
  collect_metrics()
## # A tibble: 2 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.802
## 2 roc_auc  binary         0.860

final_fit %>%
  collect_predictions() %>% 
  roc_curve(class, .pred_PS) %>% 
  autoplot()

