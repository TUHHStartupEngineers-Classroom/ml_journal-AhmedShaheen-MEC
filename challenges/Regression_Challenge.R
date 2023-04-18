# Standard
library(tidyverse)

# Modeling
library(parsnip)

# Preprocessing & Sampling
library(recipes)
library(rsample)

# Modeling Error Metrics
library(yardstick)

# Plotting Decision Trees
library(rpart.plot)

library(tidymodels)  # for the parsnip package, along with the rest of tidymodels

# Helper packages
library(broom.mixed) # for converting bayesian models to tidy tibbles
library(rstanarm)
library(dials)
library(workflows)
library(vip)   


  #Problem definition

  #Which Bike Categories are in high demand?
  #Which Bike Categories are under represented?
  
  #Goal

  #Use a pricing algorithm to determine a new product price in a category gap




category <- "category_2"

bike_features_tbl <- readRDS("challenges/bike_features_tbl.rds")
glimpse(bike_features_tbl)


bike_features_tbl_r <- bike_features_tbl %>% select(model:price, category,`Rear Derailleur`, `Shift Lever`) %>% 
                          mutate(
                            `shimano dura-ace`        = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano dura-ace ") %>% as.numeric(),
                            `shimano ultegra`         = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano ultegra ") %>% as.numeric(),
                            `shimano 105`             = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano 105 ") %>% as.numeric(),
                            `shimano tiagra`          = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano tiagra ") %>% as.numeric(),
                            `Shimano sora`            = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano sora") %>% as.numeric(),
                            `shimano deore`           = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano deore(?! xt)") %>% as.numeric(),
                            `shimano slx`             = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano slx") %>% as.numeric(),
                            `shimano grx`             = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano grx") %>% as.numeric(),
                            `Shimano xt`              = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano deore xt |shimano xt ") %>% as.numeric(),
                            `Shimano xtr`             = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano xtr") %>% as.numeric(),
                            `Shimano saint`           = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano saint") %>% as.numeric(),
                            `SRAM red`                = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram red") %>% as.numeric(),
                            `SRAM force`              = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram force") %>% as.numeric(),
                            `SRAM rival`              = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram rival") %>% as.numeric(),
                            `SRAM apex`               = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram apex") %>% as.numeric(),
                            `SRAM xx1`                = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram xx1") %>% as.numeric(),
                            `SRAM x01`                = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram x01|sram xo1") %>% as.numeric(),
                            `SRAM gx`                 = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram gx") %>% as.numeric(),
                            `SRAM nx`                 = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram nx") %>% as.numeric(),
                            `SRAM sx`                 = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram sx") %>% as.numeric(),
                            `SRAM sx`                 = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram sx") %>% as.numeric(),
                            `Campagnolo potenza`      = `Rear Derailleur` %>% str_to_lower() %>% str_detect("campagnolo potenza") %>% as.numeric(),
                            `Campagnolo super record` = `Rear Derailleur` %>% str_to_lower() %>% str_detect("campagnolo super record") %>% as.numeric(),
                            `shimano nexus`           = `Shift Lever`     %>% str_to_lower() %>% str_detect("shimano nexus") %>% as.numeric(),
                            `shimano alfine`          = `Shift Lever`     %>% str_to_lower() %>% str_detect("shimano alfine") %>% as.numeric()
                          ) %>%
                          select(-c(`Rear Derailleur`, `Shift Lever`)) %>%
                          mutate(id = row_number()) 


# 2.0 TRAINING & TEST SETS ----

##############################################################################################################################
##################3 Pre-processing
# Fix the random numbers by setting the seed 
# This enables the analysis to be reproducible when random numbers are used 
set.seed(seed = 1113)
# Put 3/4 of the data into the training set 
split_obj <- rsample::initial_split(bike_features_tbl_r, prop   = 0.80, 
                                    strata = "category_2")
# Create data frames for the two sets:
train_data <- training(split_obj) %>% na.omit()
test_data  <- testing(split_obj)

bikes_data_set <- 
  recipe(price ~ ., data = train_data) %>% 
  step_rm(model_year) %>%
  update_role(model, category_2, new_role = "ID")%>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  prep()

train_transformed_tbl <- bake(bikes_data_set, train_data)
test_transformed_tbl  <- bake(bikes_data_set, test_data)
##############################################################################################################################

recipe_obj <- recipe(...) %>% 
  step_rm(...) %>% 
  step_dummy(... ) %>% # Check out the argument one_hot = T
  prep()

train_transformed_tbl <- bake(..., ...)
test_transformed_tbl  <- bake(..., ...)





##############################################################################################################################
test_data %>% 
  distinct(dest) %>% 
  anti_join(train_data)

lr_mod <- 
  logistic_reg() %>% 
  set_engine("lm")

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
##############################################################################################################################

lr_mod_b <- linear_reg(mode = "regression") %>%
  set_engine("lm") 



bikes_workflow <- 
  workflow() %>% 
  add_model(lr_mod_b) %>% 
  add_recipe(bikes_data_set)
bikes_workflow

bikes_fit <- 
  bikes_workflow %>% 
  fit(data = train_data)

# Generalized into a function
calc_metrics <- function(model, new_data = test_tbl) {
  
  model %>%
    predict(new_data = new_data) %>%
    
    bind_cols(new_data %>% select(price)) %>%
    yardstick::metrics(truth = price, estimate = .pred)
  
}

bikes_fit %>% calc_metrics(train_data)


bikes_fit %>% 
  pull_workflow_fit() %>% 
  tidy()

bikes_pred <- predict(bikes_fit, test_data)

bikes_pred %>% 
  roc_auc(truth = price, .pred_late)
##############################################################################################################################
#Make predictions using a parsnip model_fit object.

?predict.model_fit 

model_01_linear_lm_simple %>%
  predict(new_data = test_tbl)

?metrics

model_01_linear_lm_simple %>%
  predict(new_data = test_tbl) %>%
  
  bind_cols(test_tbl %>% select(price)) %>%
  
  # Manual approach
  # mutate(residuals = price - .pred) %>% 
  # 
  # summarize(
  #   mae  = abs(residuals) %>% mean(),
  #   rmse = mean(residuals^2)^0.5
  # )
  
  yardstick::metrics(truth = price, estimate = .pred)
##############################################################################################################################


