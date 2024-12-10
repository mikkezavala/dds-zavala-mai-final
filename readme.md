---
title: "DDS 6306 Wine Quality - Final Project"
output: html_document
date: "2024-11-27" s
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r cars}
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(corrplot)

wines = read.csv(
  './Wine_Train.csv',
  header = TRUE
)

sprintf("Data composed of: %d rows and %d columns",
        nrow(wines),
        ncol(wines))
head(wines, n = 5)

emptyData <- colSums(is.na(wines))
totalRows <- nrow(wines)

tibble(Column = names(wines),
       `Total_Rows` = totalRows,
       `NA_Count` = emptyData)

selected_data <- wines |> select(-ID)
corr_matrix <- cor(selected_data, use = "complete.obs")

corrplot(corr_matrix, method = "color")

```

## Detecting correlation (High / Low)

After observing the variables correlation, some of the predictor variables are correlated, with the potential of producing collinearity and questioning independence, which we will asses later in more detail, following predictors present this behavior:

-   Density + Residual Sugar (0.55)

-   Density + Alcohol (-0.7)

-   Total Sulfur Dioxide + Residual Sugar (0.5)

-   Total Sulfur Dioxide + Free Sulfur Dioxide (0.72)

At first sigh without the intention to make inference an interest relationship in the train set found is quality has some positive correlation with alcohol (0.44), and a negative relevant correlation with density (-0.30). As an important clarification this only displaying the correlation and not making any attempt for inferences.
