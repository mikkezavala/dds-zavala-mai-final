library(shiny)
library(ggplot2)
library(dplyr)

wine_train = read.csv('./Wine_Train.csv', header = TRUE)
train_data = wine_train[, -1]

# UI
ui = fluidPage(
  titlePanel("Wine Quality Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("feature", "Select Feature:",
                  choices = colnames(train_data)[-which(colnames(train_data) == "quality")],
                  selected = "alcohol"),
      radioButtons("plotType", "Choose Plot:",
                   choices = c("Box Plot" = "Box", "Histogram" = "Histogram"),
                   selected = "Box"),
      conditionalPanel(
        condition = "input.plotType == 'Histogram'",
        sliderInput("bins", "Number of Bins:", min = 5, max = 30, value = 10)
      ),
      sliderInput("qualityFilter", "Filter by Quality:",
                  min = min(train_data$quality), 
                  max = max(train_data$quality), 
                  value = c(min(train_data$quality), max(train_data$quality))),
      hr(),
      h3("Feature Summary"),
      verbatimTextOutput("summaryOutput")
    ),
    
    mainPanel(
      h3("Scatter Plot"),
      textOutput("correlationText"),
      plotOutput("scatterPlot"),
      hr(),
      plotOutput("dynamicPlot"),
      hr(),
      textOutput("plotDescription")
    )
  )
)

server = function(input, output) {
  
  filtered_data = reactive({
    train_data %>%
      filter(quality >= input$qualityFilter[1], quality <= input$qualityFilter[2])
  })
  
  output$summaryOutput = renderPrint({
    summary(filtered_data()[[input$feature]])
  })
  
  output$correlationText = renderText({
    corr = cor(filtered_data()[[input$feature]], filtered_data()$quality, use = "complete.obs")
    paste("Correlation between", input$feature, "and quality:", round(corr, 2))
  })
  
  output$scatterPlot = renderPlot({
    ggplot(filtered_data(), aes_string(x = input$feature, y = "quality")) +
      geom_point(color = "#4B0082") +
      geom_smooth(method = "lm", color = "red", se = FALSE) +
      labs(x = input$feature, y = "Quality") +
      theme_minimal() +
      theme(
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 14)
      )
  })
  

  output$dynamicPlot = renderPlot({
    if (input$plotType == "Box") {
      anova_result = aov(as.formula(paste(input$feature, "~ as.factor(quality)")), data = filtered_data())
      p_value = summary(anova_result)[[1]]["Pr(>F)"][1]
      
      ggplot(filtered_data(), aes_string(x = "as.factor(quality)", y = input$feature)) +
        geom_boxplot(fill = "#8A2BE2", color = "#4B0082") +
        labs(x = "Quality", y = input$feature, 
             title = paste("Box Plot of", input$feature, "by Quality", 
                           "(p =", signif(p_value, 3), ")")) +
        theme_minimal() +
        theme(
          axis.title = element_text(size = 16),
          axis.text = element_text(size = 14),
          plot.title = element_text(size = 18, hjust = 0.5, face = "bold")
        )
    } else if (input$plotType == "Histogram") {
      hist_data = filtered_data() %>%
        mutate(bin = cut(get(input$feature), breaks = input$bins)) %>%
        group_by(bin) %>%
        summarise(count = n()) %>%
        mutate(percentage = count / sum(count) * 100)
      
      ggplot(hist_data, aes(x = bin, y = count)) +
        geom_col(fill = "#8A2BE2", color = "#4B0082") +
        geom_text(aes(label = paste0(count, " (", round(percentage, 1), "%)")), 
                  vjust = -0.5, color = "black", size = 5) +
        labs(x = input$feature, y = "Count", title = paste("Histogram of", input$feature)) +
        theme_minimal() +
        theme(
          axis.title = element_text(size = 16),
          axis.text = element_text(size = 14),
          plot.title = element_text(size = 18, hjust = 0.5, face = "bold")
        )
    }
  })
  
  output$plotDescription = renderText({
    paste("Explore the relationship between", input$feature, "and wine quality. Adjust the quality filter to explore subsets.")
  })
}

shinyApp(ui = ui, server = server)