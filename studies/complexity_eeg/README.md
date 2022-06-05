Optimal Selection of Delay, Embedding Dimension and Tolerance for EEG
Complexity Analysis
================

-   [Introduction](#introduction)
-   [Methods](#methods)
-   [Results](#results)
    -   [Optimization of Delay](#optimization-of-delay)
-   [References](#references)

*This study can be referenced by* [*citing the package and the
documentation*](https://neuropsychology.github.io/NeuroKit/cite_us.html).

**We’d like to improve this study, but unfortunately we currently don’t
have the time. If you want to help to make it happen, please contact
us!**

## Introduction

The aim is to assess the optimal complexity parameters.

## Methods

``` r
library(tidyverse)
library(easystats)
library(patchwork)
```

``` r
read.csv("data_delay.csv") |> 
  group_by(Dataset) |>
  summarise(SamplingRate = mean(SamplingRate),
            Lowpass = mean(Lowpass),
            n_Participants = n_distinct(Participant),
            n_Channels = n_distinct(Channel))
## # A tibble: 4 × 5
##   Dataset     SamplingRate Lowpass n_Participants n_Channels
##   <chr>              <dbl>   <dbl>          <int>      <int>
## 1 Lemon                250      50              2         61
## 2 SRM                 1024      50              4         64
## 3 Texas                256      50              4         64
## 4 Wang (2022)          500      50              2         61
```

## Results

### Optimization of Delay

``` r
data_delay <- read.csv("data_delay.csv") |> 
  mutate(Metric = str_remove_all(Metric, fixed(" (FFT)")),
         Metric = fct_relevel(Metric, "Mutual Information", "Autocorrelation", "Displacement"),
         Value = Value/SamplingRate*1000,
         Optimal = Optimal/SamplingRate*1000) |> 
  mutate(Area = str_remove_all(Channel, "[:digit:]|z"),
         Area = substring(Channel, 1, 1),
         Area = case_when(Area == "I" ~ "O",
                          Area == "A" ~ "F",
                          TRUE ~ Area))
         

# summarize(group_by(data_delay, Dataset), Value = max(Value, na.rm=TRUE))
```

``` r
# data_delay |> 
#   mutate(group = paste0(Dataset, "_", Metric)) |> 
#   estimate_density(method="kernel", select="Optimal", at = "group") |> 
#   separate("group", into = c("Dataset", "Metric")) |> 
#   ggplot(aes(x = x, y = y)) +
#   geom_line(aes(color = Dataset)) +
#   facet_wrap(~Metric, scales = "free_y")
```

#### Per Channel

``` r
delay_perchannel <- function(data_delay, dataset="Lemon") {
  data <- filter(data_delay, Dataset == dataset)
  
  
  by_channel <- data |> 
    group_by(Condition, Metric, Area, Channel, Value) |> 
    summarise_all(mean, na.rm=TRUE) 
  by_area <- data |> 
    group_by(Condition, Metric, Area, Value) |> 
    summarise_all(mean, na.rm=TRUE) 
  
  by_channel |> 
    ggplot(aes(x = Value, y = Score, color = Area)) +
    geom_line(aes(group=Channel), alpha = 0.20) +
    geom_line(data=by_area, aes(group=Area), size=1) +
    # geom_vline(xintercept = 10, linetype = "dashed", size = 0.5) +
    facet_wrap(~Condition*Metric, scales = "free_y") +
    see::scale_color_flat_d(palette = "rainbow") +
    scale_y_continuous(expand = c(0, 0)) +
    labs(title = paste0("Dataset: ", dataset), x = NULL, y = NULL) +
    guides(colour = guide_legend(override.aes = list(alpha = 1))) +
    see::theme_modern() +
    theme(plot.title = element_text(face = "plain", hjust = 0))
}

p1 <- delay_perchannel(data_delay, dataset="Lemon")
p2 <- delay_perchannel(data_delay, dataset="Texas")
p3 <- delay_perchannel(data_delay, dataset="SRM")
p4 <- delay_perchannel(data_delay, dataset="Wang (2022)")

p1 / p2 / p3 / p4 + 
  plot_layout(heights = c(2, 1, 1, 2)) + 
  plot_annotation(title = "Optimization of Delay", theme = theme(plot.title = element_text(hjust = 0.5, face = "bold")))
```

![](../../studies/complexity_eeg/figures/unnamed-chunk-7-1.png)<!-- -->

#### Per Subject

``` r
delay_persubject <- function(data_delay, dataset="Lemon") {
  data <- filter(data_delay, Dataset == dataset)

  by_subject <- data |>
    group_by(Condition, Metric, Area, Participant, Value) |>
    summarise_all(mean)
  by_area <- data |>
    group_by(Condition, Metric, Area, Value) |>
    summarise_all(mean)

  by_subject |>
    mutate(group = paste0(Participant, Area)) |>
    ggplot(aes(x = Value, y = Score, color = Area)) +
    geom_line(aes(group=group), alpha = 0.20) +
    geom_line(data=by_area, aes(group=Area), size=1) +
    # geom_vline(xintercept = 10, linetype = "dashed", size = 0.5) +
    facet_wrap(~Condition*Metric, scales = "free_y") +
    see::scale_color_flat_d(palette = "rainbow") +
    scale_y_continuous(expand = c(0, 0)) +
    labs(title = paste0("Dataset: ", dataset), x = NULL, y = NULL) +
    guides(colour = guide_legend(override.aes = list(alpha = 1))) +
    see::theme_modern() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
}

p1 <- delay_persubject(data_delay, dataset="Lemon")
p2 <- delay_persubject(data_delay, dataset="Texas")
p3 <- delay_persubject(data_delay, dataset="SRM")
p4 <- delay_persubject(data_delay, dataset="Wang (2022)")

p1 / p2 / p3 / p4 +
  plot_layout(heights = c(2, 1, 1, 2)) +
  plot_annotation(title = "Optimization of Delay", theme = theme(plot.title = element_text(hjust = 0.5, face = "bold")))
```

![](../../studies/complexity_eeg/figures/unnamed-chunk-8-1.png)<!-- -->

## References
