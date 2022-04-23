
<!-- # Benchmarking and Analysis of Complexity Measures -->

# Measuring Chaos: Complexity and Fractal Physiology using NeuroKit2

*This study can be referenced by* [*citing the package and the
documentation*](https://neuropsychology.github.io/NeuroKit/cite_us.html).

**We’d like to improve this study, but unfortunately we currently don’t
have the time. If you want to help to make it happen, please contact
us!**

## Introduction

The goal for NeuroKit is to provide the most comprehensive, accurate and
fastest base Python implementations of complexity indices (fractal
dimension, entropy, etc.).

## Methods

### Data Generation

The script to generate the data can be found at …

Generated 3 types of signals, to which we added different types of
noise.

``` r
library(tidyverse)
library(easystats)
library(patchwork)

df <- read.csv("data_Signals.csv") |> 
  mutate(Method = as.factor(Method),
         Noise = as.factor(Noise),
         Intensity = as.factor(insight::format_value(Noise_Intensity)))

df <- df |> 
  filter(Intensity %in% levels(df$Intensity)[c(1, round(length(levels(df$Intensity)) / 2), length(levels(df$Intensity)))])

make_plot <- function(method = "Random-Walk", title = "Random-Walk", color = "red") {
  df |>
    filter(Method == method) |> 
    ggplot(aes(x = Duration, y = Signal)) + 
    geom_line(color = color) +
    ggside::geom_ysidedensity(aes(x=stat(density))) +
    facet_grid(Intensity ~ Noise, labeller = label_both) +
    labs(y = NULL, title = title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          ggside.panel.border = element_blank(),
          ggside.panel.grid =element_blank(),
          ggside.panel.background = element_blank()) 
}

p1 <- make_plot(method = "Random-Walk", title = "Random-Walk", color = "red") 
p2 <- make_plot(method = "lorenz_10_2.5_28", title = "Lorenz (sigma=10, beta=2.5, rho=28)", color = "blue") 
p3 <- make_plot(method = "lorenz_20_2_30", title = "Lorenz (sigma=20, beta=2, rho=30)", color = "green") 

p1 / p2 / p3 + patchwork::plot_annotation(title = "Examples of Simulated Signals", theme = theme(plot.title = element_text(face = "bold", hjust = 0.5)))
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-2-1.png)<!-- -->

## Results

### Average Computation Time

``` r
df <- read.csv("data_Complexity.csv") |> 
  mutate(Method = as.factor(Method)) 

# Show and filter out NaNs
df[is.na(df$Result), "Index"]
## character(0)
df <- filter(df, !is.na(Result))
```

``` r
order <- df |> 
  group_by(Index) |> 
  summarize(Duration = median(Duration)) |> 
  arrange(Duration) |> 
  mutate(Index = factor(Index, levels = Index))

df <- mutate(df, Index = fct_relevel(Index, as.character(order$Index)))

df |> 
  filter(!Index %in% c("Diff", "SD")) |> 
  ggplot(aes(x = Index, y = Duration)) +
  # geom_violin(aes(fill = Index)) +
  ggdist::stat_slab(side = "bottom", aes(fill = Index), adjust = 3) +
  ggdist::stat_dotsinterval(aes(fill = Index, slab_size = NA)) +
  theme_modern() +
  scale_y_log10() +
  scale_fill_manual(values = colors, guide = "none") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  labs(x = NULL, y = "Computation Time")
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-5-1.png)<!-- -->

### Sensitivity to Signal Length

#### Computation Time

``` r
dfsummary <- df |>
  filter(!Index %in% c("Diff", "SD")) |> 
  mutate(Duration = Duration * 10000) |> 
  group_by(Index, Length) |>
  summarize(CI_low = median(Duration) - sd(Duration),
            CI_high = median(Duration) + sd(Duration),
            Duration = median(Duration))
dfsummary$CI_low[dfsummary$CI_low < 0] <- 0


dfsummary |>
  ggplot(aes(x = Index, y = Duration)) + 
  # geom_hline(yintercept = c(0.001, 0.01, 0.1, 1), linetype = "dotted") +
  geom_line(aes(alpha = Length, group = Length)) +
  geom_point(aes(color = Length)) + 
  theme_modern() +
  scale_y_log10(breaks = rep(10, 5)**seq(0, 4), labels = function(x) sprintf("%g", x)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  guides(alpha = "none") +
  labs(y = "Time to compute", x = NULL, color = "Signal length")
```

![](../../studies/complexity_benchmark/figures/time1-1.png)<!-- -->

``` r
df |> 
  filter(!Index %in% c("Diff", "SD")) |> 
  mutate(Duration = Duration * 10000) |> 
  ggplot(aes(x = as.factor(Length), y = Duration)) +
  # geom_hline(yintercept = c(0.001, 0.01, 0.1, 1), linetype = "dotted") +
  geom_line(data=dfsummary, aes(group = 1)) +
  geom_violin(aes(fill = Length)) +
  facet_wrap(~Index) +
  scale_y_log10(breaks = rep(10, 5)**seq(0, 4), labels = function(x) sprintf("%g", x)) +
  scale_fill_viridis_c(guide = "none") +
  theme_modern() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
```

![](../../studies/complexity_benchmark/figures/time2-1.png)<!-- -->

#### Results

``` r
model <- lm(Result ~ Index / poly(Length, 2), data = df)

parameters::parameters(model, keep = "poly") |> 
  arrange(desc(abs(Coefficient))) |> 
  filter(p < .05)
## Parameter                              | Coefficient |    SE |           95% CI | t(9450) |      p
## --------------------------------------------------------------------------------------------------
## Index [CREn (1000)] * poly(Length, 2)1 |      428.75 | 54.77 | [321.39, 536.11] |    7.83 | < .001
## Index [CREn (1000)] * poly(Length, 2)2 |      264.37 | 54.77 | [157.01, 371.73] |    4.83 | < .001
## Index [Hjorth] * poly(Length, 2)1      |      216.00 | 54.77 | [108.64, 323.36] |    3.94 | < .001

estimate_relation(model) |> 
  ggplot(aes(x = Length, y = Predicted)) +
  geom_ribbon(aes(ymin = CI_low, ymax = CI_high, fill = Index), alpha = 0.1) +
  geom_line(aes(color = Index)) +
  geom_point(data = df, aes(y = Result, color = Index)) + 
  scale_fill_manual(values = colors) +
  scale_color_manual(values = colors) +
  facet_wrap(~Index, scales = "free")
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-6-1.png)<!-- -->

### Correlation

``` r
data <- df |> 
  mutate(i = paste(Signal, Length, Noise, Noise_Intensity, sep = "__")) |> 
  select(i, Index, Result) |> 
  pivot_wider(names_from = "Index", values_from = "Result") |> 
  select(-i) 

cor <- correlation::correlation(data, method = "spearman", redundant = TRUE) |> 
  cor_sort(hclust_method = "ward.D2")

cor_lower <- function(cor) {
  m <- as.matrix(cor)
  
  tri <- upper.tri(m, diag = FALSE)
  rownames(tri) <- rownames(m)
  colnames(tri) <- colnames(m)
  
  toremove <- c()
  
  for(param1 in rownames(m)) {
    for(param2 in colnames(m)) {
      if(tri[param1, param2] == FALSE) {
        toremove <- c(toremove, which(cor$Parameter1 == param1 & cor$Parameter2 == param2))
      }
    } 
  }

  cor[-toremove, ]
}

cor |> 
  cor_lower() |> 
  mutate(Text = insight::format_value(rho, zap_small=TRUE, digits = 3),
         Text = str_replace(str_remove(Text, "^0+"), "^-0+", "-"),
         Parameter2 = fct_rev(Parameter2)) |> 
  ggplot(aes(x = Parameter2, y=Parameter1)) +
  geom_tile(aes(fill = rho)) +
  geom_text(aes(label = Text), size = 2) +
  scale_fill_gradient2(low = '#2196F3', mid = 'white', high = '#F44336', midpoint = 0, limit = c(-1, 1), space = 'Lab', name = 'Correlation', guide = 'legend') +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(title = "Correlation Matrix of Complexity Indices", x = NULL, y = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust = 1),
        plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) 
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-7-1.png)<!-- -->

### Duplicates

-   **CREn (D)**, **PFD (D)** and **ShanEn (D)**
    -   keep **PFD (D)** because it’s faster.
-   **CREn (B)**, and **ShanEn (B)**
    -   keep **ShanEn (B)** because it’s faster.
-   **CREn (r)**, **PFD (r)** and **ShanEn (r)**
    -   keep **PFD (r)** because it’s faster.
-   **SVDEn**, and **FI**
    -   keep **SVDEn** because it’s positively correlated with the rest.

``` r

cor |> 
  cor_lower() |> 
  arrange(desc(abs(rho))) |> 
  filter(Parameter1 != Parameter2) |> 
  filter(abs(rho) > .95)
## # Correlation Matrix (spearman-method)
## 
## Parameter1            |       Parameter2 |   rho |         95% CI |        S |         p
## ----------------------------------------------------------------------------------------
## PSDFD (Hasselman2013) | PSDFD (Voss1998) | -1.00 | [-1.00, -1.00] | 2.36e+06 | < .001***
## ShanEn (B)            |         CREn (B) |  1.00 | [ 1.00,  1.00] |     0.00 | < .001***
## CREn (D)              |          PFD (D) |  1.00 | [ 1.00,  1.00] | 2.62e-10 | < .001***
## CREn (D)              |       ShanEn (D) |  1.00 | [ 1.00,  1.00] | 2.62e-10 | < .001***
## PFD (D)               |       ShanEn (D) |  1.00 | [ 1.00,  1.00] | 2.62e-10 | < .001***
## SVDEn                 |               FI | -1.00 | [-1.00, -1.00] | 2.36e+06 | < .001***
## FuzzyEn               |        FuzzyApEn |  0.99 | [ 0.99,  0.99] |  8172.00 | < .001***
## FuzzycApEn            |               FI | -0.99 | [-0.99, -0.99] | 2.35e+06 | < .001***
## NLDFD                 |               RR |  0.99 | [ 0.99,  0.99] | 11532.00 | < .001***
## PFD (r)               |       ShanEn (r) |  0.99 | [ 0.99,  0.99] | 12160.37 | < .001***
## SVDEn                 |       FuzzycApEn |  0.99 | [ 0.99,  0.99] | 12602.00 | < .001***
## NLDFD                 |       ShanEn (r) |  0.99 | [ 0.98,  0.99] | 15180.13 | < .001***
## H (corrected)         |  H (uncorrected) |  0.98 | [ 0.98,  0.99] | 18408.00 | < .001***
## NLDFD                 |          PFD (r) |  0.98 | [ 0.98,  0.99] | 19995.77 | < .001***
## FuzzyApEn             |       FuzzycApEn |  0.98 | [ 0.97,  0.99] | 22418.00 | < .001***
## FuzzyApEn             |            SVDEn |  0.98 | [ 0.97,  0.98] | 29488.00 | < .001***
## FuzzyApEn             |               FI | -0.97 | [-0.98, -0.97] | 2.33e+06 | < .001***
## RR                    |       ShanEn (r) |  0.97 | [ 0.97,  0.98] | 30486.70 | < .001***
## RR                    |          PFD (r) |  0.97 | [ 0.96,  0.98] | 31279.30 | < .001***
## CREn (r)              |       ShanEn (r) |  0.97 | [ 0.96,  0.98] | 31910.21 | < .001***
## FuzzyEn               |       FuzzycApEn |  0.97 | [ 0.96,  0.98] | 39170.00 | < .001***
## CREn (r)              |          PFD (r) |  0.96 | [ 0.95,  0.97] | 48561.36 | < .001***
## FuzzyEn               |            SVDEn |  0.96 | [ 0.94,  0.97] | 48900.00 | < .001***
## SFD                   |         PFD (10) | -0.96 | [-0.97, -0.94] | 2.31e+06 | < .001***
## CREn (r)              |            NLDFD |  0.96 | [ 0.94,  0.97] | 51683.72 | < .001***
## FuzzyEn               |               FI | -0.96 | [-0.97, -0.94] | 2.31e+06 | < .001***
## CREn (100)            |      CREn (1000) |  0.95 | [ 0.94,  0.96] | 55968.00 | < .001***
## 
## p-value adjustment method: Holm (1979)
## Observations: 192


# Duplicates 
# ===========
averagetime <- arrange(summarize(group_by(df, Index), Duration = mean(Duration)), Duration)

filter(averagetime, Index %in% c("CREn (D)", "PFD (D)", "ShanEn (D)"))
## # A tibble: 3 x 2
##   Index      Duration
##   <fct>         <dbl>
## 1 PFD (D)    0.000161
## 2 ShanEn (D) 0.000515
## 3 CREn (D)   0.000569
filter(averagetime, Index %in% c("ShanEn (B)", "CREn (B)"))
## # A tibble: 2 x 2
##   Index      Duration
##   <fct>         <dbl>
## 1 ShanEn (B) 0.000577
## 2 CREn (B)   0.000634
filter(averagetime, Index %in% c("ShanEn (r)", "PFD (r)", "CREn (r)"))
## # A tibble: 3 x 2
##   Index      Duration
##   <fct>         <dbl>
## 1 PFD (r)    0.000103
## 2 ShanEn (r) 0.000458
## 3 CREn (r)   0.000529
filter(averagetime, Index %in% c("SVDEn", "FI"))
## # A tibble: 2 x 2
##   Index Duration
##   <fct>    <dbl>
## 1 SVDEn 0.000129
## 2 FI    0.000205

# NLFD | RR
# NLFD | RR
# - Drop RR because it's slower
# H (uncorrected) | H (corrected)
# - ??
# SVDEn | FuzzyEn
# - Drop FuzzyEn because it's slower

# Hasselman positively correlated with most of the others
# - RR: much slower



data <- data |> 
  select(
    -`CREn (D)`, -`ShanEn (D)`,
    - `ShanEn (B)`,
    - `CREn (r)`, -`ShanEn (r)`,
    -FI
    # -`PSDFD (Voss1998)`  
  )
```

### Hierarchical CLustering

``` r
n <- parameters::n_clusters(as.data.frame(t(data)), standardize = FALSE)
plot(n)
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-9-1.png)<!-- -->

``` r
rez <- parameters::cluster_analysis(as.data.frame(t(data)), 
                                    standardize = FALSE, 
                                    n=7, 
                                    method="hclust", 
                                    hclust_method="ward.D2")
# plot(rez)

attributes(rez)$model |> 
  plot(hang = -1)
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-9-2.png)<!-- -->

### Factor Analysis

``` r
plot(parameters::n_factors(data))
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-10-1.png)<!-- -->

``` r
rez <- parameters::factor_analysis(data, cor = cor(data, method = "spearman"), n = 6, rotation = "varimax", sort = TRUE)
rez
## # Rotated loadings from Factor Analysis (varimax-rotation)
## 
## Variable              |  MR1  |  MR5  |  MR2  |    MR4    |    MR3    |    MR6    | Complexity | Uniqueness
## -----------------------------------------------------------------------------------------------------------
## CD                    | 0.88  | 0.34  | -0.12 |   -0.08   |   -0.09   |   0.14    |    1.43    |    0.07   
## FuzzyEn               | 0.85  | 0.36  | 0.06  |   -0.20   |   -0.07   |   0.21    |    1.63    |    0.05   
## FuzzyApEn             | 0.83  | 0.38  | 0.06  |   -0.23   |   -0.05   |   0.24    |    1.79    |    0.04   
## SampEn                | 0.83  | 0.14  | -0.23 |   -0.30   |   -0.14   |   0.02    |    1.57    |    0.14   
## cApEn                 | 0.79  | 0.27  | 0.18  |   -0.30   |   -0.05   |   0.20    |    1.83    |    0.14   
## FuzzycApEn            | 0.78  | 0.42  | 0.15  |   -0.29   |   -0.03   |   0.25    |    2.21    |    0.05   
## SVDEn                 | 0.75  | 0.50  | 0.11  |   -0.28   |   -0.02   |   0.22    |    2.34    |    0.04   
## PFD (r)               | 0.68  | 0.42  | 0.25  |   -0.27   |   -0.02   |   0.41    |    3.15    |    0.05   
## NLDFD                 | 0.68  | 0.45  | 0.21  |   -0.30   | -8.40e-03 |   0.40    |    3.14    |    0.04   
## RR                    | 0.66  | 0.46  | 0.26  |   -0.33   |   -0.01   |   0.36    |    3.37    |    0.05   
## DiffEn                | 0.65  | 0.05  | -0.46 |   -0.15   |   -0.01   |   0.15    |    2.08    |    0.31   
## ShanEn (A)            | 0.48  | 0.16  | -0.16 |   0.07    |   -0.08   | -1.31e-03 |    1.59    |    0.71   
## H (uncorrected)       | -0.30 | -0.87 | 0.12  |   0.23    |   -0.10   |   -0.08   |    1.47    |    0.08   
## H (corrected)         | -0.35 | -0.86 | -0.02 |   0.22    |   -0.08   |   -0.12   |    1.53    |    0.07   
## ShanEn (C)            | -0.20 | -0.83 | 0.31  |   0.18    |   -0.12   |   -0.15   |    1.65    |    0.10   
## SPEn                  | 0.39  | 0.79  | -0.12 |   -0.08   |   0.02    |   0.11    |    1.58    |    0.20   
## PFD (C)               | -0.07 | -0.79 | 0.52  |   0.16    |   -0.14   |   -0.02   |    1.94    |    0.06   
## HFD                   | 0.40  | 0.78  | -0.09 |   -0.25   |   0.04    |   0.03    |    1.76    |    0.15   
## Hjorth                | -0.50 | -0.74 | 0.04  |   0.28    |   -0.07   |   -0.31   |    2.53    |    0.02   
## CREn (C)              | -0.11 | -0.72 | 0.22  |   0.17    |   -0.05   |   -0.09   |    1.41    |    0.38   
## PSDFD (Hasselman2013) | 0.54  | 0.62  | -0.33 |   0.02    |   -0.04   |   -0.24   |    2.86    |    0.15   
## PSDFD (Voss1998)      | -0.54 | -0.62 | 0.33  |   -0.02   |   0.04    |   0.24    |    2.86    |    0.15   
## PFD (D)               | 0.22  | 0.52  | 0.12  |   -0.39   |   -0.02   |   -0.10   |    2.48    |    0.51   
## PFD (3)               | 0.06  | -0.14 | 0.95  | -5.62e-03 |   -0.03   |   0.16    |    1.11    |    0.05   
## SD                    | -0.20 | 0.02  | -0.90 |   0.02    |   0.03    |   -0.27   |    1.28    |    0.07   
## ShanEn (1000)         | -0.31 | -0.05 | -0.82 |   0.33    | 4.38e-03  |   -0.10   |    1.66    |    0.11   
## PFD (10)              | -0.35 | -0.38 | 0.75  | 9.72e-03  |   -0.01   |   -0.13   |    2.03    |    0.15   
## SFD                   | 0.43  | 0.39  | -0.74 |   -0.03   |   0.03    |   0.20    |    2.39    |    0.06   
## ApEn                  | 0.23  | 0.11  | -0.72 |   -0.12   |   -0.05   |   -0.08   |    1.34    |    0.39   
## PFD (A)               | 0.22  | -0.10 | 0.67  |   0.07    |   -0.55   |   0.16    |    2.39    |    0.16   
## KFD                   | 0.45  | 0.48  | -0.52 |   -0.13   |   0.04    |   0.29    |    3.74    |    0.19   
## CREn (B)              | -0.10 | -0.19 | -0.08 |   0.86    |   0.07    |   -0.10   |    1.18    |    0.19   
## ShanEn (3)            | -0.42 | -0.25 | 0.11  |   0.80    |   -0.03   |   -0.04   |    1.80    |    0.10   
## ShanEn (10)           | -0.35 | -0.34 | 0.12  |   0.79    |   -0.01   |   0.02    |    1.85    |    0.12   
## ShanEn (100)          | -0.36 | -0.26 | -0.17 |   0.75    |   -0.02   |   0.10    |    1.89    |    0.20   
## PFD (B)               | 0.15  | -0.19 | 0.58  |   0.73    | 6.28e-03  |   0.05    |    2.18    |    0.08   
## CREn (100)            | -0.08 | 0.09  | -0.02 |   0.03    |   0.99    |   0.06    |    1.04    |  4.11e-03 
## CREn (10)             | -0.06 | 0.07  | -0.03 |   0.01    |   0.94    |   0.08    |    1.04    |    0.10   
## CREn (1000)           | -0.10 | 0.13  | -0.07 | 1.84e-03  |   0.94    |   0.05    |    1.08    |    0.09   
## CREn (A)              | 0.10  | 0.05  | -0.05 |   -0.12   |   0.43    |   -0.11   |    1.49    |    0.77   
## CREn (3)              | -0.08 | -0.07 | 0.05  |   0.11    |   0.31    |   0.05    |    1.66    |    0.88   
## PFD (100)             | -0.50 | -0.10 | -0.11 |   -0.08   |   -0.03   |   -0.79   |    1.80    |    0.10   
## PFD (1000)            | -0.38 | -0.11 | -0.14 |   -0.08   |   -0.11   |   -0.69   |    1.81    |    0.33   
## SDAFD                 | -0.27 | -0.31 | -0.17 |   0.17    |   0.06    |   -0.33   |    4.13    |    0.66   
## 
## The 6 latent factors (varimax rotation) accounted for 80.97% of the total variance of the original data (MR1 = 22.42%, MR5 = 19.84%, MR2 = 14.70%, MR4 = 10.30%, MR3 = 7.85%, MR6 = 5.85%).
```

## References
