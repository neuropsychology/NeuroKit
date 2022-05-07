
<!-- # Benchmarking and Analysis of Complexity Measures -->
<!-- # Measuring Chaos: Complexity and Fractal Physiology using NeuroKit2 -->

# Measuring Chaos with NeuroKit2, and an Empirical Relationship between Complexity Indices

*This study can be referenced by* [*citing the package and the
documentation*](https://neuropsychology.github.io/NeuroKit/cite_us.html).

**We’d like to improve this study, but unfortunately we currently don’t
have the time. If you want to help to make it happen, please contact
us!**

## Introduction

Complexity is an umbrella term for concepts derived from information
theory, chaos theory, and fractal mathematics, used to quantify the
unpredictability and randomness of a signal. Using these tools to
characterize signals has shown promising results in physiology in the
assessment and diagnostic of the health and state of living systems (a
subfield referred to as “fractal physiology”).

There has been a large and accelerating increase in the number of
complexity indices in the past few decades. They are usually
mathematically well-defined and theoretically promising. However, few
empirical data exist to understand their differences and similarities.
One of the contributing factor is the lack of free, open-source, and
easy to use software for computing various complexity indices. Indeed,
most of them are described mathematically in journal articles, and
reusable code is seldom made available.

The goal for NeuroKit is to provide the most comprehensive, accurate and
fast pure Python implementations of complexity indices (fractal
dimension, entropy, etc.).

In this study, we will compute a vast amount of complexity indices on
various types of signals, with varying degrees of noise. We will then
empirically compare the various metrics and their relationship.

## Methods

### Data Generation

The script to generate the data can be found at …

Generated 5 types of signals, to which we added different types of
noise.

``` r
library(tidyverse)
library(easystats)
library(patchwork)

df <- read.csv("data_Signals.csv") |>
  mutate(
    Method = as.factor(Method),
    Noise = as.factor(Noise),
    Intensity = as.factor(insight::format_value(Noise_Intensity))
  )

df <- df |>
  filter(Intensity %in% levels(df$Intensity)[c(1, round(length(levels(df$Intensity)) / 2), length(levels(df$Intensity)))])

make_plot <- function(method = "Random-Walk", title = "Random-Walk", color = "red") {
  df |>
    filter(Method == method) |>
    ggplot(aes(x = Duration, y = Signal)) +
    geom_line(color = color) +
    ggside::geom_ysidedensity(aes(x = stat(density))) +
    facet_grid(Intensity ~ Noise, labeller = label_both) +
    labs(y = NULL, title = title) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      ggside.panel.border = element_blank(),
      ggside.panel.grid = element_blank(),
      ggside.panel.background = element_blank()
    )
}

p1 <- make_plot(method = "Random-Walk", title = "Random-Walk", color = "red")
p2 <- make_plot(method = "lorenz_10_2.5_28", title = "Lorenz (sigma=10, beta=2.5, rho=28)", color = "blue")
p3 <- make_plot(method = "lorenz_20_2_30", title = "Lorenz (sigma=20, beta=2, rho=30)", color = "green")
p4 <- make_plot(method = "oscillatory", title = "Oscillatory", color = "orange")
p5 <- make_plot(method = "fractal", title = "Fractal", color = "purple")

p1 / p2 / p3 / p4 / p5 + patchwork::plot_annotation(title = "Examples of Simulated Signals", theme = theme(plot.title = element_text(face = "bold", hjust = 0.5)))
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-2-1.png)<!-- -->

## Results

``` r
df <- read.csv("data_Complexity.csv") |>
  mutate(Method = as.factor(Method))

# Show and filter out NaNs
df[is.na(df$Result), "Index"]
df <- filter(df, !is.na(Result))

df[is.infinite(df$Result), "Index"]
df <- filter(df, !is.infinite(Result))

df <- df |>
  group_by(Index) |>
  standardize(select = "Result") |>
  ungroup()
```

### Computation Time

``` r
order <- df |>
  group_by(Index) |>
  summarize(Duration = median(Duration)) |>
  arrange(Duration) |>
  mutate(Index = factor(Index, levels = Index))

df <- mutate(df, Index = fct_relevel(Index, as.character(order$Index)))

df |>
  filter(!Index %in% c("SD", "Length", "Noise", "Random")) |>
  mutate(Duration = Duration * 10000) |>
  ggplot(aes(x = Index, y = Duration)) +
  # geom_violin(aes(fill = Index)) +
  geom_hline(yintercept = 10**seq(0, 5, by = 2), linetype = "dotted", color = "#9E9E9E") +
  geom_hline(yintercept = 10**seq(1, 5, by = 2), color = "#9E9E9E") +
  ggdist::stat_slab(side = "bottom", aes(fill = Index), adjust = 3) +
  ggdist::stat_dotsinterval(aes(fill = Index, slab_size = NA)) +
  theme_modern() +
  scale_y_log10(breaks = 10**seq(0, 5), labels = function(x) sprintf("%g", x)) +
  scale_fill_manual(values = colors, guide = "none") +
  theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust = 1)) +
  labs(x = NULL, y = "Computation Time")
```

![](../../studies/complexity_benchmark/figures/computation_time-1.png)<!-- -->

``` r
dfsummary <- df |>
  filter(!Index %in% c("SD", "Length", "Noise", "Random")) |>
  mutate(Duration = Duration * 10000) |>
  group_by(Index, Length) |>
  summarize(
    CI_low = median(Duration) - sd(Duration),
    CI_high = median(Duration) + sd(Duration),
    Duration = median(Duration)
  )
dfsummary$CI_low[dfsummary$CI_low < 0] <- 0


dfsummary |>
  ggplot(aes(x = Index, y = Duration)) +
  # geom_hline(yintercept = c(0.001, 0.01, 0.1, 1), linetype = "dotted") +
  geom_line(aes(alpha = Length, group = Length)) +
  geom_point(aes(color = Length)) +
  theme_modern() +
  scale_y_log10(breaks = 10**seq(0, 4), labels = function(x) sprintf("%g", x)) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  guides(alpha = "none") +
  labs(y = "Time to compute", x = NULL, color = "Signal length")
```

![](../../studies/complexity_benchmark/figures/time1-1.png)<!-- -->

### Sensitivity to Signal Length

### Sensitivity to Noise

### Correlation

``` r
data <- df |>
  mutate(i = paste(Signal, Length, Noise_Type, Noise_Intensity, sep = "__")) |>
  select(i, Index, Result) |>
  pivot_wider(names_from = "Index", values_from = "Result") |>
  select(-i)



get_cor <- function(data, plot=FALSE) {
  cor <- correlation::correlation(data, method = "spearman", redundant = TRUE) |>
    correlation::cor_sort(hclust_method = "ward.D2")
  p <- cor |>
    cor_lower() |>
    mutate(
      Text = insight::format_value(rho, zap_small = TRUE, digits = 3),
      Text = str_replace(str_remove(Text, "^0+"), "^-0+", "-"),
      Parameter2 = fct_rev(Parameter2)
    ) |>
    ggplot(aes(x = Parameter2, y = Parameter1)) +
    geom_tile(aes(fill = rho)) +
    # geom_text(aes(label = Text), size = 2) +
    scale_fill_gradient2(low = "#2196F3", mid = "white", high = "#F44336", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Correlation", guide = "legend") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(title = "Correlation Matrix of Complexity Indices", x = NULL, y = NULL) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1),
      plot.title = element_text(hjust = 0.5),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
  if(plot) plot(p)
  cor
}


cor <- get_cor(data)
```

### Duplicates

-   **CREn (B)**, and **ShanEn (B)**
    -   Remove *CREn (B)* because it’s slower.
-   **CREn (D)**, **PFD (D)** and **ShanEn (D)**
    -   Remove *CREn (D)* and *ShanEn (D)* because it’s slower.
-   **CREn (r)**, **PFD (r)** and **ShanEn (r)**
    -   Remove *CREn (r)* and *ShanEn (r)* because it’s slower.
-   **PSDFD (Hasselman2013)** and **PSDFD (Voss1998)**
    -   Remove **PSDFD (Voss1998)** because it’s positively correlated
        with the rest.
-   **RangeEn (A)**, **RangeEn (Ac)** and **RangeEn (B)**
    -   Remove **RangeEn (A)**, **RangeEn (Ac)** because they yield
        undefined entropies.
-   **SVDEn**, and **FI**
    -   Remove **FI** because it’s negatively correlated with the rest.
-   **MMSEn**, and **IMSEn**
    -   Remove **MMSEn** because it’s slower.
-   **H (corrected)**, and **H (uncorrected)**
    -   Remove **H (corrected)** because it’s slower.
-   **FuzzyEn**, and **FuzzyApEn**
    -   Remove **FuzzyApEn** because it’s slower.
-   **SVDEn**, and **FuzzycApEn**
    -   Remove **FuzzycApEn** because it’s slower.
-   **CPEn**, and **CRPEn**
    -   Remove **CPEn** to keep the Renyi entropy.
-   **NLDFD**, and **RR**
    -   Remove **RR** because it’s slower.

``` r
data <- data |>
  select(
    -`CREn (B)`,
    -`CREn (D)`, -`ShanEn (D)`,
    -`CREn (r)`, -`ShanEn (r)`,
    # -`CREn (C)`, -`ShanEn (C)`,
    -`PSDFD (Voss1998)`,
    -`RangeEn (A)`, -`RangeEn (Ac)`,
    -FI,
    -MMSEn,
    -`H (corrected)`,
    -FuzzyApEn,
    -FuzzycApEn,
    -CPEn,
    -RR,
    -MFDFA_HDelta,
    # -FuzzyRCMSEn,
    # -`CREn (1000)`, 
    -`CREn (100)`,
    -RQA_VEn, -RQA_LEn
  )

cor <- get_cor(data, plot=TRUE)
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-9-1.png)<!-- -->

<!-- ### Hierarchical CLustering -->
<!-- ```{r message=FALSE, warning=FALSE} -->
<!-- n <- parameters::n_clusters(as.data.frame(t(data)), standardize = FALSE) -->
<!-- plot(n) -->
<!-- rez <- parameters::cluster_analysis(as.data.frame(t(data)),  -->
<!--                                     standardize = FALSE,  -->
<!--                                     n=4,  -->
<!--                                     method="hclust",  -->
<!--                                     hclust_method="ward.D2") -->
<!-- # plot(rez) -->
<!-- attributes(rez)$model |>  -->
<!--   plot(hang = -1) -->
<!-- ``` -->

### Factor Analysis

``` r
r <- correlation::cor_smooth(as.matrix(cor))

plot(parameters::n_factors(data, cor = r))
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-10-1.png)<!-- -->

``` r
rez <- parameters::factor_analysis(data, cor = r, n = 4, rotation = "varimax", sort = TRUE, fm="ml")
# rez <- parameters::principal_components(data, n = 13, sort = TRUE)
# rez

col <- gsub('[[:digit:]]+', '', names(rez)[2])
closest <- colnames(select(rez, starts_with(col)))[apply(select(rez, starts_with(col)), 1, \(x) which.max(abs(x)))]

loadings <- attributes(rez)$loadings_long |>
  mutate(
    Loading = Loading,
    Component = fct_relevel(Component, rev(names(select(rez, starts_with(col))))),
    Variable = fct_rev(fct_relevel(Variable, rez$Variable))
  )

colors <- setNames(see::palette_material("rainbow")(length(levels(loadings$Component))), levels(loadings$Component))


p1 <- loadings |>
  # filter(Variable == "CD") |>
  ggplot(aes(x = Variable, y = Loading)) +
  geom_bar(aes(fill = Component), stat = "identity") +
  geom_vline(xintercept = c("SD", "Length", "Noise", "Random"), color = "red") +
  geom_vline(xintercept = head(cumsum(table(closest)[levels(loadings$Component)]), -1) + 0.5) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_fill_material_d("rainbow") +
  coord_flip() +
  theme_minimal() +
  guides(fill = guide_legend(reverse = TRUE)) +
  labs(x = NULL) +
  theme(
    axis.text.y = element_text(
      color = rev(colors[closest]),
      face = rev(ifelse(rez$Variable %in% c("SD", "Length", "Noise", "Random"), "italic", "plain")),
      hjust = 0.5
    ),
    axis.text.x = element_blank(),
    plot.title = element_text(hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

p2 <- order |>
  mutate(Duration = 1 + Duration * 10000) |>
  filter(Index %in% loadings$Variable) |>
  mutate(Index = fct_relevel(Index, levels(loadings$Variable)),
         Duration = ifelse(is.na(Duration), 0, Duration)) |>
  ggplot(aes(x = log10(Duration), y = Index)) +
  geom_bar(aes(fill = log10(Duration)), stat = "identity") +
  geom_hline(yintercept = head(cumsum(sort(table(closest))), -1) + 0.5) +
  scale_x_reverse(expand = c(0, 0)) +
  # scale_x_log10(breaks = 10**seq(0, 4), labels = function(x) sprintf("%g", x), expand=c(0, 0)) +
  scale_y_discrete(position = "right") +
  scale_fill_viridis_c(guide = "none") +
  labs(x = "Computation Time", y = NULL) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.text.x = element_blank(),
    plot.title = element_text(hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

(p2 | p1) + patchwork::plot_annotation(title = "Computation Time and Factor Loading", theme = theme(plot.title = element_text(hjust = 0.5, face = "bold")))
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-11-1.png)<!-- -->

## References
