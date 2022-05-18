**The Structure of Chaos: An Empirical Comparison of Fractal Physiology
Complexity Indices using NeuroKit2**
================

<!-- # Benchmarking and Analysis of Complexity Measures -->
<!-- # Measuring Chaos: Complexity and Fractal Physiology using NeuroKit2 -->
<!-- # Measuring Chaos with NeuroKit2: An Empirical Comparison of Fractal Physiology Complexity Indices -->
<!-- # The Structure of Chaos: An Empirical Comparison of Fractal Physiology Complexity Indices using NeuroKit2 -->

*This study can be referenced by* [*citing the package and the
documentation*](https://neuropsychology.github.io/NeuroKit/cite_us.html).

**We’d like to improve this study, but unfortunately we currently don’t
have the time. If you want to help to make it happen, please contact
us!**

# Introduction

Complexity is an umbrella term for concepts derived from information
theory, chaos theory, and fractal mathematics, used to quantify
unpredictability, entropy, and/or randomness. Using these tools to
characterize signals (a subfield commonly referred to as “fractal
physiology,” Bassingthwaighte et al., 2013) has shown promising results
in physiology in the assessment and diagnostic of the state and health
of living systems Ehlers (1995).

There has been a large and accelerating increase in the number of
complexity indices in the past few decades. These new procedures are
usually mathematically well-defined and theoretically promising.
However, few empirical evidence exist to understand their differences
and similarities. Moreover, some can be very expensive in terms of
computation power and thus, time, which can become an issue in some
applications such as high sampling-rate techniques (e.g., M/EEG) or
real-time settings (brain-computer interface). As such, having a general
view depicting the relationship between the indices with information
about their computation time would be useful, for instance to guide the
indices selection in settings where time or computational power is
limited.

One of the contributing factor of this lack of empirical comparison is
the lack of free, open-source, unified, and easy to use software for
computing various complexity indices. Indeed, most of them are described
mathematically in journal articles, and reusable code is seldom made
available, which limits their further application and validation.
*NeuroKit2* (Makowski et al., 2021) is a Python package for
physiological signal processing that aims at providing the most
comprehensive, accurate and fast pure Python implementations of
complexity indices.

Leveraging this tool, the goal of this study is to empirically compare a
vast number of complexity indices, inspect how they relate to one
another, and extract some recommendations for indices selection, based
on their added-value and computational efficiency. Using *NeuroKit2*, we
will compute more than a hundred complexity indices on various types of
signals, with varying degrees of noise. We will then project the results
on a latent space through factor analysis, and report the most
interesting indices in regards to their representation of the latent
dimensions.

## Methods

![Different types of simulated signals, to which was added 5 types of
noise (violet, blue, white, pink, and brown) with different intensities.
For each signal type, the first row shows the signal with a minimal
amount of noise, and the last with a maximal amount of noise. We can see
that adding Brown noise turns the signal into a Random-walk (i.e., a
Brownian
motion).](../../studies/complexity_benchmark/figures/signals-1.png)

The script to generate the data can be found at
**github.com/neuropsychology/NeuroKit/studies/complexity_benchmark**

We started by generating 5 types of signals, one random-walk, two
oscillatory signals made (one made of harmonic frequencies that results
in a self-repeating - fractal-like - signal), and two complex signals
derived from Lorenz systems (with parameters
(![\\sigma = 10, \\beta = 2.5, \\rho = 28](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csigma%20%3D%2010%2C%20%5Cbeta%20%3D%202.5%2C%20%5Crho%20%3D%2028 "\sigma = 10, \beta = 2.5, \rho = 28"));
and
(![\\sigma = 20, \\beta = 2, \\rho = 30](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csigma%20%3D%2020%2C%20%5Cbeta%20%3D%202%2C%20%5Crho%20%3D%2030 "\sigma = 20, \beta = 2, \rho = 30")),
respectively). Each of this signal was iteratively generated at …
different lengths (). The resulting vectors were standardized and each
were added 5 types of
![(1/f)^\\beta](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%281%2Ff%29%5E%5Cbeta "(1/f)^\beta")
noise (namely violet
![\\beta=-2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D-2 "\beta=-2"),
blue
![\\beta=-1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D-1 "\beta=-1"),
white
![\\beta=0](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D0 "\beta=0"),
pink
![\\beta=1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D1 "\beta=1"),
and brown
![\\beta=2](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cbeta%3D2 "\beta=2")
noise). Each noise type was added at 48 different intensities (linearly
ranging from 0.1 to 4). Examples of generated signals are presented in
**Figure 1**.

The combination of these parameters resulted in a total of 6000 signal
iterations. For each of them, we computed 128 complexity indices, and
additionally basic metric such as the standard deviation (*SD*), the
*length* of the signal and its dominant *frequency*. The parameters used
(such as the time-delay
![\\tau](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau "\tau")
or the embedding dimension) are documented in the data generation
script. For a complete description of the various indices included,
please refer to NeuroKit’s documentation
(<https://neuropsychology.github.io/NeuroKit>).

## Results

The data analysis script, the data and the code for the figures is fully
available at **ADD LINK**. The analysis was performed in R using the
*easystats* collection of packages (Lüdecke et al., 2021; Lüdecke et
al., 2020; Makowski et al., 2020/2022, 2020).

### Computation Time

Despite the relative shortness of the signals considered (a few thousand
points at most), the fully-parallelized data generation script took 12h
to run on a 48-cores machine. After summarizing and sorting the indices
by computation time, the most striking feature are the orders of
magnitude of difference between the fastest and slowest indices. Some of
them are also particularly sensitive to the data length, a property
which combined with computational expensiveness leads to indices being
100,000 slower to compute than other basic metrics.

Multiscale indices are among the slowest, due to their iterative nature
(a given index is computed multiple times on coarse-grained subseries of
the signal). Indices related to Recurrence Quantification Analysis (RQA)
are also relatively slow and don’t scale well with signal length.

<!-- ### Duplicates -->

``` r
data <- df |>
  mutate(i = paste(Signal, Length, Noise_Type, Noise_Intensity, sep = "__")) |>
  select(i, Index, Result) |>
  pivot_wider(names_from = "Index", values_from = "Result") |>
  select(-i)

# pca <- principal_components(data, n=1) |>
#   arrange(desc(sign(PC1)), desc(abs(PC1)))

get_cor <- function(data, plot=FALSE) {
  cor <- correlation::correlation(data, method = "pearson", redundant = TRUE) |>
    correlation::cor_sort(hclust_method = "ward.D2")

  if(plot) {
    p_data <- cor |>
      cor_lower() |>
      mutate(
        Text = insight::format_value(r, zap_small = TRUE, digits = 3),
        Text = str_replace(str_remove(Text, "^0+"), "^-0+", "-"),
        Parameter2 = fct_rev(Parameter2)
      )

    p <- p_data |>
      ggplot(aes(x = Parameter2, y = Parameter1)) +
      geom_tile(aes(fill = r)) +
      # geom_text(aes(label = Text), size = 2) +
      scale_fill_gradient2(low = "#2196F3", mid = "white", high = "#F44336", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Correlation", guide = "legend") +
      scale_x_discrete(expand = c(0, 0)) +
      scale_y_discrete(expand = c(0, 0)) +
      labs(title = "Correlation Matrix of Complexity Indices", x = NULL, y = NULL) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5, face="bold"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
      )
    plot(p)
  }
  cor
}

cor <- get_cor(data)
```

For the subsequent analyses, we removed statistically redundant indices,
such as *PowEn* (identical to *SD*), *CREn (100)* (identical to *CREn
(10)*), and *FuzzyRCMSEn* (identical to *RCMSEn*).

### Correlation

``` r
data <- data |>
  select(
    -`FuzzyRCMSEn`,
    -`CREn (100)`,
    -`PowEn`
  )


cor <- get_cor(data, plot=TRUE)
```

![Correlation
Matrix.](../../studies/complexity_benchmark/figures/correlation-1.png)

The Pearson correlation analysis revealed that complexity indices,
despite their multitude, their unicities and specificities, do indeed
share similarities. They form clusters, with two major ones easily
appearing to the naked eye (the blue and the red groups). These two
anti-correlated groups are driven by the fact that some indices, by
design, index the “predictability”, whereas other the “randomness”, and
thus are negatively related to one-another (see **Figure 2**).

### Factor Analysis

``` r
r <- correlation::cor_smooth(as.matrix(cor))
```

``` r
n <- parameters::n_factors(data, cor = r, n_max=20)
n
## # Method Agreement Procedure:
## 
## The choice of 14 dimensions is supported by 3 (50.00%) methods out of 6 (Optimal coordinates, Parallel analysis, Kaiser criterion).

plot(n) +
  see::theme_modern()
```

![Agreement procedure for the optimal number of
factors.](../../studies/complexity_benchmark/figures/nfactors-1.png)

``` r
rez <- parameters::factor_analysis(data, cor = r, n = 14, rotation = "varimax", sort = TRUE, fm="mle")
# rez <- parameters::principal_components(data, n = 15, sort = TRUE)
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

# Sort by sign too
names(closest) <- rev(levels(loadings$Variable))

idx_order <- loadings |>
  mutate(Closest = closest[as.character(loadings$Variable)],
         Sign = sign(Loading)) |>
  filter(Component == Closest) |>
  arrange(desc(Component), desc(Sign), desc(abs(Loading))) |>
  pull(Variable) |>
  as.character()

separations <- table(closest)[intersect(levels(loadings$Component), unique(closest))]

selection <- c("ShanEn (D)",
               "NLDFD",
               "SVDEn",
               "AttEn",
               "PSDFD",
               "MFDFA (Mean)",
               "FuzzyMSEn",
               "MSWPEn",
               "MFDFA (Increment)",
               "ShanEn (r)",
               "Hjorth",
               "WPEn")
face <- rep("plain", length(idx_order))
face[idx_order %in% c("SD", "Length", "Noise", "Random", "Frequency")] <- "italic"
face[idx_order %in% selection] <- "bold"


p1 <- loadings |>
  mutate(Variable = fct_relevel(Variable, rev(idx_order))) |>
  ggplot(aes(x = Variable, y = Loading)) +
  geom_bar(aes(fill = Component), stat = "identity") +
  geom_vline(xintercept = c("SD", "Length", "Noise", "Random"), color = "red") +
  geom_vline(xintercept = head(cumsum(separations), -1) + 0.5) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_fill_material_d("rainbow") +
  coord_flip() +
  theme_minimal() +
  guides(fill = guide_legend(reverse = TRUE)) +
  labs(x = NULL) +
  theme(
    axis.text.y = element_text(
      color = rev(colors[closest]),
      face = rev(face),
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
  mutate(Index = fct_relevel(Index, rev(idx_order)),
         Duration = ifelse(is.na(Duration), 0, Duration)) |>
  ggplot(aes(x = log10(Duration), y = Index)) +
  geom_bar(aes(fill = log10(Duration)), stat = "identity") +
  geom_hline(yintercept = head(cumsum(separations), -1) + 0.5) +
  scale_x_reverse(expand = c(0, 0)) +
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

![Factor loadings and computation times of the complexity indices,
colored by the factor they represent the
most.](../../studies/complexity_benchmark/figures/loadings-1.png)

The agreement procedure for the optimal number of factors suggested that
the 125 indices can be mapped on a multidimensional space of 14
orthogonal latent factors, that we extracted using a *varimax* rotation.
We then took interest in the loading profile of each indices, and in
particular the latent dimension that maximally related to each index
(see **Figure 3**).

The first factor is the closest to the largest amount of indices. Many
indices with positive and strong loadings are particularly sensitive to
the deviation of consecutive differences (e.g., *ShanEn - D*, *NLDFD*,
*PFD - D*). It was negatively loaded by indices related to Detrended
Fluctuation Analysis (DFA), which tend to index the presence of
long-term correlations. This latent factor might encapsulate the
predominance of short-term vs. long-term unpredictability. Indices that
are the most representative (positively and negatively) and have a
relatively low computational cost include *ShanEn - D*, *NLDFD*, *PFD -
D*, and *AttEn*, *PSDFD*, *FuzzyMSEn*. The second factor was loaded
maximally by signal *length* and *SD*, and thus might not capture
features of complexity *per se*. Indices the most related to it were
indices known to be sensitive to signal length, such as *ApEn*. The
third factor included multiscale indices, such as *MSWPEn*. The fourth
factor included indices that quantified the diversity of the tendency of
a signal to revisit a past state (within a certain tolerance threshold).
It was positively loaded by *ShanEn - r* and negatively by *RQA -
Reccurence Rate*. The fifth factor was loaded by permutation entropy
indices, such as *WPEn*. The sixth factor was driven by indices that
were based on converting the signal into a number of bins. The seventh
factor was loaded positively by the amount of noise, and negatively by
multifractal indices such as *MFDFA - Increment*, suggesting a
sensitivity to regularity. The last notable result is that indices based
on a symbolization (discretization) of the time series do tend to create
factors alongside the symbolization method. Finally, as a manipulation
check of our factorization method, the random vector does indeed form
its own factor, and doesn’t load unto anything else.

### Correlation Network

For illustration purposes, we also represented the correlation matrix as
a connectivity graph (see **Figure 4**).

``` r
library(ggraph)
extrafont::loadfonts()

g <- cor |>
  cor_lower() |>
  mutate(width = abs(r),
         edgecolor = as.character(sign(r))) |>
  filter(!Parameter1 %in% c("SD", "Length", "Random", "Frequency", "Noise"),
         !Parameter2 %in% c("SD", "Length", "Random", "Frequency", "Noise")) |>
  tidygraph::as_tbl_graph(directed=FALSE) |>
  mutate(importance = tidygraph::centrality_authority(weights = abs(r)),
         group = as.factor(tidygraph::group_louvain(weights = abs(r)))) |>
  tidygraph::activate("edges") |>
  filter(abs(r) > 0.6) |>
  tidygraph::activate("nodes") |>
  filter(!tidygraph::node_is_isolated()) |>
  mutate(colors = closest[name],
         selection = ifelse(name %in% selection, TRUE, FALSE))


p1 <- g |>
  ggraph(layout = 'linear', circular = TRUE) +  # fr # lgl # drl # kk
  ggraph::geom_edge_arc(aes(edge_width=width, edge_colour=edgecolor), alpha=0.66, strength=0.3) +
  # ggraph::geom_conn_bundle(aes(edge_width=width, edge_colour=edgecolor), alpha=0.66) +
  ggraph::geom_node_point(aes(size = importance)) +
  ggraph::geom_node_text(aes(x = x*1.05, y=y*1.05, label = name, angle = -((-node_angle(x, y)+90)%%180)+90, hjust='outward', color=selection)) +
  scale_edge_color_manual(values = c("1" = "#2E7D32", "-1"="#C62828"), guide = "none") +
  scale_edge_width_continuous(range = c(0.005, 0.66), guide = "none") +
  scale_size_continuous(range = c(0.1, 2), guide = "none") +
  scale_fill_material_d(guide= "none") +
  scale_colour_manual(values=c("TRUE" = "red", "FALSE"="black"), guide= "none") +
  ggtitle("Correlation Network") +
  ggraph::theme_graph() +
  theme(plot.title = element_text(face="bold", hjust=0.5)) +
  expand_limits(x = c(-1.25, 1.25), y = c(-1.25, 1.25))

p1
```

![Correlation network of the complexity indices. Only the links where
\|r\| \> 0.6 are
displayed.](../../studies/complexity_benchmark/figures/ggm-1.png)

### Hierarchical Clustering

We ran hierarchical clustering (with a Ward D2 distance) to provide
additional information or confirmation about the groups discussed above.
This allowed us to fine-grain our recommendations of complimentary
complexity indices (see **Figure 5**).

``` r
clust <- data |>
  select(-SD, -Length, -Random, -Frequency, -Noise) |>
  t() |>
  as.data.frame() |>
  dist() |>
  hclust(method = "ward.D2")


clusters <- cutree(clust, h = 150)
colors <- c("red", "black", see::palette_material("rainbow")(max(clusters)))
names(colors) <- c("TRUE", "FALSE", seq(1:max(clusters)))

dat <- clust |>
  create_layout(layout = 'dendrogram', circular = TRUE, repel=TRUE) |>
  attr("graph") |>
  tidygraph::activate("nodes") |>
  mutate(colors = closest[label],
         selection = ifelse(label %in% selection, TRUE, FALSE),
         cluster = as.factor(clusters[label]))


p2 <- dat |>
  tidygraph::activate("edges") |>
  mutate(height = as.data.frame(dat)$height[from]) |>
  ggraph(layout = 'dendrogram', circular = TRUE, repel=TRUE) +
  geom_edge_elbow(aes(edge_width = height)) +
  geom_node_point(aes(filter=leaf, color = cluster), size=3) +
  geom_node_text(aes(x = x*1.05, y=y*1.05, label = label, angle = -((-node_angle(x, y)+90)%%180)+90, hjust='outward', color = selection)) +
  scale_colour_manual(values=colors, guide= "none") +
  scale_edge_width_continuous(range=c(1, 1.15), guide= "none") +
  coord_fixed() +
  ggtitle("Hierarchical Clustering") +
  theme_graph() +
  theme(plot.title = element_text(face="bold", hjust=0.5)) +
  expand_limits(x = c(-1.25, 1.25), y = c(-1.25, 1.25))
p2
```

![Dendrogram representing the hierarchical clustering of the complexity
indices.](../../studies/complexity_benchmark/figures/clustering-1.png)

``` r
# ggsave("figures/g.png", p2, width=13.4, height=13.4)
```

### Visualization

Finally, we visualized the expected value of our selection of indices
for different types of signals under different conditions of noise (see
**Figure 1**).

``` r
model <- mgcv::gam(Result ~ s(Noise_Intensity, by = interaction(Index, Signal)),
            data=df |>
              filter(Index %in% selection) |>
              mutate(Noise_Type = as.factor(Noise_Type)))

estimate_means(model, at = c("Index", "Signal", "Noise_Intensity")) |>
  ggplot(aes(y = Mean, x = Noise_Intensity)) +
  geom_line(aes(color = Signal), size=1) +
  facet_wrap(~Index) +
  scale_linetype_manual(values = c("-2" = 3, "-1" = 4, "0" = 2, "1" = 5, "2" = 1)) +
  theme_classic() +
  scale_color_material("rainbow") +
  theme(panel.grid.major = element_line(colour = "#EEEEEE"),
        strip.background = element_blank()) +
  labs(y = "Standardized Index Value", x = "Noise Intensity", color = "Signal Type")
```

![Visualization of the expected value of a selection of indices
depending on the signal type and of the amount of
noise.](../../studies/complexity_benchmark/figures/models-1.png)

## Discussion

As complexity science grows in size and application, a systematic
approach to compare their “performance” becomes necessary to increase
the clarity and structure of the field. The word *performance* is here
to be understood in a relative sense, as any such endeavor faces the
“hard problem” of complexity science. The indices are sensitive to
specific objective properties of a signal that we consider part of
over-arching concepts such as “complex” and “chaotic”. But it is unclear
how these high-level concepts transfer back, in a top-down fashion, into
a combination of lower-level features, such as short-term vs. long-term
variability, auto-correlation, information, randomness, and so on. As
such, it is conceptually complicated to benchmark complexity measures
against “objectively” complex vs. non-complex signals. In other words,
we know that different characteristics can contribute to the
“complexity” of a signal, but there is not a one-to-one correspondence
between the latter and the former.

Hence the choice of the current paradigm, in which we generated
different types of signals to which we systematically added different
types and amount of perturbations. However, we did not seek at measuring
how complexity indices can discriminate between these features or
systems, nor did we attempt at mimicking real-life signals or scenarios.
The goal was instead to generate enough variability to reliably map the
relationships between the indices.

The plurality of underlying components of empirical complexity (what is
measured by complexity indices) seems to be somewhat confirmed by our
results, showing that complexity indices are more or less sensitive to
various orthogonal latent dimensions. One of the limitation of the
current study has to do with the limited possibilities of interpretation
of these underlying dimensions, and future studies are needed to discuss
them.

Indices that were highlighted as encapsulating information about
different underlying dimensions at a relatively low computational cost
include *ShanEn (D)*, *NLDFD*, *SVDEn*, *AttEn*, *PSDFD*, *MFDFA
(Mean)*, *FuzzyMSEn*, *MSWPEn*, *MFDFA (Increment)*, *ShanEn (r)*,
*Hjorth* and *WPEn*. These indices might be complimentary in offering a
comprehensive profile of the complexity of a time series. Future studies
are needed to analyze the nature of the dominant sensitivities of group
of indices, so that results can be more easily interpreted and
integrated into new studies and novel theories.

## References

<div id="refs" class="references csl-bib-body hanging-indent"
line-spacing="2">

<div id="ref-bassingthwaighte2013fractal" class="csl-entry">

Bassingthwaighte, J. B., Liebovitch, L. S., & West, B. J. (2013).
*Fractal physiology*. Springer.

</div>

<div id="ref-ehlers1995chaos" class="csl-entry">

Ehlers, C. L. (1995). Chaos and complexity: Can it help us to understand
mood and behavior? *Archives of General Psychiatry*, *52*(11), 960–964.

</div>

<div id="ref-lau2021brain" class="csl-entry">

Lau, Z. J., Pham, T., Annabel, S., & Makowski, D. (2021). *Brain
entropy, fractal dimensions and predictability: A review of complexity
measures for EEG in healthy and neuropsychiatric populations*.

</div>

<div id="ref-parametersArticle" class="csl-entry">

Lüdecke, D., Ben-Shachar, M., Patil, I., & Makowski, D. (2020).
Extracting, computing and exploring the parameters of statistical models
using R. *Journal of Open Source Software*, *5*(53), 2445.
<https://doi.org/10.21105/joss.02445>

</div>

<div id="ref-seeArticle" class="csl-entry">

Lüdecke, D., Patil, I., Ben-Shachar, M. S., Wiernik, B. M., Waggoner,
P., & Makowski, D. (2021). <span class="nocase">see</span>: An R package
for visualizing statistical models. *Journal of Open Source Software*,
*6*(64), 3393. <https://doi.org/10.21105/joss.03393>

</div>

<div id="ref-correlationArticle" class="csl-entry">

Makowski, D., Ben-Shachar, M., Patil, I., & Lüdecke, D. (2020). Methods
and algorithms for correlation analysis in R. *Journal of Open Source
Software*, *5*(51), 2306. <https://doi.org/10.21105/joss.02306>

</div>

<div id="ref-modelbasedPackage" class="csl-entry">

Makowski, D., Lüdecke, D., Ben-Shachar, M. S., & Patil, I. (2022).
*<span class="nocase">modelbased</span>: Estimation of model-based
predictions, contrasts and means* (Version 0.7.2.1) \[Computer
software\]. <https://CRAN.R-project.org/package=modelbased> (Original
work published 2020)

</div>

<div id="ref-Makowski2021neurokit" class="csl-entry">

Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F.,
Pham, H., Schölzel, C., & Chen, S. H. A. (2021). NeuroKit2: A python
toolbox for neurophysiological signal processing. *Behavior Research
Methods*, *53*(4), 1689–1696.
<https://doi.org/10.3758/s13428-020-01516-y>

</div>

</div>
