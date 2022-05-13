
<!-- # Benchmarking and Analysis of Complexity Measures -->
<!-- # Measuring Chaos: Complexity and Fractal Physiology using NeuroKit2 -->
<!-- # Measuring Chaos with NeuroKit2: An Empirical Comparison of Fractal Physiology Complexity Indices -->

# The Structure of Chaos: An Empirical Comparison of Fractal Physiology Complexity Indices using NeuroKit2

*This study can be referenced by* [*citing the package and the
documentation*](https://neuropsychology.github.io/NeuroKit/cite_us.html).

**We’d like to improve this study, but unfortunately we currently don’t
have the time. If you want to help to make it happen, please contact
us!**

## Introduction

Complexity is an umbrella term for concepts derived from information
theory, chaos theory, and fractal mathematics, used to quantify
unpredictability, entropy, and/or randomness. Using these tools to
characterize signals (a subfield commonly referred to as “fractal
physiology,” Bassingthwaighte, Liebovitch, & West, 2013) has shown
promising results in physiology in the assessment and diagnostic of the
state and health of living systems **\[REF\]**.

There has been a large and accelerating increase in the number of
complexity indices in the past few decades. These new procedures are
usually mathematically well-defined and theoretically promising.
However, few empirical evidence exist to understand their differences
and similarities. Moreover, some can be very expensive in terms of
computation power and thus, time, which can become an issue in some
applications such as high sampling-rate techniques (e.g., M.EEG) or
real-time settings (brain-computer interface). As such, having a general
view depicting the relationship between the indices with information
about their computation time would be useful, for instance to guide the
indices selection in settings where time or computational power is
limited.

One of the contributing factor of this lack of empirical comparison is
the lack of free, open-source, and easy to use software for computing
various complexity indices. Indeed, most of them are described
mathematically in journal articles, and reusable code is seldom made
available. NeuroKit2 (Makowski et al., 2021) is a Python package for
physiological signal processing that aims at providing the most
comprehensive, accurate and fast pure Python implementations of
complexity indices (fractal dimension, entropy, information, etc.).

The goal of this study is to empirically compare a vast number of
complexity indices, inspect how they relate to one another, and extract
some recommendations for indices selection, based on their added-value
and computational efficiency. Using NeuroKit2, we will compute more than
a hundred complexity indices on various types of signals, with varying
degrees of noise. We will then project the results on a latent space
through factor analysis, and report the most interesting indices in
regards to their representation of the latent dimensions.

## Methods

![Different types of simulated signals, to which was added 5 types of
noise (violet, blue, white, pink, and brown) with different intensities.
For each signal type, the first row shows the signal with a minimal
amount of noise, and the last with a maximal amount of noise. We can see
that adding Brown noise turns the signal into a Random-walk (i.e., a
Brownian
motion).](../../studies/complexity_benchmark/figures/fig1_signals-1.png)

The script to generate the data can be found at …

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
noise). Each noise type was added at … different intensities (linearly
ranging from 0.1 to 4). Examples of generated signals are presented in
**Figure 1**.

The combination of these parameters resulted in a total of 3200 signal
iterations. For each of them, we computed … indices, and additionally
basic metric such as the SD, the Length of the signal and its mean
frequency. The parameters used (such as the time-delay
![\\tau](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctau "\tau")
or the embedding dimension) are documented in the data generation
script. For a complete description of the various indices used, please
refer to NeuroKit’s documentation
(<https://neuropsychology.github.io/NeuroKit>).

## Results

### Computation Time

![](../../studies/complexity_benchmark/figures/computation_time-1.png)<!-- -->

After summarizing and sorting the indices by computation time, the most
striking feature are the orders of magnitude of difference between the
fastest and slowest indices. Some of them are also particularly
sensitive to the data length, a property which combined with
computational expensiveness leads to indices being 100,000 slower to
compute than other basic metrics.

Multiscale indices are among the slowest, due to their iterative nature
(a given index is computed multiple times on coarse-grained subseries of
the signal).

<!-- ### Duplicates -->
<!-- ```{r message=FALSE, warning=FALSE, fig.width=16, fig.height=15, cache=FALSE} -->
<!-- data <- df |> -->
<!--   mutate(i = paste(Signal, Length, Noise_Type, Noise_Intensity, sep = "__")) |> -->
<!--   select(i, Index, Result) |> -->
<!--   pivot_wider(names_from = "Index", values_from = "Result") |> -->
<!--   select(-i) -->
<!-- # pca <- principal_components(data, n=1) |> -->
<!-- #   arrange(desc(sign(PC1)), desc(abs(PC1))) -->
<!-- get_cor <- function(data, plot=FALSE) { -->
<!--   cor <- correlation::correlation(data, method = "pearson", redundant = TRUE) |> -->
<!--     correlation::cor_sort(hclust_method = "ward.D2") -->
<!--   if(plot) { -->
<!--     p_data <- cor |> -->
<!--       cor_lower() |> -->
<!--       mutate( -->
<!--         Text = insight::format_value(r, zap_small = TRUE, digits = 3), -->
<!--         Text = str_replace(str_remove(Text, "^0+"), "^-0+", "-"), -->
<!--         Parameter2 = fct_rev(Parameter2) -->
<!--       ) -->
<!--     p <- p_data |> -->
<!--       ggplot(aes(x = Parameter2, y = Parameter1)) + -->
<!--       geom_tile(aes(fill = r)) + -->
<!--       # geom_text(aes(label = Text), size = 2) + -->
<!--       scale_fill_gradient2(low = "#2196F3", mid = "white", high = "#F44336", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Correlation", guide = "legend") + -->
<!--       scale_x_discrete(expand = c(0, 0)) + -->
<!--       scale_y_discrete(expand = c(0, 0)) + -->
<!--       labs(title = "Correlation Matrix of Complexity Indices", x = NULL, y = NULL) + -->
<!--       theme_minimal() + -->
<!--       theme( -->
<!--         axis.text.x = element_text(angle = 90, hjust = 1), -->
<!--         plot.title = element_text(hjust = 0.5), -->
<!--         panel.grid.major = element_blank(), -->
<!--         panel.grid.minor = element_blank() -->
<!--       ) -->
<!--     plot(p) -->
<!--   } -->
<!--   cor -->
<!-- } -->
<!-- cor <- get_cor(data) -->
<!-- ``` -->
<!-- ```{r eval=TRUE, message=FALSE, warning=FALSE, include=FALSE} -->
<!-- cor |> -->
<!--   cor_lower() |> -->
<!--   filter(Parameter1 %in% names(data), Parameter2 %in% names(data)) |> -->
<!--   arrange(desc(abs(r)), Parameter1) |> -->
<!--   filter(Parameter1 != Parameter2) |> -->
<!--   filter(abs(r) > .97) |> -->
<!--   select(Parameter1, Parameter2, r) -->
<!-- ``` -->
<!-- We will start by removing redundant indices. -->
<!-- - **CREn (B)**, and **ShanEn (B)** -->
<!--   - Remove *CREn (B)*  because it's slower. -->
<!-- - **CREn (D)**, **PFD (D)** and **ShanEn (D)** -->
<!--   - Remove *CREn (D)* and *ShanEn (D)* because it's slower. -->
<!-- - **CREn (r)**, **PFD (r)** and **ShanEn (r)** -->
<!--   - Remove *CREn (r)* and *ShanEn (r)* because it's slower. -->
<!-- - **PSDFD (Hasselman2013)** and **PSDFD (Voss1998)** -->
<!--   - Remove **PSDFD (Voss1998)** because it's positively correlated with the rest. -->
<!-- - **RangeEn (A)**, **RangeEn (Ac)** and **RangeEn (B)** -->
<!--   - Remove **RangeEn (A)**, **RangeEn (Ac)**  because they yield undefined entropies. -->
<!-- - **SVDEn**, and **FI** -->
<!--   - Remove **FI**  because it's negatively correlated with the rest. -->
<!-- - **MMSEn**, and **IMSEn** -->
<!--   - Remove **MMSEn**  because it's slower. -->
<!-- - **H (corrected)**, and **H (uncorrected)** -->
<!--   - Remove **H (corrected)**  because it's slower. -->
<!-- - **FuzzyEn**, and **FuzzyApEn** -->
<!--   - Remove **FuzzyApEn**  because it's slower. -->
<!-- - **SVDEn**, and **FuzzycApEn** -->
<!--   - Remove **FuzzycApEn**  because it's slower. -->
<!-- - **CPEn**, and **CRPEn** -->
<!--   - Remove **CPEn**  to keep the Renyi entropy. -->
<!-- - **NLDFD**, and **RR** -->
<!--   - Remove **RR**  because it's slower. -->
<!-- ```{r eval=FALSE, message=FALSE, warning=FALSE, cache=FALSE, include=FALSE} -->
<!-- # Duplicates -->
<!-- # =========== -->
<!-- averagetime <- arrange(summarize(group_by(df, Index), Duration = mean(Duration)), Duration) -->
<!-- filter(averagetime, Index %in% c("CREn (D)", "PFD (D)", "ShanEn (D)")) -->
<!-- filter(averagetime, Index %in% c("ShanEn (B)", "CREn (B)")) -->
<!-- filter(averagetime, Index %in% c("ShanEn (r)", "PFD (r)", "CREn (r)")) -->
<!-- filter(averagetime, Index %in% c("ShanEn (C)", "PFD (C)", "CREn (C)")) -->
<!-- filter(averagetime, Index %in% c("CREn (10)", "CREn (100)")) -->
<!-- filter(averagetime, Index %in% c("SVDEn", "FI")) -->
<!-- filter(averagetime, Index %in% c("PSDFD (Hasselman2013)", "PSDFD (Voss1998)")) -->
<!-- filter(averagetime, Index %in% c("MMSEn", "IMSEn")) -->
<!-- filter(averagetime, Index %in% c("H (corrected)", "H (uncorrected)")) -->
<!-- filter(averagetime, Index %in% c("FuzzyEn", "FuzzyApEn")) -->
<!-- filter(averagetime, Index %in% c("RCMSEn", "FuzzyRCMSEn")) -->
<!-- filter(averagetime, Index %in% c("SVDEn", "FuzzycApEn")) -->
<!-- filter(averagetime, Index %in% c("CPEn", "CRPEn")) -->
<!-- filter(averagetime, Index %in% c("NLDFD", "RR")) -->
<!-- ``` -->
<!-- ### Correlation -->
<!-- ```{r message=FALSE, warning=FALSE, cache=FALSE, fig.width=16, fig.height=15} -->
<!-- data <- data |> -->
<!--   select( -->
<!--     -`CREn (B)`, -->
<!--     -`CREn (D)`, -`ShanEn (D)`, -->
<!--     -`CREn (r)`, -`ShanEn (r)`, -->
<!--     -`CREn (C)`, -`ShanEn (C)`, -->
<!--     -`CREn (100)`, -->
<!--     -`PowEn`, -->
<!--     # -`PSDFD (Voss1998)`, -->
<!--     # -`RangeEn (A)`, -`RangeEn (Ac)`, -->
<!--     # -FI, -->
<!--     # -MMSEn, -->
<!--     -`H (corrected)` -->
<!--     # -FuzzyApEn, -->
<!--     # -FuzzycApEn, -->
<!--     # -CPEn, -->
<!--     # -RR, -->
<!--     # -MFDFA_HDelta, -->
<!--     # -FuzzyRCMSEn -->
<!--     # -`CREn (1000)`, -->
<!--     # -`CREn (100)`, -->
<!--     # -RQA_VEn, -RQA_LEn -->
<!--   ) -->
<!-- cor <- get_cor(data, plot=TRUE) -->
<!-- ``` -->
<!-- Complexity indices, despite their multitude, their unicities and specificities, do indeed share similarities. They form clusters (some indices, by design, index the predictability, whereas other the randomness). -->
<!-- #### Graph -->
<!-- ```{r graph, message=FALSE, warning=FALSE, fig.width=14, fig.height=7, include=FALSE, eval=FALSE} -->
<!-- library(ggraph) -->
<!-- g <- cor |> -->
<!--   cor_lower() |> -->
<!--   mutate(width = abs(r), -->
<!--          edgecolor = as.character(sign(r))) |> -->
<!--   filter(!Index %in% c("SD", "Length", "Random")) |> -->
<!--   tidygraph::as_tbl_graph(directed=FALSE) -->
<!-- g |> -->
<!--   mutate(importance = tidygraph::centrality_authority(weights = abs(r)), -->
<!--          group = as.factor(tidygraph::group_louvain(weights = abs(r)))) |> -->
<!--   tidygraph::activate("edges") |> -->
<!--   filter(abs(r) > 0.66) |> -->
<!--   tidygraph::activate("nodes") |> -->
<!--   filter(!tidygraph::node_is_isolated()) |> -->
<!--   mutate(angle = 360 * seq(0, n(), length.out=n()) / sum(n()), -->
<!--          hjust = ifelse(angle > 90 & angle < 270, 1, 0), -->
<!--          angle = ifelse(angle > 90 & angle < 270, 180 + angle, angle)) |> -->
<!--   ggraph(layout = 'circle') +  # fr # lgl # drl # kk -->
<!--   ggraph::geom_edge_arc(aes(edge_width=width, edge_colour=edgecolor), strength=0.3, alpha=0.66) + -->
<!--   ggraph::geom_node_point(aes(size = importance), colour = "black") + -->
<!--   # ggraph::geom_node_label(aes(label = name, fill = group, size=importance), repel=TRUE) + -->
<!--   ggraph::geom_node_text(aes(x = x*1.05, y=y*1.05, label = name, angle = angle, hjust=hjust)) + -->
<!--   scale_edge_color_manual(values = c("1" = "#2E7D32", "-1"="#C62828"), guide = "none") + -->
<!--   scale_edge_width_continuous(range = c(0.005, 0.66), guide = "none") + -->
<!--   scale_size_continuous(range = c(0.1, 2), guide = "none") + -->
<!--   scale_fill_material_d(guide= "none") + -->
<!--   scale_colour_material_d(guide= "none") + -->
<!--   ggraph::theme_graph() + -->
<!--   expand_limits(x = c(-1.5, 1.5), y = c(-1.5, 1.5)) -->
<!-- ``` -->
<!-- ### Misc -->
<!-- #### Sensitivity to Signal Length -->
<!-- ```{r message=FALSE, warning=FALSE, include=FALSE, cache=FALSE} -->
<!-- model <- lm(Result ~ Index / poly(Length, 2), data = filter(df, !Index %in% c("SD", "Length", "Noise", "Random"))) -->
<!-- parameters::parameters(model, keep = "poly.*1") |> -->
<!--   arrange(desc(abs(Coefficient))) |> -->
<!--   filter(p < .05) -->
<!-- estimate_relation(model) |> -->
<!--   ggplot(aes(x = Length, y = Predicted)) + -->
<!--   geom_ribbon(aes(ymin = CI_low, ymax = CI_high, fill = Index), alpha = 0.1) + -->
<!--   geom_line(aes(color = Index)) + -->
<!--   geom_point2( -->
<!--     data = filter(df, Index != "SD"), -->
<!--     aes(y = Result, color = Index), -->
<!--     alpha = 0.1, size = 2 -->
<!--   ) + -->
<!--   scale_fill_manual(values = colors) + -->
<!--   scale_color_manual(values = colors) + -->
<!--   theme(legend.position = "none") + -->
<!--   facet_wrap(~Index, scales = "free") -->
<!-- ``` -->
<!-- #### Sensitivity to Noise -->
<!-- ```{r message=FALSE, warning=FALSE, include=FALSE, cache=FALSE} -->
<!-- model <- lm(Result ~ Index / poly(Noise_Intensity, 2), data = filter(df, !Index %in% c("SD", "Length", "Noise", "Random"))) -->
<!-- parameters::parameters(model, keep = "poly.*1") |> -->
<!--   arrange(abs(p)) |> -->
<!--   filter(p < .05) -->
<!-- estimate_relation(model) |> -->
<!--   ggplot(aes(x = Noise_Intensity, y = Predicted)) + -->
<!--   geom_ribbon(aes(ymin = CI_low, ymax = CI_high, fill = Index), alpha = 0.1) + -->
<!--   geom_line(aes(color = Index)) + -->
<!--   geom_point2( -->
<!--     data = filter(df, Index != "SD"), -->
<!--     aes(y = Result, color = Index), -->
<!--     alpha = 0.1, size = 2 -->
<!--   ) + -->
<!--   scale_fill_manual(values = colors) + -->
<!--   scale_color_manual(values = colors) + -->
<!--   theme(legend.position = "none") + -->
<!--   facet_wrap(~Index, scales = "free_y") -->
<!-- ``` -->
<!-- <!-- ### Hierarchical CLustering -->

–\>

<!-- <!-- ```{r message=FALSE, warning=FALSE} -->

–\>
<!-- <!-- n <- parameters::n_clusters(as.data.frame(t(data)), standardize = FALSE) -->
–\> <!-- <!-- plot(n) --> –\>

<!-- <!-- rez <- parameters::cluster_analysis(as.data.frame(t(data)),  -->

–\>
<!-- <!--                                     standardize = FALSE,  -->
–\> <!-- <!--                                     n=4,  --> –\>
<!-- <!--                                     method="hclust",  --> –\>
<!-- <!--                                     hclust_method="ward.D2") -->
–\> <!-- <!-- # plot(rez) --> –\>

<!-- <!-- attributes(rez)$model |>  -->

–\> <!-- <!--   plot(hang = -1) --> –\> <!-- <!-- ``` --> –\>

<!-- ### Factor Analysis -->
<!-- ```{r message=FALSE, warning=FALSE, fig.width=14, fig.height=7} -->
<!-- r <- correlation::cor_smooth(as.matrix(cor)) -->
<!-- plot(parameters::n_factors(data, cor = r, n_max=20)) -->
<!-- # plot(parameters::n_components(data, cor = r)) -->
<!-- ``` -->
<!-- ```{r message=FALSE, warning=FALSE, fig.width=12, fig.height=18} -->
<!-- rez <- parameters::factor_analysis(data, cor = r, n = 12, rotation = "varimax", sort = TRUE, fm="mle") -->
<!-- # rez <- parameters::principal_components(data, n = 15, sort = TRUE) -->
<!-- # rez -->
<!-- col <- gsub('[[:digit:]]+', '', names(rez)[2]) -->
<!-- closest <- colnames(select(rez, starts_with(col)))[apply(select(rez, starts_with(col)), 1, \(x) which.max(abs(x)))] -->
<!-- loadings <- attributes(rez)$loadings_long |> -->
<!--   mutate( -->
<!--     Loading = Loading, -->
<!--     Component = fct_relevel(Component, rev(names(select(rez, starts_with(col))))), -->
<!--     Variable = fct_rev(fct_relevel(Variable, rez$Variable)) -->
<!--   ) -->
<!-- colors <- setNames(see::palette_material("rainbow")(length(levels(loadings$Component))), levels(loadings$Component)) -->
<!-- # Sort by sign too -->
<!-- names(closest) <- rev(levels(loadings$Variable)) -->
<!-- idx_order <- loadings |> -->
<!--   mutate(Closest = closest[as.character(loadings$Variable)], -->
<!--          Sign = sign(Loading)) |> -->
<!--   filter(Component == Closest) |> -->
<!--   arrange(desc(Component), desc(Sign), desc(abs(Loading))) |> -->
<!--   pull(Variable) |> -->
<!--   as.character() -->
<!-- separations <- table(closest)[intersect(levels(loadings$Component), unique(closest))] -->
<!-- p1 <- loadings |> -->
<!--   mutate(Variable = fct_relevel(Variable, rev(idx_order))) |> -->
<!--   # filter(Variable == "CD") |> -->
<!--   ggplot(aes(x = Variable, y = Loading)) + -->
<!--   geom_bar(aes(fill = Component), stat = "identity") + -->
<!--   geom_vline(xintercept = c("SD", "Length", "Noise", "Random"), color = "red") + -->
<!--   geom_vline(xintercept = head(cumsum(separations), -1) + 0.5) + -->
<!--   scale_y_continuous(expand = c(0, 0)) + -->
<!--   scale_fill_material_d("rainbow") + -->
<!--   coord_flip() + -->
<!--   theme_minimal() + -->
<!--   guides(fill = guide_legend(reverse = TRUE)) + -->
<!--   labs(x = NULL) + -->
<!--   theme( -->
<!--     axis.text.y = element_text( -->
<!--       color = rev(colors[closest]), -->
<!--       face = rev(ifelse(idx_order %in% c("SD", "Length", "Noise", "Random"), "italic", "plain")), -->
<!--       hjust = 0.5 -->
<!--     ), -->
<!--     axis.text.x = element_blank(), -->
<!--     plot.title = element_text(hjust = 0.5), -->
<!--     panel.grid.major = element_blank(), -->
<!--     panel.grid.minor = element_blank() -->
<!--   ) -->
<!-- p2 <- order |> -->
<!--   mutate(Duration = 1 + Duration * 10000) |> -->
<!--   filter(Index %in% loadings$Variable) |> -->
<!--   mutate(Index = fct_relevel(Index, levels(loadings$Variable)), -->
<!--          Duration = ifelse(is.na(Duration), 0, Duration)) |> -->
<!--   ggplot(aes(x = log10(Duration), y = Index)) + -->
<!--   geom_bar(aes(fill = log10(Duration)), stat = "identity") + -->
<!--   geom_hline(yintercept = head(cumsum(separations), -1) + 0.5) + -->
<!--   scale_x_reverse(expand = c(0, 0)) + -->
<!--   # scale_x_log10(breaks = 10**seq(0, 4), labels = function(x) sprintf("%g", x), expand=c(0, 0)) + -->
<!--   scale_y_discrete(position = "right") + -->
<!--   scale_fill_viridis_c(guide = "none") + -->
<!--   labs(x = "Computation Time", y = NULL) + -->
<!--   theme_minimal() + -->
<!--   theme( -->
<!--     axis.text.y = element_blank(), -->
<!--     axis.text.x = element_blank(), -->
<!--     plot.title = element_text(hjust = 0.5), -->
<!--     panel.grid.major = element_blank(), -->
<!--     panel.grid.minor = element_blank() -->
<!--   ) -->
<!-- (p2 | p1) + patchwork::plot_annotation(title = "Computation Time and Factor Loading", theme = theme(plot.title = element_text(hjust = 0.5, face = "bold"))) -->
<!-- ``` -->
<!-- ### Visualization -->
<!-- ```{r message=FALSE, warning=FALSE, fig.width=14, fig.height=7} -->
<!-- model <- # lm(Result ~ Index * (Length + Noise_Type + poly(Noise_Intensity, 2) * Signal), -->
<!--          mgcv::gam(Result ~ s(Noise_Intensity, by = interaction(Index, Signal)) + Index * (Length + Noise_Type), -->
<!--             data=df |> -->
<!--               filter(Index %in% c( -->
<!--                 "NLDFD", -->
<!--                 "SVDEn", -->
<!--                 # "SampEn", -->
<!--                 "PEn", -->
<!--                 # "FuzzyEn", -->
<!--                 # "WPEn", -->
<!--                 "K2En", -->
<!--                 # "PSDFD (Hasselman 2013)", -->
<!--                 "BubbEn", -->
<!--                 "MSPEn", -->
<!--                 "MFDFA_Increment" -->
<!--                 # "MFDFA_Width", -->
<!--                 # "MFDFA_Delta" -->
<!--                 )) |> -->
<!--               mutate(Noise_Type = as.factor(Noise_Type))) -->
<!-- # estimate_means(model, at = c("Index", "Signal", "Noise_Intensity", "Noise_Type")) |> -->
<!-- #   ggplot(aes(y = Mean, x = Noise_Intensity)) + -->
<!-- #   geom_line(aes(color = Index, size=Noise_Type)) + -->
<!-- #   facet_grid(Noise_Type~Signal) + -->
<!-- #   scale_linetype_manual(values = c("-2" = 3, "-1" = 4, "0" = 2, "1" = 5, "2" = 1)) + -->
<!-- #   scale_size_manual(values = c("-2" = 0.2, "-1" = 0.4, "0" = 0.6, "1" = 0.8, "2" = 1)) + -->
<!-- #   theme_classic() + -->
<!-- #   theme(panel.grid.major = element_line(colour = "#EEEEEE")) -->
<!-- estimate_means(model, at = c("Index", "Signal", "Noise_Intensity")) |> -->
<!--   ggplot(aes(y = Mean, x = Noise_Intensity)) + -->
<!--   geom_line(aes(color = Index)) + -->
<!--   facet_grid(~Signal) + -->
<!--   scale_linetype_manual(values = c("-2" = 3, "-1" = 4, "0" = 2, "1" = 5, "2" = 1)) + -->
<!--   theme_classic() + -->
<!--   theme(panel.grid.major = element_line(colour = "#EEEEEE")) -->
<!-- ``` -->

## Discussion

## References

<div id="refs" class="references csl-bib-body hanging-indent"
line-spacing="2">

<div id="ref-bassingthwaighte2013fractal" class="csl-entry">

Bassingthwaighte, J. B., Liebovitch, L. S., & West, B. J. (2013).
*Fractal physiology*. Springer.

</div>

<div id="ref-Makowski2021neurokit" class="csl-entry">

Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F.,
Pham, H., … Chen, S. H. A. (2021). NeuroKit2: A python toolbox for
neurophysiological signal processing. *Behavior Research Methods*,
*53*(4), 1689–1696. <https://doi.org/10.3758/s13428-020-01516-y>

</div>

</div>
