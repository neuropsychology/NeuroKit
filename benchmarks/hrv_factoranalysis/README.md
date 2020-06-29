
# The Factor Structure of Heart Rate Variability (HRV) Indices

*This study can be referenced by* [*citing the
package*](https://github.com/neuropsychology/NeuroKit#citation).

**We’d like to publish this study, but unfortunately we currently don’t
have the time. If you want to help to make it happen, please contact
us\!**

## Introduction

The aim of this study is to explore the factor structure of HRV indices.

## Databases

We used the same databases as in [this
study](https://github.com/neuropsychology/NeuroKit/tree/master/benchmarks/ecg_preprocessing#databases).

## Procedure

``` python
import pandas as pd
import numpy as np
import neurokit2 as nk

# Load True R-peaks location
rpeaks_gudb = pd.read_csv("../../data/gudb/Rpeaks.csv")
rpeaks_mit1 = pd.read_csv("../../data/mit_arrhythmia/Rpeaks.csv")
rpeaks_mit2 = pd.read_csv("../../data/mit_normal/Rpeaks.csv")

datafiles = [rpeaks_gudb, rpeaks_mit1, rpeaks_mit2]

# Get results
all_results = pd.DataFrame()

for file in datafiles:
    for database in np.unique(file["Database"]):
        data = file[file["Database"] == database]
        for participant in np.unique(data["Participant"]):
            data_participant = data[data["Participant"] == participant]
            sampling_rate = np.unique(data_participant["Sampling_Rate"])[0]
            rpeaks = data_participant["Rpeaks"].values

            results = nk.hrv(rpeaks, sampling_rate=sampling_rate)
            results["Participant"] = participant
            results["Database"] = database
            results["Recording_Length"] = rpeaks[-1] / sampling_rate / 60

            all_results = pd.concat([all_results, results], axis=0)

all_results.to_csv("data.csv", index=False)
```

## Results

``` r
library(tidyverse)
library(easystats)

data <- read.csv("data.csv", stringsAsFactors = FALSE) %>% 
  select(-HRV_S, -HRV_SD1)  # Redundant
names(data) <- stringr::str_remove(names(data), "HRV_")
```

### Recording Length

#### Investigate effect

``` r
correlation(data) %>% 
  filter(Parameter2 == "Recording_Length") %>% 
  arrange(desc(abs(r)))
```

#### Adjust the data for recording length

``` r
data <- effectsize::adjust(data, effect="Recording_Length") %>% 
  select(-Recording_Length)
```

### Gaussian Graphical Model

``` r
library(ggraph)

data %>% 
  select(-ULF) %>%   # Empty 
  correlation::correlation(partial=FALSE) %>% 
  filter(abs(r) > 0.2) %>% 
  tidygraph::as_tbl_graph(directed=FALSE) %>% 
  dplyr::mutate(closeness = tidygraph::centrality_closeness(normalized = TRUE),
                degree = tidygraph::centrality_degree(normalized = TRUE),
                betweeness = tidygraph::centrality_betweenness(normalized = TRUE)) %>%
  tidygraph::activate(nodes) %>%
  dplyr::mutate(group1 = as.factor(tidygraph::group_edge_betweenness()),
                group2 = as.factor(tidygraph::group_optimal()),
                group3 = as.factor(tidygraph::group_walktrap()),
                group4 = as.factor(tidygraph::group_spinglass()),
                group5 = as.factor(tidygraph::group_louvain())) %>% 
  ggraph::ggraph(layout = "lgl") +
    ggraph::geom_edge_arc(aes(colour = r, edge_width = abs(r)), strength = 0.1, show.legend = FALSE) +
    ggraph::geom_node_point(aes(size = degree, color = group2), show.legend = FALSE) +
    ggraph::geom_node_text(aes(label = name), colour = "white") +
    ggraph::scale_edge_color_gradient2(low = "#a20025", high = "#008a00", name = "r") +
    ggraph::theme_graph() +
    guides(edge_width = FALSE) +
    scale_x_continuous(expand = expand_scale(c(.10, .10))) +
    scale_y_continuous(expand = expand_scale(c(.10, .10))) +
    scale_size_continuous(range = c(20, 30)) +
    scale_edge_width_continuous(range = c(0.5, 2)) +
    see::scale_color_material_d(palette="rainbow", reverse=TRUE)
```

![](figures/unnamed-chunk-6-1.png)<!-- -->

Groups were identified using the
[**tidygraph::group\_optimal**](https://rdrr.io/cran/tidygraph/man/group_graph.html)
algorithm.

### Factor Analysis

#### How many factors

``` r
n <- parameters::n_factors(data[sapply(data, is.numeric)])
## 
##  These indices are only valid with a principal component solution.
##  ...................... So, only positive eugenvalues are permitted.

plot(n) +
  theme_modern()
```

![](figures/unnamed-chunk-7-1.png)<!-- -->

#### Interpret

``` r
fa <- parameters::factor_analysis(data[sapply(data, is.numeric)], n=7, rotation="varimax")

print(fa, threshold="max", sort=TRUE)
## # Rotated loadings from Factor Analysis (varimax-rotation)
## 
## Variable     |  MR1 |  MR3 |  MR4 |   MR2 |  MR6 |  MR5 |   MR7 | Complexity | Uniqueness
## -----------------------------------------------------------------------------------------
## TINN         | 0.99 |      |      |       |      |      |       |       1.01 |       0.01
## VLF          | 0.99 |      |      |       |      |      |       |       1.16 |      -0.05
## LF           | 0.98 |      |      |       |      |      |       |       1.02 |       0.03
## RMSSD        | 0.95 |      |      |       |      |      |       |       1.19 |   9.25e-03
## SDSD         | 0.95 |      |      |       |      |      |       |       1.19 |   9.24e-03
## SDNN         | 0.95 |      |      |       |      |      |       |       1.19 |       0.02
## CVNN         | 0.94 |      |      |       |      |      |       |       1.24 |   6.56e-03
## CVSD         | 0.94 |      |      |       |      |      |       |       1.27 |  -5.60e-04
## SD2          | 0.93 |      |      |       |      |      |       |       1.24 |       0.03
## LFHF         | 0.82 |      |      |       |      |      |       |       1.61 |       0.13
## PIP          |      | 0.99 |      |       |      |      |       |       1.05 |   1.47e-04
## IALS         |      | 0.98 |      |       |      |      |       |       1.07 |  -2.97e-03
## PSS          |      | 0.90 |      |       |      |      |       |       1.01 |       0.19
## PAS          |      | 0.76 |      |       |      |      |       |       1.53 |       0.27
## MCVNN        |      |      | 0.96 |       |      |      |       |       1.07 |       0.05
## MadNN        |      |      | 0.94 |       |      |      |       |       1.08 |       0.08
## IQRNN        |      |      | 0.79 |       |      |      |       |       1.58 |       0.20
## pNN50        |      |      | 0.66 |       |      |      |       |       2.86 |       0.17
## SampEn       |      |      |      |  0.81 |      |      |       |       1.58 |       0.16
## ApEn         |      |      |      |  0.78 |      |      |       |       1.69 |       0.20
## CSI_Modified |      |      |      | -0.73 |      |      |       |       2.40 |       0.14
## HTI          |      |      |      |  0.58 |      |      |       |       1.51 |       0.59
## CSI          |      |      |      | -0.57 |      |      |       |       3.39 |       0.28
## LnHF         |      |      |      |       | 0.67 |      |       |       3.43 |       0.09
## VHF          |      |      |      |       | 0.64 |      |       |       2.39 |       0.31
## CVI          |      |      |      |       | 0.62 |      |       |       3.69 |       0.07
## SD1SD2       |      |      |      |       | 0.58 |      |       |       2.86 |       0.29
## HF           |      |      |      |       | 0.55 |      |       |       2.27 |       0.36
## MeanNN       |      |      |      |       |      | 0.93 |       |       1.29 |       0.02
## MedianNN     |      |      |      |       |      | 0.87 |       |       1.44 |       0.09
## pNN20        |      |      |      |       |      | 0.53 |       |       3.99 |       0.13
## LFn          |      |      |      |       |      |      | -0.92 |       1.28 |       0.04
## HFn          |      |      |      |       |      |      |  0.76 |       1.66 |       0.24
## 
## The 7 latent factors (varimax rotation) accounted for 87.35% of the total variance of the original data (MR1 = 29.97%, MR3 = 11.63%, MR4 = 11.62%, MR2 = 10.31%, MR6 = 8.76%, MR5 = 8.28%, MR7 = 6.78%).
```

<!-- #### Visualize -->

<!-- ```{r, message=FALSE, warning=FALSE, fig.width=17, fig.height=17} -->

<!-- plot(fa) -->

<!-- ``` -->

# References
