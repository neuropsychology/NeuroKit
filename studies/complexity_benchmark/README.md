
# Benchmarking of Complexity Measures

*This study can be referenced by* [*citing the
package*](https://github.com/neuropsychology/NeuroKit#citation)
(Makowski et al., 2021).

**We’d like to improve this study, but unfortunately we currently don’t
have the time. If you want to help to make it happen, please contact
us!**

## Introduction

The goal for NeuroKit is to provide the most comprehensive, accurate and
fastest base Python implementations of complexity indices (fractal
dimension, entropy, etc.).

## Make data

``` python
import neurokit2 as nk
import pandas as pd
import numpy as np
from timeit import default_timer as timer


# Utility function
def time_function(i, x, fun=nk.fractal_petrosian, index="FD_Petrosian", method="nk_fractal_petrosian"):
  t0 = timer()
  rez, _ = fun(x)
  t1 = timer() - t0
  dat = {
    "Duration" : [t1],
    "Result" : [rez],
    "Length" : [len(x)],
    "Index" : [index],
    "Method" : [method],
    "Iteration" : [i],
  }
  return pd.DataFrame.from_dict(dat)

# Iterations
data = []
for n in np.power(10, range(2, 6)):
  print(n)
  x = nk.signal_simulate(duration=1, sampling_rate=n, frequency=[5, 10], noise=0.5)
  for i in range(100):
    data.append(time_function(i, x, nk.complexity_rr, index="RR", method="nk_complexity_rr"))
    data.append(time_function(i, x, nk.complexity_hjorth, index="Hjorth", method="nk_complexity_hjorth"))
    data.append(time_function(i, x, nk.fisher_information, index="Fisher", method="nk_fisher_information"))
    data.append(time_function(i, x, nk.entropy_shannon, index="ShanEn", method="nk_entropy_shannon"))
    data.append(time_function(i, x, nk.entropy_cumulative_residual, index="CREn", method="nk_entropy_cumulative_residual"))
    data.append(time_function(i, x, nk.entropy_differential, index="DiffEn", method="nk_entropy_differential"))
    data.append(time_function(i, x, nk.entropy_svd, index="SVDen", method="nk_entropy_svd"))
    data.append(time_function(i, x, nk.entropy_spectral, index="SpEn", method="nk_entropy_spectral"))
    data.append(time_function(i, x, nk.fractal_katz, index="Katz", method="nk_fractal_katz"))
    data.append(time_function(i, x, nk.fractal_sevcik, index="Sevcik", method="nk_fractal_sevcik"))
    data.append(time_function(i, x, nk.fractal_petrosian, index="FD_Petrosian", method="nk_fractal_petrosian"))

pd.concat(data).to_csv("data.csv", index=False)
```

## Benchmark

``` r
library(tidyverse)
library(easystats)

df <- read.csv("data.csv") |>
  mutate(Length = as.factor(Length))

order <- arrange(summarize(group_by(df, Method), Duration = mean(Duration)), Duration)
order 
## # A tibble: 11 x 2
##    Method                          Duration
##    <chr>                              <dbl>
##  1 nk_fractal_petrosian           0.0000722
##  2 nk_fractal_katz                0.000201 
##  3 nk_complexity_hjorth           0.000288 
##  4 nk_entropy_svd                 0.000352 
##  5 nk_fisher_information          0.000363 
##  6 nk_fractal_sevcik              0.000384 
##  7 nk_entropy_differential        0.00314  
##  8 nk_entropy_spectral            0.00342  
##  9 nk_entropy_shannon             0.00575  
## 10 nk_entropy_cumulative_residual 0.0484   
## 11 nk_complexity_rr               0.466

df <- mutate(df, Method = fct_relevel(Method, order$Method))

dfsummary <- df |>
  group_by(Method, Length) |>
  summarize(Duration = median(Duration))

n <- length(unique(df$Method))

df |> 
  ggplot(aes(x = Length, y = Duration)) +
  geom_line(data=dfsummary, aes(group = 1)) +
  geom_violin(aes(fill = Length)) +
  facet_wrap(~Method) +
  scale_y_log10() +
  scale_fill_viridis_d(guide = "none") +
  theme_modern() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
```

![](../../studies/complexity_benchmark/figures/unnamed-chunk-3-1.png)<!-- -->

## References

<div id="refs" class="references csl-bib-body hanging-indent"
line-spacing="2">

<div id="ref-Makowski2021neurokit" class="csl-entry">

Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F.,
Pham, H., … Chen, S. H. A. (2021). NeuroKit2: A python toolbox for
neurophysiological signal processing. *Behavior Research Methods*,
*53*(4), 1689–1696. <https://doi.org/10.3758/s13428-020-01516-y>

</div>

</div>
