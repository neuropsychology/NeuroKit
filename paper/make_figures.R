library(tidyverse)
library(easystats)
library(formattable)

# Setup python - you need to change the path to your python distribution
library(reticulate)
# reticulate::use_python("C:/Users/Zen Juen/Downloads/WPy-3710b/python-3.7.1.amd64/")
reticulate::use_python("D:/Downloads/WPy64-3810/python-3.8.1.amd64/")
matplotlib <- import("matplotlib")
# matplotlib$use("Agg", force = TRUE)




# Python ------------------------------------------------------------------
reticulate::repl_python()

import neurokit2 as nk
import pandas as pd
import numpy as np

# Data
data = nk.data("bio_resting_8min_100hz").drop(['PhotoSensor', 'Unnamed: 0'], axis=1)

Photosensor = list(np.full(len(data)-4, 0))
Photosensor.insert(100, 5)
Photosensor.insert(500, 5)
Photosensor.insert(900, 5)
Photosensor.insert(1300, 5)
Photosensor = pd.DataFrame(Photosensor).rename({0: 'Photosensor'}, axis=1)
data = pd.concat([data, Photosensor], axis=1)

exit
# -------------------------------------------------------------------------

library(ggpubr)
theme_set(theme_pubr())

df_p <- py$data %>%
  slice(0:2000) %>%
  standardize() %>%
  mutate(Time = 1:n() / 100) %>%
  pivot_longer(1:3) %>%
  mutate(name = fct_relevel(name, c("ECG", "RSP", "EDA")))

plot <- df_p %>%
  ggplot(aes(x=Time, y=value, color=name, size=name)) +

  # shaded areas
  geom_rect(aes(xmin = 0, xmax = 2, ymin = -Inf, ymax = Inf, fill = "Event-related Analysis"), alpha = 0, color="#FF9800") +
  geom_rect(aes(xmin = 4, xmax = 6, ymin = -Inf, ymax = Inf, fill = "Event-related Analysis"), alpha = 0, color="#FF9800") +
  geom_rect(aes(xmin = 8, xmax = 10, ymin = -Inf, ymax = Inf, fill = "Event-related Analysis"), alpha = 0, color="#FF9800") +
  geom_rect(aes(xmin = 12, xmax = 14, ymin = -Inf, ymax = Inf, fill = "Event-related Analysis"), alpha = 0, color="#FF9800") +
  geom_rect(aes(xmin = 0, xmax = 20, ymin = min(df_p$value), ymax = max(df_p$value), fill = "Interval-related Analysis"), alpha = 0, color="darkgrey") +

  # signals
  geom_line() +

  # event markers
  geom_vline(xintercept=1, linetype="dashed", size=1) +
  geom_vline(xintercept=5, linetype="dashed", size=1) +
  geom_vline(xintercept=9, linetype="dashed", size=1) +
  geom_vline(xintercept=13, linetype="dashed", size=1) +
  annotate("text", label = "Event Markers", x = 4.8, y = 5, angle=90) +

  # aesthetics
  theme_modern() +
  scale_color_manual('Signal type',
                     values=c("ECG"="red", "EDA"="#9C27B0", "RSP"="#2196F3", "Photosensor"="#FF9800")) +
  scale_size_manual(values=c("ECG"=0.66, "EDA"=2, "RSP"=2), guide=FALSE) +
  scale_fill_manual('Analysis type',
                    values =c("Event-related Analysis"="orange",
                              "Interval-related Analysis"="darkgrey"),
                    guide=guide_legend(override.aes = list(colour=c("#FF9800", "darkgrey")))) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        legend.position = "right") +
  ylab("Time (s)") +
  ggtitle("Physiological Signals")

# Tables
# py$event_table <- py$event_table %>%
#   mutate(ECG_Rate_Mean = round(ECG_Rate_Mean, 2),
#          RSP_Rate_Mean = round(RSP_Rate_Mean, 2))
# py$interval_table <- py$interval_table %>%
#   mutate_at(c(1:5, 7), funs(round(., 2)))

# eventrelated <- ggtexttable(py$event_table, rows = NULL,
#                             theme = ttheme("mOrange"))
# intervalrelated <- ggtexttable(py$interval_table, rows = NULL)




table <- data.frame("A" = c("ECG Rate Characteristics: Mean, Amplitude",
                            "Heart Rate Variability (HRV) indices",
                            "Respiratory Rate Variability (RRV) indices",
                            "Respiratory Sinus Arrhythmia (RSA) indices",
                            "Number of SCR Peaks and mean amplitude"),

                    "B" = c("ECG Rate Changes: Min, Mean, Max, Time of Min and  Max, Trend (Linear, Quadratic, R2)",
                            "RSP Rate Changes: Min, Mean, Max, Time of Min and Max",
                            "RSP Amplitude Measures: Min, Mean, Max",
                            "ECG and RSP Phase: Type (Inspiration/Expiration, Systole/Diastole), Completion",
                            "SCR peak and its characteristics (amplitude, rise time, recovery time)"))

colnames(table) <- c("Event-related Features", "Interval-related Features")




table_features <- ggtexttable(table, rows = NULL, theme = ttheme("default"))

p <- ggarrange(plot, table_features,
               ncol=1, nrow=2,
               heights=c(1, 0.5))
p


figheight <- 6
figwidth <- 6 * 1.618034
# ggsave("figures/features.png", p, height=figwidth, width=figwidth*1.5, dpi=600)
