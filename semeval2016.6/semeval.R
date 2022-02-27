library(ggplot2)
library(tidyr)
library(dplyr)
# library(ggpubr)
library(ggrepel)
library(scales)
# library(ggalt)
# library(stringr)

script.dir <- dirname(sys.frame(1)$ofile)
data <- read.csv(paste0(script.dir, "/upper_bound_emb.csv"))


summ <- data %>%
  group_by(refit, loss, oversample) %>%
  summarize(bacc = mean(bacc))