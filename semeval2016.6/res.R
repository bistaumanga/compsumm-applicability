library(ggplot2)
library(tidyr)
library(dplyr)
library(stringr)

script.dir <- dirname(sys.frame(1)$ofile)

res <- read.csv(paste0(script.dir, "/res_f1.csv"))
res <- res %>% mutate(f1 =  (f1.Against + f1.For ) / 2.0, 
                      method = factor(method), target = factor(target) )

gg <- ggplot(res, aes(x = k, y = f1, color = method)) +
  geom_line() + facet_grid(~ target) +
  theme_bw() +
  theme(legend.position = "bottom")
print(gg)
