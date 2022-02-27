library(ggplot2)
library(tidyr)
library()
library(dplyr)
library(ggpubr)
library(ggrepel)
library(scales)
library(introdataviz)
# library(ggalt)
# library(stringr)

script.dir <- dirname(sys.frame(1)$ofile)
save_as <- "pdf"

ideology <- read.csv(sprintf("%s/media_bias_ideology/joined_ideology.csv", script.dir)) %>% 
  rename(comp.unit = Month)
temporal <- read.csv(sprintf("%s/media_bias_temporal/joined_temporal.csv", script.dir)) %>% 
  rename(comp.unit = Ideology)

data <- rbind(ideology, temporal) %>%
  mutate(k = factor(k), 
         comp1 = factor(comp1), comp2 = factor(comp2),
         comp = factor(comp), by = factor(by), Features = factor(Features),
         Hop = factor(Hop)
  ) %>%
  rename(comparing = by)

rm(ideology, temporal)

summ <- data %>% group_by(k, Features, comparing) %>%
  summarize(avg.diff = mean(protos.diff),
            sd.diff=sd(protos.diff),
            se.diff=sd(protos.diff)/sqrt(n()),
            n = n())

print(summ)

protos.diff.range <-  ggplot(summ, aes(x = k, y = avg.diff, 
                                       fill = Features, color = Features)) + 
  geom_pointrange(aes(ymin = avg.diff - sd.diff, ymax = avg.diff + sd.diff),
                  position = position_dodge2(0.25)) +
  # geom_point(position = position_dodge2(0.9)) +
  facet_grid(comparing~.) +
  labs(x = "Number of prototypes", y = "Summarizability difference (MMD - Random)") +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  theme_bw() +
  theme(legend.position= "right" )#c(.85,.36))
  # coord_flip()

ggsave(sprintf("%s/plots/protos.diff.pointrange.%s", script.dir, save_as),
       plot = protos.diff.range,
       width = 16, height = 10, units = "cm", dpi = 300)

protos.diff.box <- ggplot(data, aes(x = k, y = protos.diff, 
                                    fill = Features, color = Features)) + 
  geom_split_violin(alpha = 0.15, size = 0.25) +
  facet_grid(~comparing) +
  geom_pointrange(data=summ, aes(y = avg.diff, 
                                 ymin = avg.diff - sd.diff, ymax = avg.diff + sd.diff,
                                 x = k, fill = Features, color = Features),
                  position = position_dodge2(0.3), size = 0.35) +
  # stat_summary(fun="mean", geom = "crossbar",
  #              # shape=18,
  #              size=0.25,
  #              width=0.6,show_guide = FALSE,
  #              position = position_dodge2(width = 0.75,preserve = "single")) +
  stat_summary(fun=mean, geom="text", show_guide = FALSE,
               vjust=0.4, hjust = 0.5, size = 3, 
               aes( label=sprintf("%.3f", round(..y.., digits=3))), 
               position = position_dodge2(width = 0.8,   
                                          preserve = "single")) +
  labs(x = "Number of prototypes", y = "Summarizability difference (MMD - Random)") +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  theme_bw() +
  theme(legend.position=c(.5,.5)) +
  coord_flip()

ggsave(sprintf("%s/plots/protos.diff.%s", script.dir, save_as),
       plot = protos.diff.box,
       width = 20, height = 10, units = "cm", dpi = 300)