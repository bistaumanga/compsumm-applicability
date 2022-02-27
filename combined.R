library(ggplot2)
library(tidyr)
library()
library(dplyr)
library(ggpubr)
library(ggrepel)
library(scales)
# library(ggalt)
# library(stringr)

script.dir <- dirname(sys.frame(1)$ofile)
save_as <- "pdf"

ideology <- read.csv(sprintf("%s/media_bias_ideology/summ_ideology.csv", script.dir))
temporal <- read.csv(sprintf("%s/media_bias_temporal/summ_temporal.csv", script.dir))

data <- rbind(ideology, temporal) %>%
  mutate(k = factor(k), 
         comp1 = factor(comp1), comp2 = factor(comp2),
         comp = factor(comp), by = factor(by), Features = factor(Features),
         Hop = factor(Hop), correlation = factor(correlation)
         ) %>%
  rename(comparing = by)
# 
# labels = c("ER", "R", "RC", "C", "LC", "L")
# find_label <- function(str, order){
#   comps <- sort(factor( strsplit(str, split='_', fixed=TRUE)[[1]], 
#                         levels = labels))
#   return(comps[order])
# }
# 
# vi <- read.csv(paste0(script.dir, "/media_bias_ideology/vocab_diff.csv")) %>%
#   group_by(comp, topic, month) %>% ## some weird bug
#   mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
#   mutate(topic = factor(topic),
#          month = factor(month), 
#          units = factor(units)) %>%
#   mutate(hop = factor(as.numeric(comp2) - as.numeric(comp1), levels = c(1,2,3,4,5) )) %>%
#   mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
#   rename(Topic = topic, Hop = hop, Month = month) %>%
#   mutate(all = (Month == "all" ), comparing = "Ideology") %>%
#   mutate(jsd1 = jsd / log(vocab))
# 
# labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep")
# find_label <- function(str, order){
#   comps <- sort(factor( strsplit(str, split='_', fixed=TRUE)[[1]], 
#                         levels = labels))
#   return(comps[order])
# }
# 
# vt <- read.csv(paste0(script.dir, "/media_bias_temporal/vocab_diff.csv")) %>%
#   group_by(comp, topic, ideology) %>% ## some weird bug
#   mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
#   mutate(topic = factor(topic), 
#          ideology = factor(ideology), 
#          units = factor(units)) %>%
#   mutate(hop = factor(as.numeric(comp2) - as.numeric(comp1), levels = c(1,2,3,4,5,6,7,8,9) )) %>%
#   mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
#   rename(Topic = topic, Hop = hop, Ideology = ideology) %>%
#   mutate(all = factor(Ideology == "all" ), comparing = "Month") %>%
#   mutate(jsd1 = jsd / log(vocab))
# 
# vi.summ <- vi %>%
#   filter(all == F) %>%
#   group_by(comp, comp1, comp2, Hop, units, comparing) %>%
#   summarize(jsd = mean(jsd), 
#             jsd1 = mean(jsd1),
#             cosd = 1 - mean(cos)) %>%
#   ungroup()
# 
# vt.summ <- vt %>%
#   filter(all == F) %>%
#   group_by(comp, comp1, comp2, Hop, units, comparing) %>%
#   summarize(jsd = mean(jsd), 
#             jsd1 = mean(jsd1),
#             cosd = 1 - mean(cos)) %>%
#   ungroup()
# 
# vocab <- rbind(vi.summ, vt.summ) %>%
#   mutate(comparing = factor(comparing))

temp <- data %>% filter(k == "8")

plot1.2d <- ggplot(temp, 
                   aes(x = bacc.svm, y = bacc.protos) ) +
  geom_point(aes(color = Hop, shape = comparing), size = 2) +
  geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
  facet_grid(.~Features) +
  guides(color = guide_legend(nrow = 2, direction = "vertical"),
         shape = guide_legend(direction = "vertical")) +
  geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
  # geom_smooth(aes(linetype = by), color = "blue", method = "lm", se = F, size = 0.5) +
  theme_bw() + 
  geom_text_repel(data = temp %>% filter(comparing == "Ideology"),
                   aes(label = sprintf("%s", comp), color = Hop), size = 2.5, 
                  max.overlaps = 30) +
  # scale_fill_manual(Hop_colors) +
  labs(x = "Distinguishability", y = "Summarizability") +
  # geom_errorbar(aes(ymin = bacc.protos - se.protos,
  #                   ymax = bacc.protos + se.protos, color = Hop)) +
  # geom_errorbarh(aes(xmin = bacc.svm - se.svm,
  #                    xmax = bacc.svm + se.svm, color = Hop)) +
  theme(legend.position = "bottom")

ggsave(sprintf("%s/plots/2dplot1.%s", script.dir, save_as),
       plot = plot1.2d,
       width = 7.2, height = 4.4, units = "in", dpi = 300)

plot2.2d <- ggplot(temp, 
                   aes(y = bacc.svm, x = bacc.protos) ) +
  geom_point(aes(color = Hop, shape = Features), size = 2) +
  guides(color = guide_legend(nrow = 2, direction = "vertical"),
         shape = guide_legend(direction = "vertical")) +
  geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
  facet_wrap(.~comparing, ncol = 2, scales = "free") +
  geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
  # geom_smooth(aes(linetype = by), color = "blue", method = "lm", se = F, size = 0.5) +
  theme_bw() + 
  geom_text_repel(data = temp %>% filter(Features == "SBERT", comparing == "Ideology"),
                  aes(label = sprintf("%s", comp), color = Hop), size = 3.0, 
                  max.overlaps = 100) +
  # scale_fill_manual(Hop_colors) +
  labs(y = "Distinguishability", x = "Summarizability") +
  # geom_errorbar(aes(ymin = bacc.protos - se.protos,
  #                   ymax = bacc.protos + se.protos, color = Hop)) +
  # geom_errorbarh(aes(xmin = bacc.svm - se.svm,
  #                    xmax = bacc.svm + se.svm, color = Hop)) +
  theme(legend.position = "bottom")

ggsave(sprintf("%s/plots/2dplot2.%s", script.dir, save_as),
       plot = plot2.2d,
       width = 7.2, height = 4.4, units = "in", dpi = 300)

# 
# temp2 <- full_join(temp, vocab, 
#                   by = c("comp", "comp1", "comp2", "Hop", "comparing"))
# 
# plot1.2d <- ggplot(temp2, 
#                    aes(x = bacc.svm, y = cosd) ) +
#   geom_point(aes(color = Hop, shape = comparing), size = 2) +
#   geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
#   facet_grid(Features~units) +
#   geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
#   # geom_smooth(aes(linetype = by), color = "blue", method = "lm", se = F, size = 0.5) +
#   theme_bw() + 
#   geom_text_repel(data = temp2 %>% filter(comparing == "Ideology"),
#                   aes(label = sprintf("%s", comp), color = Hop), size = 2.5, 
#                   max.overlaps = 30) +
#   # scale_fill_manual(Hop_colors) +
#   labs(x = "SVM Balanced Acc.", y = "Vocab dist") +
#   # geom_errorbar(aes(ymin = bacc.protos - se.protos,
#   #                   ymax = bacc.protos + se.protos, color = Hop)) +
#   # geom_errorbarh(aes(xmin = bacc.svm - se.svm,
#   #                    xmax = bacc.svm + se.svm, color = Hop)) +
#   theme(legend.position = "bottom")
# 
# ggsave(sprintf("%s/plots/2dplot_1v.%s", script.dir, save_as),
#        plot = plot1.2d,
#        width = 8.4, height = 8.0, units = "in", dpi = 300)
# 
# plot2.2d <- ggplot(temp2, 
#                    aes(x = bacc.svm, y = cosd) ) +
#   geom_point(aes(color = Hop, shape = Features), size = 2) +
#   guides(color = guide_legend(nrow = 4)) +
#   geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
#   facet_grid(units~comparing) +
#   geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
#   # geom_smooth(aes(linetype = by), color = "blue", method = "lm", se = F, size = 0.5) +
#   theme_bw() + 
#   geom_text_repel(data = temp2 %>% filter(Features == "SBERT", comparing == "Ideology"),
#                   aes(label = sprintf("%s", comp), color = Hop), size = 3.0, 
#                   max.overlaps = 100) +
#   # scale_fill_manual(Hop_colors) +
#   labs(x = "Distinguishability", y = "Vocab dist") +
#   # geom_errorbar(aes(ymin = bacc.protos - se.protos,
#   #                   ymax = bacc.protos + se.protos, color = Hop)) +
#   # geom_errorbarh(aes(xmin = bacc.svm - se.svm,
#   #                    xmax = bacc.svm + se.svm, color = Hop)) +
#   theme(legend.position = "right")
# 
# ggsave(sprintf("%s/plots/2dplot_2v.%s", script.dir, save_as),
#        plot = plot2.2d,
#        width = 8.4, height = 8.0, units = "in", dpi = 300)