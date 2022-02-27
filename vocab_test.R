library(ggplot2)
library(tidyr)
library(dplyr)
library(ggpubr)
library(ggrepel)
library(scales)
# library(ggalt)
library(assertr)
library(stringr)

script.dir <- dirname(sys.frame(1)$ofile)
save_as <- "pdf"
loss <- "squared_hinge"
labels = c("ER", "R", "RC", "C", "LC", "L")
topics <- c("cbp01", "cbp02", "chr01", "guncontrol", "climatechange")
find_label <- function(str, order){
  comps <- sort(factor( strsplit(str, split='_', fixed=TRUE)[[1]], 
                        levels = labels))
  return(comps[order])
}

vocab.ideology <- read.csv(paste0(script.dir, "/media_bias_ideology/vocab_diff.csv")) %>%
  group_by(comp, topic, month) %>% ## some weird bug
  mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
  mutate(topic = factor(topic),
         month = factor(month), 
         units = factor(units)) %>%
  mutate(hop = factor(as.numeric(comp2) - as.numeric(comp1), levels = c(1,2,3,4,5) )) %>%
  mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
  rename(Topic = topic, Hop = hop, Month = month) %>%
  mutate(all = (Month == "all" )) %>%
  mutate(p.value = pchisq(llr, df = vocab -1, lower.tail = F))

cbc_list <- list()
svm_list <- list()
i <- 1
j <- 1
ub_loss <- "squared_hinge" # or "log"

for (topic in topics){
  svm_bow <- read.csv(sprintf("%s/media_bias_ideology/res/%s_ideology_%s_bow.csv", script.dir, ub_loss, topic)) %>%
    mutate(feats = "BoW")
  svm_emb <- read.csv(sprintf("%s/media_bias_ideology/res/%s_ideology_%s_emb3.csv", script.dir, ub_loss, topic)) %>%
    mutate(feats = "SBERT")
  
  svm_list[[i]] <- svm_bow
  svm_list[[i+1]] <- svm_emb
  
  cbc_bow <- read.csv(sprintf("%s/media_bias_ideology/res/cbc_ideology_%s.csv", script.dir, topic)) %>%
    mutate(feats = case_when(method =="tok_bow_exp_greedy-diff" ~ "BoW", 
                             method == "emb_exp_greedy-diff" ~ "SBERT", TRUE ~ "NA") )
  
  cbc_list[[j]] <- cbc_bow
  # cbc_list[[j+1]] <- cbc_emb
  j <- j + 1
  i <- i + 2
}

cbc <- Reduce(rbind, cbc_list)
svm <- Reduce(rbind, svm_list)
rm(cbc_list, svm_list, cbc_bow, svm_bow, svm_emb)

cbc.ideology <- cbc %>% select(-c(val, summ1, summ2, hyps)) %>% 
  group_by(feats, comp, topic, k, month) %>% ## some weird bug
  summarize(protos = mean(test), n = n()) %>%
  mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
  mutate(topic = factor(topic),
         k = factor(k),
         feats = factor(feats),
         month = factor(month)) %>%
  mutate(hop = factor(as.numeric(comp2) - as.numeric(comp1), levels = c(1,2,3,4,5) )) %>%
  mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
  rename(Features = feats, Topic = topic, Hop = hop, Month = month)

svm.ideology <-  svm %>% select(-c(time, val)) %>% 
  group_by(feats, comp, topic, month) %>% ## required due to some weird bug
  summarize(svm = mean(test), n = n() ) %>%
  mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
  mutate(topic = factor(topic),
         feats = factor(feats),
         month = factor(month)) %>%
  mutate(hop = factor(as.numeric(comp2) - as.numeric(comp1), levels = c(1,2,3,4,5) )) %>%
  mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
  rename(Features = feats, Topic = topic, Hop = hop, Month = month)

######## zoom out by aggregating over 10 random splits
summ.ideology <- inner_join(cbc.ideology, svm.ideology,
                         by = c("comp1", "comp2", "comp", "n",
                                "Features", "Topic", "Hop", "Month")) %>%
  mutate(diff = svm - protos)

#################### temporal

labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep")
find_label <- function(str, order){
  comps <- sort(factor( strsplit(str, split='_', fixed=TRUE)[[1]], 
                        levels = labels))
  return(comps[order])
}

vocab.month <- read.csv(paste0(script.dir, "/media_bias_temporal/vocab_diff.csv")) %>%
  group_by(comp, topic, ideology) %>% ## some weird bug
  mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
  mutate(topic = factor(topic), 
         ideology = factor(ideology), 
         units = factor(units)) %>%
  mutate(hop = factor(as.numeric(comp2) - as.numeric(comp1), levels = c(1,2,3,4,5,6,7,8,9) )) %>%
  mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
  rename(Topic = topic, Hop = hop, Ideology = ideology) %>%
  mutate(all = factor(Ideology == "all" )) %>%
  mutate(p.value = pchisq(llr, df = vocab -1, lower.tail = F))


cbc_list <- list()
svm_list <- list()
i <- 1
j <- 1
for (topic in topics){
  svm_bow <- read.csv(sprintf("%s/media_bias_temporal/res/%s_temporal_%s_bow.csv", script.dir, loss, topic)) %>%
    mutate(feats = "BoW")
  svm_emb <- read.csv(sprintf("%s/media_bias_temporal/res/%s_temporal_%s_emb3.csv", script.dir, loss, topic)) %>%
    mutate(feats = "SBERT")
  
  svm_list[[i]] <- svm_bow
  svm_list[[i+1]] <- svm_emb
  i <- i + 2
  
  cbc_bow <- read.csv(sprintf("%s/media_bias_temporal/res/cbc_temporal_%s.csv", script.dir, topic)) %>%
    mutate(feats = case_when(method =="tok_bow_exp_greedy-diff" ~ "BoW", method == "emb_exp_greedy-diff" ~ "SBERT", TRUE ~ "NA") )
  # cbc_emb <- read.csv(sprintf("%s/res/cbc4_temporal_%s.csv", script.dir, topic)) %>%
  # mutate(feats = "emb")
  
  cbc_list[[j]] <- cbc_bow
  # cbc_list[[i+1]] <- cbc_emb
  j <- j + 1
}

# stop("forced error")
cbc <- Reduce(rbind, cbc_list)
svm <- Reduce(rbind, svm_list)
rm(cbc_list, svm_list, cbc_bow, svm_bow, svm_emb)

cbc.month <- cbc %>%
  select(-c(val, summ1, summ2, hyps)) %>%
  # filter(topic != "AusPol", feats != "sbert") %>%
  group_by(feats, comp, topic, k, ideology) %>% ## some weird bug
  summarize(protos = mean(test), n = n()) %>%
  mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
  mutate(topic = factor(topic),
         k = factor(k),
         feats = factor(feats)) %>%
  mutate(Hop = factor(as.numeric(comp2) - as.numeric(comp1), levels = c(1,2,3,4,5,6,7,8,9) )) %>%
  mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
  rename(Ideology = ideology, Topic = topic, Features = feats)

svm.month <-  svm %>% select(-c(time, val)) %>%  
  # filter(topic != "AusPol", feats != "sbert") %>%
  group_by(feats, comp, topic, ideology) %>% ## required due to some weird bug
  summarize(svm = mean(test), n = n() ) %>%
  mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
  mutate(topic = factor(topic),
         feats = factor(feats)) %>%
  mutate(Hop = factor(as.numeric(comp2) - as.numeric(comp1),
                      levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9) )) %>%
  mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
  rename(Ideology = ideology, Topic = topic, Features = feats)

# ######## zoom out by aggregating over 10 random splits
summ.month <- inner_join(cbc.month, svm.month,
                         by = c("comp1", "comp2", "comp", "n",
                                "Features", "Topic", "Hop", "Ideology")) %>%
  mutate(diff = svm - protos)

rm(cbc, svm, i, j, topic)

####################################################
## ideology
vocab.svm.ideology  <- full_join(vocab.ideology, svm.ideology, 
                        by = c("Topic", "comp1", "comp2", 
                               "comp", "Month", "Hop")) %>%
                      mutate(dist = 1-cos)

gg.vocab <- ggplot(vocab.svm.ideology %>% filter(all == F, Features == "BoW"), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  geom_smooth(method = "lm", se = F, size = 0.5)+
  stat_cor(method = "pearson")+
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/ideology/jsd_bacc.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 9, height = 6, units = "in", dpi = 300)

gg.vocab <- ggplot(vocab.svm.ideology %>% 
                     filter(all == F, Features == "BoW") %>%
                     group_by(Topic, comp, units) %>%
                     mutate(dist = mean(dist), svm = mean(svm)), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  geom_smooth(method = "lm", se = F, size = 0.5)+
  stat_cor(method = "pearson")+
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/ideology/jsd_bacc_agg_months.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 9, height = 6, units = "in", dpi = 300)

gg.vocab <- ggplot(vocab.svm.ideology %>% filter(all == F, Features == "BoW"), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  facet_wrap(~Topic, ncol = 3)+
  geom_smooth(method = "lm", se = F, size = 0.5)+
  stat_cor(method = "pearson")+
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/ideology/jsd_bacc_topic.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 18, height = 8, units = "in", dpi = 300)

gg.vocab <- ggplot(vocab.svm.ideology %>% filter(all == F, Features == "BoW"), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  geom_smooth(method = "lm", se = F, size = 0.5) +
  facet_grid(comp1~comp2) +
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/ideology/jsd_bacc_comps.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 15, height = 12, units = "in", dpi = 300)

############################################################################
## month
vocab.svm.month  <- full_join(vocab.month, svm.month, 
                                 by = c("Topic", "comp1", "comp2", 
                                        "comp", "Ideology", "Hop")) %>%
                    mutate(dist = 1-cos)

gg.vocab <- ggplot(vocab.svm.month %>% filter(all == F, Features == "BoW"), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  geom_smooth(method = "lm", se = F, size = 0.5)+
  stat_cor(method = "pearson")+
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/temporal/jsd_bacc.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 9, height = 6, units = "in", dpi = 300)

gg.vocab <- ggplot(vocab.svm.month %>% 
                     filter(all == F, Features == "BoW") %>%
                     group_by(Topic, comp, units) %>%
                     mutate(dist = mean(dist), svm = mean(svm)), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  geom_smooth(method = "lm", se = F, size = 0.5)+
  stat_cor(method = "pearson")+
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/temporal/jsd_bacc_agg_ideology.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 9, height = 6, units = "in", dpi = 300)

gg.vocab <- ggplot(vocab.svm.month %>% 
                     filter(all == F, Features == "BoW") %>%
                     group_by(Ideology, comp, units) %>%
                     mutate(dist = mean(dist), svm = mean(svm)), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  geom_smooth(method = "lm", se = F, size = 0.5)+
  stat_cor(method = "pearson")+
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/temporal/jsd_bacc_agg_topic.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 9, height = 6, units = "in", dpi = 300)

gg.vocab <- ggplot(vocab.svm.month %>% filter(all == F, Features == "BoW"), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  facet_wrap(~Topic, ncol = 3)+
  stat_cor(method = "pearson")+
  geom_smooth(method = "lm", se = F, size = 0.5)+
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/temporal/jsd_bacc_topic.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 18, height = 8, units = "in", dpi = 300)

gg.vocab <- ggplot(vocab.svm.month %>% filter(all == F, Features == "BoW"), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  facet_wrap(~Ideology, ncol = 3)+
  geom_smooth(method = "lm", se = F, size = 0.5)+
  stat_cor(method = "pearson")+
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/temporal/jsd_bacc_ideology.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 18, height = 8, units = "in", dpi = 300)

gg.vocab <- ggplot(vocab.svm.month %>% filter(all == F, Features == "BoW"), 
                   aes (x = svm, y = dist, color = units)) +
  geom_point() +
  geom_smooth(method = "lm", se = F, size = 0.5) +
  facet_grid(comp1~comp2) +
  labs(x = "Distinguishability", y = "Vocab Dist.")+
  theme_bw()

ggsave(sprintf("%s/plots/temporal/jsd_bacc_comps.%s", script.dir, save_as),
       plot = gg.vocab,
       width = 15, height = 12, units = "in", dpi = 300)

########################################################################

summ.ideology.month <- vocab.svm.ideology %>% 
                          filter(all == F) %>%
                          mutate(comparing = "Ideology") %>%
                          ungroup() %>%
                          group_by(Topic, comp, units, comp1, comp2, Hop, Features, comparing) %>%
                          summarize(dist = mean(dist), svm = mean(svm))

summ.month.ideology <- vocab.svm.month %>% 
  filter(all == F) %>%
  mutate(comparing = "Month") %>%
  group_by(Topic, comp, units, comp1, comp2, Hop, Features, comparing) %>%
  summarize(dist = mean(dist), svm = mean(svm))

summ.vocab.combined <- rbind(summ.ideology.month, summ.month.ideology) %>%
  mutate(comparing = factor(comparing))


plot1.2d <- ggplot(summ.vocab.combined, 
                   aes(x = svm, y = dist) ) +
  geom_point(aes(color = comparing, shape = Topic), size = 1) +
  guides(color = guide_legend(direction = "vertical"),
         shape = guide_legend(direction = "vertical", nrow = 2)) +
  # geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
  facet_grid(Features~units) +
  # geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
  geom_smooth(aes(color = comparing), 
              linetype = 2, method = "lm", se = F, size = 0.5) +
  stat_cor(aes(color = comparing, label = paste(..r.label..,
                                                cut(..p..,breaks = c(0.0, 0.0001, 0.001, 0.01, 0.05, 1.0),
                                                    labels = c("'****'", "'***'", "'**'", "'*'", "'-'")),
                                                sep = "~")), method = "pearson")+
  theme_bw() + 
  # geom_text_repel(data = summ.vocab.combined %>% filter(comparing == "Ideology"),
  #                 aes(label = sprintf("%s", comp), color = Hop), size = 2.5, 
  #                 max.overlaps = 30) +
  # scale_fill_manual(Hop_colors) +
  labs(x = "Distinguishability", y = "Vocab Dist.") +
  theme(legend.position = "bottom")

ggsave(sprintf("%s/plots/dist_svm_zoomed_topics.%s", script.dir, save_as),
       plot = plot1.2d,
       width = 8.0, height = 7.2, units = "in", dpi = 300)

temp <- summ.vocab.combined %>% ungroup %>%
  group_by(comp, comp1, comp2, Hop, comparing, Features, units) %>%
  summarize(dist = mean(dist), svm = mean(svm))
plot1.2d <- ggplot(temp, 
                   aes(x = svm, y = dist) ) +
  geom_point(aes(color = comparing), size = 1) +
  # geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
  facet_grid(Features~units) +
  # guides(color = guide_legend(nrow = 2, direction = "vertical"),
  #        shape = guide_legend(direction = "vertical")) +
  # geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
  geom_smooth(aes(color = comparing), linetype = 2, 
              method = "lm", se = F, size = 0.5) +
  stat_cor(aes(color = comparing, label = paste(..r.label..,
                                                cut(..p..,breaks = c(0.0, 0.0001, 0.001, 0.01, 0.05, 1.0),
                                                    labels = c("'****'", "'***'", "'**'", "'*'", "'-'")),
                                                sep = "~")), method = "pearson")+
  theme_bw() + 
  geom_text_repel(data = temp %>% filter(comparing == "Ideology"),
                  aes(label = sprintf("%s", comp), color = comparing), size = 3.0,
                  max.overlaps = 30) +
  # scale_fill_manual(Hop_colors) +
  labs(x = "Distinguishability", y = "Vocab Dist.") +
  theme(legend.position = "bottom")

ggsave(sprintf("%s/plots/dist_svm_aggregated.%s", script.dir, save_as),
       plot = plot1.2d,
       width = 8.0, height = 6.5, units = "in", dpi = 300)



# 
# vi.summ <- vi %>% group_by(comp, comp1, comp2, all, units) %>%
#   summarize(jsd = mean(jsd), llr = mean(llr), cosd = 1 - mean(cos))
# 
# gg.vi <- ggplot(vi.summ %>% filter(all == F), aes(x = comp1, y = comp2)) +
#   geom_text(aes(label = sprintf("%.2g\n%.2g", cosd, jsd))) +
#   facet_grid(.~units) +
#   theme_bw()
# 
# ggsave(sprintf("%s/plots/vocab_ideology_comp.%s", script.dir, save_as),
#        plot = gg.vi,
#        width = 9.6, height = 5.0, units = "in", dpi = 300)
# 
# vt.summ <- vt %>% group_by(comp, comp1, comp2, all, units) %>%
#   summarize(jsd = mean(jsd), llr = mean(llr), cosd = 1 - mean(cos))
# 
# gg.vt <- ggplot(vt.summ %>% filter(all == F), aes(x = comp1, y = comp2)) +
#   geom_text(aes(label = sprintf("%.2g\n%.2g", cosd, jsd))) +
#   facet_grid(.~units) +
#   theme_bw()
# 
# ggsave(sprintf("%s/plots/vocab_temporal_comp.%s", script.dir, save_as),
#        plot = gg.vt,
#        width = 9.6, height = 6.0, units = "in", dpi = 300)
# 
