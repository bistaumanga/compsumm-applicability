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
labels = c("ER", "R", "RC", "C", "LC", "L")
topics <- c("cbp01", "cbp02", "chr01", "guncontrol", "climatechange")
save_as <- "pdf"
Features <- "both"
hop_colors <- c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854")

find_label <- function(str, order){
  comps <- sort(factor( strsplit(str, split='_', fixed=TRUE)[[1]], 
                        levels = labels))
  return(comps[order])
}

cbc_list <- list()
svm_list <- list()
i <- 1
j <- 1
ub_loss <- "squared_hinge" # or "log"

for (topic in topics){
  svm_bow <- read.csv(sprintf("%s/res/%s_ideology_%s_bow.csv", script.dir, ub_loss, topic)) %>%
    mutate(feats = "BoW")
  svm_emb <- read.csv(sprintf("%s/res/%s_ideology_%s_emb3.csv", script.dir, ub_loss, topic)) %>%
    mutate(feats = "SBERT")
  
  svm_list[[i]] <- svm_bow
  svm_list[[i+1]] <- svm_emb
  
  cbc_bow <- read.csv(sprintf("%s/res/cbc_ideology_%s.csv", script.dir, topic)) %>%
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

cbc <- cbc %>% select(-c(val, summ1, summ2, hyps)) %>% 
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

svm <-  svm %>% select(-c(time, val)) %>% 
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
summ.comps <- inner_join(cbc, svm,
                by = c("comp1", "comp2", "comp", "n",
                       "Features", "Topic", "Hop", "Month")) %>%
  mutate(diff = svm - protos)
print(summ.comps %>% ungroup %>% group_by(Features) %>% summarize(s = mean(svm)))
# ############## points plot
for(f in unique(summ.comps$Features)){
  gg.sc <- ggscatter(
    summ.comps %>% filter(Features == f), y = "protos", x = "svm",
    color = "k", palette = "jco",
    add = "reg.line",
    alpha = 0.25,
    add.params = list(size = 0.5),
    facet.by = c("comp1", "comp2"),
    ggtheme = theme_bw()
  ) +
    stat_mean(aes(color = k, shape = k), size = 3) +
    stat_cor(aes(color = k,
                 label = paste(..r.label..,
                               cut(..p..,breaks = c(0.0, 0.001, 0.01, 0.05, 1.0),
                                   labels = c("'***'", "'**'", "'*'", "'-'")),
                               sep = "~")), method = "pearson",
    ) +
    labs(y = "Prototypes Balanced Acc.", x = "SVM Balaced Acc.") +
    # geom_hline(aes(yintercept = mean(bacc.svm)))+
    geom_abline(slope = 1, size = 0.5, linetype = 2, color = "black")
  
  ggsave(sprintf("%s/plots/%s/points-pearson.%s", script.dir, f, save_as),
         plot = gg.sc,
         width = 16, height = 12, units = "in", dpi = 300)
}

######## zoom out by aggregating 9 months
summ.months.svm <- svm %>%
  group_by(comp1, comp2, Hop, comp, Topic, Features) %>%
  summarize(bacc.svm = mean(svm),
            sd.svm=sd(svm),
            n = n()
  )
svm.comps.mean <- svm %>%
  group_by(comp1, comp2, Hop, comp, Features) %>%
  summarize( avg = mean(svm) )

# gg.bar <- ggplot(summ.months.svm, aes(y = bacc.svm, x = Topic, fill = Topic, color = Features)) +
#   geom_bar(stat = "identity", position = "dodge") +
#   facet_grid(comp1 ~ comp2) +
#   scale_fill_brewer(palette="Set2") +
#   geom_errorbar(aes(ymin=bacc.svm-se.svm, ymax=bacc.svm+se.svm), width=.2,
#                 position=position_dodge(.9)) +
#   geom_text(aes(label=sprintf(".%2.0f", 100*round(bacc.svm, 2)) ),
#             position=position_dodge(.9), vjust=3, size = 4) +
#   coord_cartesian(ylim=c(0.5,0.8)) +
#   theme_bw() +
#   labs(y = "SVM Balanced Acc.") +
#   theme(legend.position = "right",
#         axis.title.x=element_blank(),
#         axis.text.x=element_blank(),
#         axis.ticks.x=element_blank())
# 
# gg.bar <- gg.bar + 
#   geom_hline(data = svm.comps.mean, 
#              aes(yintercept = avg, color = Features), linetype = 3) +
#   geom_text(data = svm.comps.mean, aes(label = sprintf("%.2g", avg ),
#                         x = 3, y = avg + .05, color = Features ), 
#                         size = 4, inherit.aes = FALSE)
# 
# ggsave(sprintf("%s/plots/%s/svm.%s", script.dir, Features, save_as),
#        plot = gg.bar,
#        width = 16, height = 8, units = "in", dpi = 300)

# map_feats_hjust <- function(feat){
#   if(feat)
# }
gg.paired <- ggplot(summ.months.svm, aes(x = Topic, 
                      y = bacc.svm, color = Features, group = Features)) +
  geom_pointrange(aes(ymin=bacc.svm - sd.svm, 
                      ymax=bacc.svm + sd.svm),
                  position=position_dodge(.5), size = 0.2) +
  scale_color_manual(values = c("#003FBB", "#FC4E07")) +
  facet_grid(comp1 ~ comp2) +
  theme_bw() +
  labs(y = "SVM Balanced Acc.", x = "") +
  theme(legend.position = "right") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1.0, hjust=1)) +
  geom_hline(data = svm.comps.mean,
             aes(yintercept = avg, color = Features), linetype = 3) +
  geom_text_repel(data = svm.comps.mean, aes(label = sprintf("%.2g", avg ),
                                       x = 3, y = avg, #+ ifelse(Features == "bow", -0.05, 0.05), 
                                       color = Features ),
            size = 3, inherit.aes = FALSE)

ggsave(sprintf("%s/plots/%s/paired_svm.%s", script.dir, Features, save_as),
       plot = gg.paired,
       width = 9.6, height = 7.0, units = "in", dpi = 300)


summ.months.cbc <- cbc %>%
  group_by(k, comp1, comp2, Hop, comp, Topic, Features) %>%
  summarize(bacc.protos = mean(protos),
            sd.protos=sd(protos),
            n = n()
  )

summ.months <- inner_join(summ.months.cbc, summ.months.svm,
                          by = c("comp1", "comp2", "comp", "Topic", "Hop", "n", "Features")) %>%
  mutate(diff = bacc.svm - bacc.protos)
# rm(summ.months.cbc, summ.months.svm)
# #############################################################################
# ########### plots for each Topic ###############
for (t in unique(summ.months$Topic)){
  temp <- summ.months %>% filter(Topic == t)
  error.2d <- ggplot(temp, aes(x = bacc.svm, y = bacc.protos) ) +
    geom_point(aes(color = Hop, shape = Features), size = 2) +
    geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
    facet_wrap(.~k, ncol = 2) +
    geom_abline(slope = 1, color = "black", linetype = 2, size = 0.5) +
    # geom_smooth(aes(linetype = Features), color = "blue", method = "lm", se = F, size = 0.5) +
    theme_bw() +
    geom_text_repel(data = summ.months %>% filter(Topic == t, Features == "BoW"), 
                    aes(group = comp, label = sprintf("%s", comp), 
                        color = Hop, x = bacc.svm, y = bacc.protos), 
                    size = 2.5, inherit.aes = F) +
    scale_fill_manual(hop_colors) +
    labs(x = "SVM Balanced Acc.", y = "Prototypes Balanced Acc.")
    # geom_errorbar(aes(ymin = bacc.protos - sd.protos/sqrt(n),
    #                   ymax = bacc.protos + sd.protos/sqrt(n), color = Hop)) +
    # geom_errorbarh(aes(xmin = bacc.svm - sd.svm/sqrt(n),
    #                    xmax = bacc.svm + sd.svm/sqrt(n), color = Hop))
  
    ggsave(sprintf("%s/plots/%s/error2d_%s.%s", script.dir, Features, t, save_as),
           plot = error.2d,
           width = 9.6, height = 6.4, units = "in", dpi = 300)
    rm(temp)
}

# # ### zoom out by aggregating over random splits * months * Topics
summ.months.topics.cbc <- cbc %>%
  group_by(k, comp1, comp2, Hop, comp, Features) %>%
  summarize(bacc.protos = mean(protos),
            se.protos=sd(protos)/sqrt(n()),
            n.cbc = n()
  )

summ.months.topics.svm <- svm %>%
  group_by(comp1, comp2, Hop, comp, Features) %>%
  summarize(bacc.svm = mean(svm),
            se.svm=sd(svm)/sqrt(n()),
            n.svm = n()
  )

corr.months.topics <- inner_join(cbc, svm,
          by = c( "comp1", "comp2", "comp", "Hop", "Month", "Topic", "Features") ) %>%
  group_by( comp1, comp2, comp, Hop, k, Features ) %>%
  summarize(corr = cor(svm, protos)) %>%
  mutate(correlation = cut(corr, breaks = c(-1, 0.2, 0.4, 0.6, 0.8, 1.0),
                           labels = c("negligible (-1 .2]" , "low (.2 .4]", "moderate (.4 .6]",
                           "strong (.6 .8]", "very strong (.8 1]")))

summ.months.topics <- inner_join(
  inner_join(summ.months.topics.cbc, summ.months.topics.svm,
                                 by = c( "comp1", "comp2", "comp", "Hop", "Features")),
  corr.months.topics, by = c( "comp1", "comp2", "comp", "Hop", "k", "Features" ) ) %>%
  mutate(diff = bacc.svm - bacc.protos) %>%
  verify(n.cbc == n.svm) %>%
  rename(n = n.svm) %>%
  select(-n.cbc)
rm(summ.months.topics.cbc, summ.months.topics.svm)



####
corr.counts.k <- summ.months.topics %>%
  group_by(k, correlation, Features) %>%
  count() %>%
  pivot_wider(names_from = correlation, values_from = n)
corr.counts.k[is.na(corr.counts.k)] <- 0

corr.counts.k <- tryCatch( {
  corr.counts.k %>%
    mutate(corr.counts = sprintf("%d-%d-%d-%d-%d", `negligible (-1 .2]`, `low (.2 .4]`, `moderate (.4 .6]`, `strong (.6 .8]`, `very strong (.8 1]`))
  },
  error = function(e){
    print("Error with 5 corrs.")
    corr.counts.k %>%
      mutate(corr.counts = sprintf("%d-%d-%d-%d", `low (.2 .4]`, `moderate (.4 .6]`, `strong (.6 .8]`, `very strong (.8 1]`))
  },
  error = function(e){
    print("Error with 4 corrs.")
    corr.counts.k %>%
      mutate(corr.counts = sprintf("%d-%d-%d", `low (.2 .4]`, `moderate (.4 .6]`, `strong (.6 .8]`))
  },
  error = function(e){
    print("Error with 4 corrs.")
    corr.counts.k %>%
      mutate(corr.counts = sprintf("%d-%d-%d", `moderate (.4 .6]`, `strong (.6 .8]`, `very strong (.8 1]`))
  }
)

diffs.k <- summ.months.topics %>%
  group_by(k, Features) %>%
  summarize(diff = mean(diff))

temp.k <- inner_join(corr.counts.k, diffs.k, by = c( "k", "Features" ))
rm(diffs.k, corr.counts.k)
write.csv(summ.months.topics %>% 
            mutate_if(is.numeric, round, digits = 4) %>%
            mutate(by = "Ideology"), 
          sprintf("%s/summ_ideology.csv", script.dir), 
          row.names = F, quote = F)
# # ######## balloon plot ##########

# 

# combined balloon
stars.pval <- function(x){
  stars <- c("****", "***", "**", "*", ".")
  vec <- c(0, 0.0001, 0.001, 0.01, 0.05, 1)
  i <- findInterval(x, vec)
  stars[i]
}
t.test.feats <- summ.comps %>% 
  group_by(k, comp, comp1, comp2, Hop) %>%
  summarize(
    pval.svm = t.test(svm ~ Features, var.equal = T, paired = T)$p.value,
    pval.protos = t.test(protos ~ Features, var.equal = T, paried = T)$p.value
  ) %>%
  mutate(stars.svm = stars.pval(pval.svm),
         stars.protos = stars.pval(pval.protos))
betters <- summ.comps %>% 
  pivot_wider(id_cols = c("k", "comp1", "comp2", "Hop", "comp", "Month", "Topic"),
              values_from = c("svm", "protos", "diff", "n"),
              names_from = "Features") %>%
  verify(n_SBERT == n_BoW) %>%
  rename(n = n_SBERT) %>%
  select(-n_BoW) %>%
  mutate(diff.svm = svm_SBERT - svm_BoW ,
         diff.protos = protos_SBERT - protos_BoW  ) %>%
  group_by(k, comp, comp1, comp2, Hop) %>%
  summarize(
    n = n(),
    se.diff.svm = sd(diff.svm) / sqrt(n()),
    se.diff.protos = sd(diff.protos) / sqrt(n()),
    diff.svm = mean(diff.svm),
    diff.protos = mean(diff.protos),
  ) %>%
  inner_join(t.test.feats) %>%
  mutate(better.protos.sbert = diff.protos > 0,
         better.svm.sbert =  diff.svm > 0)

summ.months.topics <- summ.months.topics %>% 
  inner_join(betters) %>%
  mutate(better.protos = ! xor(better.protos.sbert, Features == "SBERT"),
         better.svm = ! xor(better.svm.sbert, Features == "SBERT")) %>%
  select( -better.protos.sbert, -better.svm.sbert )

ggb1 <- ggplot(summ.months.topics,
              aes(x = comp1, y = comp2)) +
  geom_point( aes( fill = bacc.protos, shape = Features, 
                   #size = corr, 
                   group = Features, color = better.protos ), 
              position = position_dodge(0.7), size = 10) +
  # scale_color_distiller(palette = "RdYlGn", direction = 1, name = "Protos BAcc",
  #                       guide = guide_colorsteps(direction = "vertical")) +
  scale_color_manual(values = c("white", "black"), 
                     name = "Summarizability SBERT > BoW ?", 
                     guide = F ) +
  scale_fill_distiller(palette = "RdYlGn", direction = 1, name = "Summarizability", 
                       guide = guide_colorbar(direction = "horizontal",
                                                # label.vjust = 2, 
                                                # label.theme = element_text(angle = 90)
                                              )) +
  # scale_shape_manual(values = c( 23, 22, 21, 25, 24), name = "Corr. Strength") +
  scale_shape_manual(values = c(23, 21), 
                     guide = guide_legend(direction = "horizontal")) +
  # scale_size(range = c(4, 12), name = "Correlation", 
  #            guide = guide_legend(direction = "vertical"))+
  geom_text(aes(label = sprintf(".%2.0f", 100*round(bacc.svm, 2)), 
                group = Features  ),
            position = position_dodge(0.7),
            color = "black", size = 4.0 ) +
  geom_text(data = temp.k ,
             aes(label = sprintf(" Avg. diff %s: %.2g", Features, diff), 
                 x = 4.5, y = 1.2 + 0.35 * (Features == "SBERT")), size = 4.0 ) +
  # geom_text(data = temp.k,
  #            aes(label = sprintf("Corr.#s %6s: %s", Features, corr.counts), 
  #                x = 4.5, y = 2.2 + 0.35 * (Features == "SBERT")), size = 3.5 ) +
  labs(x = "", y = "")+
  facet_wrap(.~k, ncol = 2) +
  theme_bw() + theme(legend.position = "bottom")

ggsave(sprintf("%s/plots/%s/balloon_ideology_pres.%s", script.dir, "both", save_as),
       plot = ggb1,
       width = 9.0, height = 7.2, units = "in", dpi = 300)


ggb2 <- ggplot(summ.months.topics,
              aes(x = comp1, y = comp2)) +
  geom_point( aes( fill = bacc.protos, shape = Features, 
                   group = Features, color = better.protos ), 
              position = position_dodge(0.8), size = 10) +
  # scale_color_distiller(palette = "RdYlGn", direction = 1, name = "Protos BAcc",
  #                       guide = guide_colorsteps(direction = "vertical")) +
  scale_color_manual(values = c("white", "black"), 
                     name = "Protos SBERT > BoW ?", 
                     guide = F ) +
  scale_fill_distiller(palette = "RdYlGn", direction = 1, name = "Summarizability", 
                       guide = guide_colorbar(direction = "horizontal",
                                              barwidth = 10, barheight = 1.0
                                              # label.vjust = 2,
                                              # label.theme = element_text(angle = 90)
                       )) +
  # scale_shape_manual(values = c( 23, 22, 21, 25, 24), name = "Corr. Strength") +
  scale_shape_manual(values = c(23, 21), 
                     guide = guide_legend(direction = "horizontal")) +
  scale_size(range = c(5, 15), name = "Correlation",
             guide = guide_legend(direction = "vertical"))+
  geom_text(aes(label = sprintf(".%2.0f", 100*round(bacc.svm, 2)), 
                group = Features  ),
            position = position_dodge(0.7),
            color = "black", size = 3.5 ) +
  geom_text(data = temp.k ,
            aes(label = sprintf(" Avg. diff %s: %.2g", Features, diff), 
                x = 4.25, y = 1.0 + 0.35 * (Features == "SBERT")), size = 3.0 ) +
  geom_text(data = temp.k,
             aes(label = sprintf("Corr.#s %6s: %s", Features, corr.counts),
                 x = 4.25, y = 2.0 + 0.35 * (Features == "SBERT")), size = 3.0 ) +
  labs(x = "", y = "")+
  facet_wrap(.~k, ncol = 2) +
  theme_bw() + theme(legend.position = "bottom")

ggsave(sprintf("%s/plots/%s/balloon_ideology.%s", script.dir, "both", save_as),
       plot = ggb2,
       width = 8.5, height = 6.8, units = "in", dpi = 300)

### combined error2d
error.2d <- ggplot(summ.months.topics, 
                   aes(x = bacc.svm, y = bacc.protos) ) +
  geom_point(aes(color = Hop, shape = Features), size = 2) +
  geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
  facet_wrap(.~k, ncol = 2) +
  geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
  # geom_smooth(linetype = 3, color = "blue", method = "lm", se = F, size = 0.5) +
  theme_bw() +
  geom_text_repel(data = summ.months.topics %>% filter(Features == "SBERT"),
                  aes(label = sprintf("%s", comp), color = Hop), 
                  size = 2.5, force_pull = 0.1, force = 5.0) +
  scale_fill_manual(hop_colors) +
  labs(x = "SVM Balanced Acc.", y = "Prototypes Balanced Acc.")
  # geom_errorbar(aes(ymin = bacc.protos - se.protos,
  #                   ymax = bacc.protos + se.protos, color = Hop)) +
  # geom_errorbarh(aes(xmin = bacc.svm - se.svm,
  #                    xmax = bacc.svm + se.svm, color = Hop))

error.2d <- error.2d +
  geom_text(data = temp.k,
            aes(label = sprintf("Avg. diff %5s: %.2g", Features, diff),
                x = 0.65, y = 0.7 + 0.02 * (Features == "BoW")),
            size = 3.0, inherit.aes = FALSE )

ggsave(sprintf("%s/plots/%s/error2d.%s", script.dir, "both", save_as),
       plot = error.2d,
       width = 9.6, height = 7.2, units = "in", dpi = 300)


for(f in unique(summ.months.topics$Features)){
# # breaks = c(0.6, 0.64, 0.68, 0.72)
  temp <- summ.months.topics %>% filter(Features == f)
  ggb <- ggplot(temp, 
                aes(x = comp1, y = comp2)) +
    geom_point( aes( fill = diff, color = diff, shape = correlation), size = 10) +
    scale_color_distiller(palette = "Spectral", name = "SVM - Protos") +
    scale_fill_distiller(palette = "Spectral", name = "SVM - Protos") +
    scale_shape_manual(values = c( 23, 22, 21, 25, 24), name = "Corr. Strength") +
    scale_size(range = c(4, 14), name = "SVM Bacc")+
    geom_text(aes(label = sprintf(".%2.0f", 100*round(bacc.svm, 2)) ),
              color = "black", size = 3.5 ) +
    geom_label(data = temp.k %>% filter(Features == f), 
               aes(label = sprintf("Avg. diff: %.2g", diff), x = 4.5, y = 1.75), size = 3.5 ) +
    geom_label(data = temp.k %>% filter(Features == f), 
               aes(label = sprintf("Corr. #s: %s", corr.counts), x = 4.5, y = 1.15), size = 3.5 ) +
    labs(x = "", y = "")+
    facet_wrap(.~k, ncol = 2) +
    theme_bw()
  
  ggsave(sprintf("%s/plots/%s/balloon.%s", script.dir, f, save_as),
         plot = ggb,
         width = 9.6, height = 6.4, units = "in", dpi = 300)
  
  # ###### 2d error points plot
  error.2d <- ggplot(temp, 
                     aes(x = bacc.svm, y = bacc.protos) ) +
    geom_point(aes(color = Hop, shape = Features), size = 2) +
    geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
    facet_wrap(.~k, ncol = 2) +
    geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
    geom_smooth(linetype = 3, color = "blue", method = "lm", se = F, size = 0.5) +
    theme_bw() +
    geom_text_repel(data = temp,
                    aes(label = sprintf("%s", comp), color = Hop), size = 2.5) +
    scale_fill_manual(hop_colors) +
    labs(x = "SVM Balanced Acc.", y = "Prototypes Balanced Acc.") +
    geom_errorbar(aes(ymin = bacc.protos - se.protos,
                      ymax = bacc.protos + se.protos, color = Hop)) +
    geom_errorbarh(aes(xmin = bacc.svm - se.svm,
                       xmax = bacc.svm + se.svm, color = Hop))
  
  error.2d <- error.2d + 
    geom_text(data = temp.k %>% filter(Features == f),
              aes(label = sprintf("Avg. diff: %.2g", diff), 
                  x = 0.62, y = 0.68),
              size = 3.5, inherit.aes = FALSE )
    # geom_text(data = temp.k %>% filter(Features == "sbert"),
    #           aes(label = sprintf("Avg. diff sbert: %.2g", diff), 
    #               x = 0.6, y = 0.69),
    #           size = 3.5, inherit.aes = FALSE )
  
  ggsave(sprintf("%s/plots/%s/error2d.%s", script.dir, f, save_as),
         plot = error.2d,
         width = 9.6, height = 6.4, units = "in", dpi = 300)
  rm(temp)
}

# 
# # ######################## line-plot
gg.line <- ggplot(summ.months.topics, aes(y = bacc.protos, x = k, 
                              color = Features)) +
  geom_point() +
  geom_line(aes(x = as.numeric(k))) +
  facet_grid(comp1 ~ comp2) +
  scale_color_brewer(palette="Set1") +
  labs(x = "number of prototypes", y = "SVM Balanced Acc.") +
  theme_bw() +
  theme(legend.position = "right")

ggsave(sprintf("%s/plots/%s/line.%s", script.dir, Features, save_as),
       plot = gg.line,
       width = 15, height = 10, units = "in", dpi = 300)

#### jsd vs svm
# jsd <- read.csv(paste0(script.dir, "/vocab_diff.csv")) %>%
#   group_by(comp, topic, month) %>% ## some weird bug
#   mutate(comp1 = find_label(comp, 1), comp2 = find_label(comp, 2)) %>%
#   mutate(topic = factor(topic),
#          month = factor(month), 
#          units = factor(units)) %>%
#   mutate(hop = factor(as.numeric(comp2) - as.numeric(comp1), levels = c(1,2,3,4,5) )) %>%
#   mutate(comp = factor(sprintf("%s-%s", comp1, comp2)) ) %>%
#   rename(Topic = topic, Hop = hop, Month = month) %>%
#   mutate(all = (Month == "all" ))
# 
# vocab_clf  <- left_join(jsd, svm, 
#                          by = c("Topic", "comp1", "comp2", 
#                                 "comp", "Month", "Hop"))
# 
# gg.vocab <- ggplot(vocab_clf %>% filter(all == F), 
#                    aes (x = svm, y = cos, color = Features, shape = units)) +
#   geom_point() +
#   geom_smooth(method = "lm", se = F, size = 0.5) +
#   facet_grid(comp1~comp2) +
#   theme_bw()
# 
# ggsave(sprintf("%s/plots/%s/jsd_bacc.%s", script.dir, "both", save_as),
#        plot = gg.vocab,
#        width = 15, height = 12, units = "in", dpi = 300)
# 
# vocab.clf.summ <- vocab_clf %>%
#   group_by(comp, comp1, comp2, Hop, Features, all, units) %>%
#   summarize(se.jsd = sd(jsd) / sqrt(n()), jsd = mean(jsd),
#             se.cos = sd(cos) / sqrt(n()), cos = mean(cos),
#             se.svm = sd(svm) / sqrt(n()), svm = mean(svm))
# 
# error.2d.vocab <- ggplot(vocab.clf.summ %>% filter(all == F), 
#                    aes(x = svm, y = cos) ) +
#   geom_point(aes(color = Hop, shape = Features), size = 2) +
#   geom_line(aes(group = comp, color = Hop), linetype = 3, size = 0.5) +
#   facet_wrap(.~units) +
#   geom_abline(slope = 1, color = "black", linetype = 3, size = 0.5) +
#   # geom_smooth(linetype = 3, color = "blue", method = "lm", se = F, size = 0.5) +
#   theme_bw() +
#   geom_text_repel(data = vocab.clf.summ %>% filter(Features == "BoW"),
#                   aes(label = sprintf("%s", comp), color = Hop), size = 2.5) +
#   scale_fill_manual(hop_colors) +
#   labs(x = "Dis.", y = "JSD") 
#   # geom_errorbar(aes(ymin = bacc.protos - se.protos,
#   #                   ymax = bacc.protos + se.protos, color = Hop)) +
#   # geom_errorbarh(aes(xmin = bacc.svm - se.svm,
#   #                    xmax = bacc.svm + se.svm, color = Hop))
# 
# ggsave(sprintf("%s/plots/%s/error2d_vocab.%s", script.dir, "both", save_as),
#        plot = error.2d.vocab,
#        width = 9.6, height = 4.4, units = "in", dpi = 300)