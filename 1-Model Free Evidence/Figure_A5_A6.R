######################################
# Hyperparameter Tuning
######################################

rm(list=ls())
library(dplyr)
library(ggplot2)
library(tidyr)

#Import Data
data <-read.csv("Gamma_tau.csv")
attach(data)
data_long <- gather(data, param, return, Gamma_500:tau_05, factor_key=TRUE)

subset_gamma <-data_long %>% filter(param %in% c("Gamma_500","Gamma_1k","Gamma_8k","Gamma_20k"))
subset_tau <-data_long %>% filter(param %in% c("tau_0001","tau_001","tau_005","tau_01","tau_05"))
#Gamma
ggplot(subset_gamma, aes(x = Time.step..1e5., y = return)) + 
  geom_line(aes(color = param, linetype = param)) +  
  theme( panel.grid.major.x = element_blank() ,
         panel.grid.major.y = element_line( size=.1, color="black" ) ,panel.background = element_blank(),
         axis.line = element_line(colour = "black"),legend.position="bottom",legend.title = element_blank()
  ) + labs(y="Average Return",x="Time Steps (1e6)")+
  scale_x_continuous(limits=c(0,10),breaks=c(1:10))+
  scale_linetype_manual(values=c("dashed", "dotted","solid","dotdash"))+
  scale_color_manual(values = c("indianred", "orange","royalblue","darkgreen"))
ggsave("Gamma.png",width = 5, height = 3.5)

#tau
ggplot(subset_tau, aes(x = Time.step..1e5., y = return)) + 
  geom_line(aes(color = param, linetype = param)) +  
  theme( panel.grid.major.x = element_blank() ,
         panel.grid.major.y = element_line( size=.1, color="black" ) ,panel.background = element_blank(),
         axis.line = element_line(colour = "black"),legend.position="bottom",legend.title = element_blank()
  ) + labs(y="Average Return",x="Time Steps (1e6)")+
  scale_x_continuous(limits=c(0,10),breaks=c(1:10))+
  scale_linetype_manual(values=c("dashed", "dotted","solid","dotdash","longdash"))+
  scale_color_manual(values = c("indianred", "orange","royalblue","darkgreen","pink"))
ggsave("tau.png",width = 5, height = 3.5)




