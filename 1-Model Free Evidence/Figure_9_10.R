rm(list=ls())
library(ggplot2)
df<-read.csv('figure9.csv')
ggplot(df, aes(x=G1, y=G2)) + geom_point(alpha = 0.1) + 
  geom_density_2d(bins = 5,color='purple4') +
  xlab("Predicted Gain (Measured by Doubly Robust)") + ylab("Actual Gain (Measured by Field Experiment)") + 
  theme(text = element_text(size=12),panel.background = element_blank(),axis.line = element_line(colour = "black"),panel.grid.major.x = element_line( size=.1, color="grey" ),panel.grid.major.y = element_line( size=.1, color="grey" )) +xlim(-1,4)+ylim(-1,4)

ggsave("actual_predicted.png", width = 4.2, height = 4.2, units = "in")



#Histogram of Revenue Gain Using Field Experiment
ggplot(df, aes(x=G2)) + geom_histogram(binwidth=0.1,color='purple4',fill='purple4')+
  xlab("CLV Gain (Measured by Field Experiment)")  + 
  theme(text = element_text(size=12),panel.background = element_blank(),axis.line = element_line(colour = "black"),panel.grid.major.x = element_blank(),panel.grid.major.y = element_line( size=.1, color="grey" )) 
ggsave("gain_hist.png", width = 3.9, height = 3, units = "in")
