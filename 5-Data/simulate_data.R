# Simulate Data
rm(list=ls())
library(dplyr)
library(caret)

# Columns: buyer_id, receive_time_id,s1,...,sS,a1,...,aA,ns1,...,nsS,div_pay_amt_fillna
#n_buyer = 1000000
n_buyer = 10
#number of states
S = 375
#number of actions
A = 25
#number of time periods for each buyer
n_t = round(rnorm(n_buyer,25,5),0)
n_t = pmax(n_t,rep(10,n_buyer))
cumsum_n_t = cumsum(n_t)
n_row =sum(n_t)

buyer_id <- rep(1:n_buyer,n_t)
df_1 <-data.frame(buyer_id)
df_1 <- df_1 %>% group_by(buyer_id) %>% mutate(receive_time_id = row_number())
#state
state = matrix(round(rnorm(n_row*(S-1),0,1),2),nrow=n_row)
#the last state dimension is churn
churn = rep(0,n_row)
churn[cumsum_n_t]=1
#df should have 377 columns
df_1 <-data.frame(df_1,state,churn)

#action
action <- floor(runif(n_row,min=1,max=26))
df_1 <-data.frame(df_1,action)
df_1$action<-as.factor(df_1$action)
dmy <- dummyVars(" ~ .", data = df_1)
#df_2 should have 402 columns
df_2 <- data.frame(predict(dmy, newdata = df_1))

#next state, lead
tmp <-data.frame(buyer_id,state)

f_lead <- function(V) lead(V)
tmp2 <- tmp %>% mutate_at(.vars = 2:ncol(tmp), 
                        .funs = list(n = f_lead))

f_na <- function(V) ifelse(row_number()==n(),"NA",V)
tmp3<-tmp2 %>%
  group_by(buyer_id) %>%
  mutate_at(.vars = (S+1):ncol(tmp2), 
            .funs = list(nn = f_na))
#df_3 should have 402+374=776 columns
df_3 <- data.frame(df_2,tmp3[,c((ncol(tmp3)-S+2):ncol(tmp3))])
#div_pay_amt_fillna
div_pay_amt_fillna =rnorm(n_row,-2,2)
div_pay_amt_fillna = pmax(div_pay_amt_fillna,rep(0,n_row))
#df should have 2+375+25+375=777 columns
df<-data.frame(df_3,div_pay_amt_fillna)
write.csv(df,'/Users/xiaoliu/Dropbox/Projects/12-feiyu/1-Livestream RL/Review/publish/ToSubmit/3-Replication/simulated_data.txt')
