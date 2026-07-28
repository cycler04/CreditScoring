library(data.table)

#Load Datasets
train <- fread("../input/home-credit-default-risk/application_train.csv")
test  <- fread("../input/home-credit-default-risk/application_test.csv")
dim(train)
dim(test)

test[, TARGET := NA ]
test <- test[,colnames(train),with=F]
train[, type := 0 ]
test [, type := 1 ]

#Load ikiri_DS team best blend
sub <- fread("../input/homecredit-best-solution/stochastic_blending_v3-2_0.308rk7_0.431mw1_0.261t1_0.81050CV_0.3Adv_Leak005.csv", key="SK_ID_CURR")
train[, ypred := 1.0 ]
test [, ypred := sub$TARGET ]

#Concatenate train/test
raw <- rbind(train,test)
rm(train,test)

#Sort Feature by Number of Unique values
l <- as.integer( lapply( lapply( raw, unique ), length ) )
lo <- order(l, decreasing=T)
raw <- raw[,lo,with=F]

#Move TARGET to the left
cols <- colnames(raw)
cols <- cols[ !cols %in% c("TARGET") ]
cols <- c( cols[1], "TARGET", cols[2:length(cols)]  )
raw <- raw[,cols,with=F]
tail(raw)
gc()


#Merge User_id in dataset
dt <- fread("../input/homecredituserv8/user_id_v8.csv",key="SK_ID_CURR")
raw[, user := dt[J(raw$SK_ID_CURR)]$user_id ]

#Sort by user and age
setorderv( raw,c("user","DAYS_BIRTH","DAYS_REGISTRATION")  )
raw[, userN := .N, by="user"]
raw[, ord := 1:.N, by="user"]
raw[is.na(raw$user), ord := 1]
raw[, userf := 1*!is.na(raw$user)]
#Show number of records per user
table(raw$userN)

#Leak part
#Calculate difference of days between AGE and other date based features
#For the same user, that different is most of the time the same.
raw[, dif1 := DAYS_REGISTRATION - DAYS_BIRTH ]
raw[, dif2 := DAYS_ID_PUBLISH   - DAYS_BIRTH ]
raw[, dif3 := DAYS_EMPLOYED     - DAYS_BIRTH ]
raw[, dif4 := DAYS_LAST_PHONE_CHANGE- DAYS_BIRTH ]
tail(raw); gc()


#Weights for post processing algorithm (I optimized only after competition deadline)
par <- c(5.190,0.9497,0.530,4.194,0.639,1.033,-6.515,2.556)

#Group by dimensions (We are trying to find multiple records of the same user and take advantage of LAG/LEAD TARGET values )
feats <- c("dif1","dif2",'CODE_GENDER',"REGION_POPULATION_RELATIVE")
dt <- raw[, .N, keyby=feats ]
dt <- dt[N>1]
setorderv( dt, "N",-1  )
dt[, cid := 1:nrow(dt) ]
setkeyv( dt, feats  )
raw[, cid := dt[J(raw[,feats,with=F])]$cid  ]
setorderv( raw, c("cid","DAYS_BIRTH"),c(1,-1)  )
raw[, lag_target  := shift(TARGET,1,type="lag")  , by="cid"  ]
raw[, lead_target := shift(TARGET,1,type="lead") , by="cid"  ]

dt <- raw[ !is.na(raw$cid)  ]
dt[ , list(sum(type==1), mean(TARGET,na.rm=T),sum(type==0) ), keyby="lag_target" ]
dt[ , list(sum(type==1), mean(TARGET,na.rm=T),sum(type==0) ), keyby="lead_target" ]

un3 <- dt$SK_ID_CURR[ which( (dt$lag_target==1) ) ]
un4 <- dt$SK_ID_CURR[ which( is.na(dt$lead_target) ) ]
un5 <- dt$SK_ID_CURR[ which( (dt$lead_target==0) ) ]
un6 <- dt$SK_ID_CURR[ which( (dt$lead_target==1) ) ]
px <- which( (raw$SK_ID_CURR %in% un3) )
raw$ypred[px] <- raw$ypred[px] * par[1]
px <- which( (raw$SK_ID_CURR %in% un4) )
raw$ypred[px] <- raw$ypred[px] * par[2]
px <- which( (raw$SK_ID_CURR %in% un5)  )
raw$ypred[px] <- raw$ypred[px] * par[3]
px <- which( (raw$SK_ID_CURR %in% un6)  )
raw$ypred[px] <- raw$ypred[px] * par[4]

# #
#Group by User (We are trying to find multiple records of the same user and take advantage of LAG/LEAD TARGET values )
feats <- c("user","CODE_GENDER","NAME_CONTRACT_TYPE")
dt <- raw[ raw$userf==1 , .N, keyby=feats ]
dt <- dt[N>1]
setorderv( dt, "N",-1  )
dt[, cid := 1:nrow(dt) ]
setkeyv( dt, feats  )
raw[, cid := dt[J(raw[,feats,with=F])]$cid  ]
setorderv( raw, c("cid","DAYS_BIRTH"),c(1,-1)  )
raw[, lag_target  := shift(TARGET,1,type="lag")  , by="cid"  ]
raw[, lead_target := shift(TARGET,1,type="lead") , by="cid"  ]

dt <- raw[ !is.na(raw$cid)  ]
dt[ , list(sum(type==1), mean(TARGET,na.rm=T),sum(type==0) ), keyby="lag_target" ]
dt[ , list(sum(type==1), mean(TARGET,na.rm=T),sum(type==0) ), keyby="lead_target" ]

un1 <- dt$SK_ID_CURR[ which( is.na(dt$lag_target) ) ]
un4 <- dt$SK_ID_CURR[ which( is.na(dt$lead_target) ) ]
un5 <- dt$SK_ID_CURR[ which( (dt$lead_target==0) ) ]
un6 <- dt$SK_ID_CURR[ which( (dt$lead_target==1) ) ]
px <- which( (raw$SK_ID_CURR %in% un1))
raw$ypred[px] <- raw$ypred[px] * par[5]
px <- which( (raw$SK_ID_CURR %in% un4) )
raw$ypred[px] <- raw$ypred[px] * par[6]
px <- which( (raw$SK_ID_CURR %in% un5)  )
raw$ypred[px] <- raw$ypred[px] * par[7]
px <- which( (raw$SK_ID_CURR %in% un6)  )
raw$ypred[px] <- raw$ypred[px] * par[8]


#Write submission
sub <- raw[ raw$type==1 ,c("SK_ID_CURR","ypred"),with=F]
sub[, TARGET := frank(sub$ypred)/nrow(sub) ]
sub[, ypred := NULL ]
fwrite( sub, "simple_blend_user_leak_1.csv" )
print(sub)


