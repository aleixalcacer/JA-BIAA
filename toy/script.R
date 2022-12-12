
load('cargas.RData')

library(biaa)
 set.seed(1)
 ob43=biaa( as.matrix( dataf),   k=4,c=3 )


#BMM
library("blockcluster")

 set.seed(10)
 out43 <- cocluster(as.matrix(dataf), datatype = "continuous", nbcocluster = c(4, 3))
