### Load in the R libraries ###
library(Matrix)
library(MASS)
library(truncnorm)

### Set the Seed for the analysis ###
set.seed(1128)


### Overall Population Statistics ###
ind = 10000; nsnp = 2500; ###Tune number of individuals and SNPs

maf <- 0.05 + 0.45*runif(nsnp)
Geno   <- (runif(ind*nsnp) < maf) + (runif(ind*nsnp) < maf)  ###0,1,2
Geno   <- matrix(as.double(Geno),ind,nsnp,byrow = TRUE)
Xmean=apply(Geno, 2, mean); Xsd=apply(Geno, 2, sd);

for(i in 1:ncol(Geno)){
    Geno[,i] = (Geno[,i]-Xmean[i])/Xsd[i]
}

ind = nrow(Geno); nsnp = ncol(Geno)

pve=0.4 ###Set PVE for trait
ncausal= 1e3 #Set of causal SNPs, tune the proportion of causal SNPs

#Select Causal SNPs
s = 1:nsnp
s1=sample(s, ncausal, replace=F)

#Marginal effects only
Xmarginal=Geno[,s1]
beta=rnorm(dim(Xmarginal)[2])
z_marginal=c(Xmarginal%*%beta)
beta=beta*sqrt(pve/var(z_marginal))
z_marginal=Xmarginal%*%beta

#Error Terms
z_err=rnorm(ind)
z_err=z_err*sqrt((1-pve)/var(z_err))


z = z_marginal + z_err


