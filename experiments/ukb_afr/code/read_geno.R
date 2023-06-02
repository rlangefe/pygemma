library(snpStats)
# Read PLINK files (.bed, .fam, .bim)
chr = 1 
plink_data <- read.plink(paste0("/net/fantasia/home/borang/Robert/UKB_AFR/Geno/AFR/chr_",chr,".bed"))

###ID info which can be matched up eid 
geno_id<- plink_data$fam$member
###Genotype
genotypes <- as(plink_data$genotypes,"numeric")
