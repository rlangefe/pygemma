library(data.table)
library(dplyr)
library(rtracklayer)
library(gwascat)
#library(reticulate)

#use_python(system("which python",intern=T))
#use_virtualenv("/net/mulan/home/rlangefe/gemma_work/test-env")

time.command = function(command, start=Sys.time()) {
  tryCatch({
    system(command)
    return(as.numeric(difftime(Sys.time(), start, units = "secs")))
  }, error = function(e) {
    # Return NA if command fails
    return(NA)
  })
}

replicate = 1

# Get cores from SLURM_CPUS_PER_TASK
cores = as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))

geno_dir<-"/net/fantasia/home/borang/Robert/UKB_EUR/Geno/EUR/"
pheno_dir<-"/net/fantasia/home/borang/Robert/UKB_EUR/Trait/"
#subsample_data_dir<-"/net/fantasia/home/borang/Robert/UKB_EUR/subsample/"
system(paste0("mkdir -p ",subsample_data_dir))
##################################################################
#
#		Read in fam file and subsampe/ Write out ID first
#
##################################################################

fam_file<-fread(paste0(geno_dir,"chr_1.fam"))
idx<-fam_file[sample(seq_len(nrow(fam_file)),N_subsample),1:2]
colnames(idx)<-c("FID","IID")
idx_name<-paste0(subsample_data_dir,"N_",N_subsample,"_rep_",replicate,".txt")
Regenie_idx_name<-paste0(subsample_data_dir,"Regenie_N_",N_subsample,"_rep_",replicate,".txt")
write.table(idx,idx_name,col.names=F,row.names=F,quote=F)
write.table(idx,Regenie_idx_name,col.names=T,row.names=F,quote=F)
######################################################################################################
#
#
#				Prepare Phenotype and PCs for GCTA/Regenie (GCTA requires eid fid trait/cov without column names)
#
#
######################################################################################################
##Prepare phenotype
###Use standing height as example
load(paste0(pheno_dir,"Continuous_Trait.RData"))
GCTA_pheno<-data.frame(FID = pheno_c1$eid, IID = pheno_c1$eid, Height = scale(pheno_c1[,2]))
GCTA_pheno_name<-paste0(subsample_data_dir,"GCTA_N_",N_subsample,"_rep_",replicate,"_pheno.txt")
Regenie_pheno_name<-paste0(subsample_data_dir,"Regenie_N_",N_subsample,"_rep_",replicate,"_pheno.txt")
write.table(GCTA_pheno,GCTA_pheno_name,col.names=F,row.names=F,quote=F)
write.table(GCTA_pheno,Regenie_pheno_name,col.names=T,row.names=F,quote=F)

if(N_covar > 0){
  ###PC file 
  pc_file<-data.frame(fread(paste0(pheno_dir,"EUR_pc.txt")))

  #pc_file = pc_file[,2:(N_covar+1)]

  GCTA_pc<-data.frame(FID = pc_file$eid, IID = pc_file$eid)
  GCTA_pc = cbind(GCTA_pc, pc_file[,2:(N_covar+1)])
  colnames(GCTA_pc) = c("FID", "IID", paste0("PC", 1:N_covar))

  GCTA_pc_name<-paste0(subsample_data_dir,"GCTA_N_",N_subsample,"_rep_",replicate,"_pc.txt")
  Regenie_pc_name<-paste0(subsample_data_dir,"Regenie_N_",N_subsample,"_rep_",replicate,"_pc.txt")
  write.table(GCTA_pc,GCTA_pc_name,col.names=F,row.names=F,quote=F)
  write.table(GCTA_pc,Regenie_pc_name,col.names=T,row.names=F,quote=F)
}
##SNP file 
GCTA_SNP_list<-fread(paste0(geno_dir,"chr_1.bim"))$V2[1:N_SNP]
GCTA_SNP_list_name<-paste0(subsample_data_dir,"GCTA_N_",N_subsample,"_rep_",replicate,"_SNP.txt")
write.table(GCTA_SNP_list,GCTA_SNP_list_name,col.names=F,row.names=F,quote=F)


geno_chrs<-paste0(geno_dir,"chr_",seq_len(22))
geno_chrs_file_name<-paste0(geno_dir,"geno_chr.txt")
write.table(geno_chrs,geno_chrs_file_name,col.names=F,row.names=F,quote=F)

############################################################################################
#
#
#				Run GCTA
#				
#
############################################################################################
###https://yanglab.westlake.edu.cn/software/gcta/#MLMA GCTA 2012
setwd(subsample_data_dir)
gcta_path<-"/net/fantasia/home/borang/software/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1"
gcta_null_dir<-"/net/fantasia/home/borang/Robert/UKB_EUR/Geno/EUR/GCTA_Null/"
grm_path<-paste0(gcta_null_dir,"grm")
sp_grm_path<-paste0(gcta_null_dir,"grm_sp")

res_file<-paste0("N_",N_subsample,"_rep_",replicate)
start = Sys.time()
if(N_covar > 0){
  gcta_model_cmd<-paste0(gcta_path," --mlma --mbfile ",geno_chrs_file_name," --grm ",grm_path, " --pheno ",GCTA_pheno_name," --qcovar ",GCTA_pc_name," --keep ",idx_name," --extract ",GCTA_SNP_list_name," --thread-num ",cores," --out ",res_file)

} else {  
  gcta_model_cmd<-paste0(gcta_path," --mlma --mbfile ",geno_chrs_file_name," --grm ",grm_path, " --pheno ",GCTA_pheno_name, " --keep ",idx_name," --extract ",GCTA_SNP_list_name," --thread-num ", cores, " --out ",res_file)
}
# system(gcta_model_cmd)

# # Time difference in seconds
# time_gcta = as.numeric(difftime(Sys.time(), start, units = "secs"))

time_gcta = time.command(gcta_model_cmd, start)

###fastGWA GCTA 2019 which is specificlly desgined for UKB data (fastGWA)
###fastGWA is a much faster version
start = Sys.time()
if(N_covar > 0){
  fastGWA_model_cmd<-paste0(gcta_path," --mbfile ",geno_chrs_file_name," --grm-sparse ",sp_grm_path," --fastGWA-mlm --pheno ",GCTA_pheno_name," --qcovar ",GCTA_pc_name," --keep ",idx_name," --extract ",GCTA_SNP_list_name," --thread-num ", cores, " --out ",res_file)
} else {
  fastGWA_model_cmd<-paste0(gcta_path," --mbfile ",geno_chrs_file_name," --grm-sparse ",sp_grm_path," --fastGWA-mlm --pheno ",GCTA_pheno_name," --keep ",idx_name," --extract ",GCTA_SNP_list_name," --thread-num ", cores, " --out ",res_file)
}
# system(fastGWA_model_cmd)
# # Time difference in seconds
# time_fastgwa = as.numeric(difftime(Sys.time(), start, units = "secs"))

#fastGWA_model_cmd = paste0(gcta_model_cmd, " >gcta_out.txt 2>&1")

time_fastgwa = time.command(fastGWA_model_cmd, start)

# # Read in contents of gcta_out.txt
# gcta_out = readLines("gcta_out.txt")

# # Check if "Error" contained in gcta_out.txt using grep
# if (length(grep("Error", gcta_out)) > 0) {
#   time_fastgwa = NA
# }

# # Remove gcta_out.txt
# system("rm gcta_out.txt")



############################################################################################
#
#
#      Run Regenie Step 1 prediction model 
#       
#
############################################################################################
regenie_path<-"/net/fantasia/home/borang/software/regenie/regenie-master/regenie"
###Use chr_all_filter to construct step 1 prediction model 
bed_file_input<-"/net/fantasia/home/borang/Robert/UKB_EUR/Geno/EUR/Regenie_Null/chr_all"
res_regenie<-paste0(subsample_data_dir,"regenie_N_",N_subsample,"_rep_",replicate)

start = Sys.time()
if(N_covar > 0){
  regenie_cmd_1<-paste0("regenie --step 1 --bed ",bed_file_input," --covarFile ",Regenie_pc_name," --phenoFile ",Regenie_pheno_name," --keep ",Regenie_idx_name," --extract ",GCTA_SNP_list_name," --threads ", cores, " --bsize 1000 --lowmem --lowmem-prefix tmp_rg --out ",res_regenie)
}else{
  regenie_cmd_1<-paste0("regenie --step 1 --bed ",bed_file_input," --phenoFile ",Regenie_pheno_name," --keep ",Regenie_idx_name," --extract ",GCTA_SNP_list_name," --threads ", cores, " --bsize 1000 --lowmem --lowmem-prefix tmp_rg --out ",res_regenie)
}
#system(regenie_cmd_1)

############################################################################################
#
#
#      Run Regenie Step 2 model 
#       
#
############################################################################################
###Step 2
bed_file_input<-"/net/fantasia/home/borang/Robert/UKB_EUR/Geno/EUR/chr_all"
pred_regenie<-paste0(res_regenie,"_pred.list")
if(N_covar > 0){
  regenie_cmd_2<-paste0("regenie --step 2 --bed ",bed_file_input," --covarFile ",Regenie_pc_name," --phenoFile ",Regenie_pheno_name," --keep ",Regenie_idx_name," --extract ",GCTA_SNP_list_name," --threads ", cores, " --bsize 1000 --lowmem --pred ",pred_regenie," --out  ",res_regenie)
}else{
  regenie_cmd_2<-paste0("regenie --step 2 --bed ",bed_file_input," --phenoFile ",Regenie_pheno_name," --keep ",Regenie_idx_name," --extract ",GCTA_SNP_list_name," --threads ", cores, " --bsize 1000 --lowmem --pred ",pred_regenie," --out  ",res_regenie)
}
#system(regenie_cmd_2)
# Time difference in seconds
#time_regenie = as.numeric(difftime(Sys.time(), start, units = "secs"))
regenie_cmd = paste0(regenie_cmd_1, " && ", regenie_cmd_2)
time_regenie = time.command(regenie_cmd, start)


############################################################################################################################################################################################################
#
#
#				Prepare GEMMA input (phenotype file , covariate file , kinship matrix file)
#				The GRM matrix is precomputed by GCTA, use it as input(check the correlation of GCTA dense GRM and GEMMA GRM here(/net/fantasia/home/borang/Robert/UKB_EUR/code/explore_grm_GCTA_Gemma.R))
#
#
#############################################################################################################################################################################################################

############################################################################################
#
#
#    GEMMA GRM 
#       
#
############################################################################################

###ReadGRMBin is a function provided by GCTA to read the binary GRM in R, and we can write out the subset GRM matrix out for GEMMA using the same GRM for GCTA and GEMMA
# ReadGRMBin=function(prefix, AllN=F, size=4){
#   sum_i=function(i){
#     return(sum(1:i))
#   }
#   BinFileName=paste(prefix,".grm.bin",sep="")
#   NFileName=paste(prefix,".grm.N.bin",sep="")
#   IDFileName=paste(prefix,".grm.id",sep="")
#   id = read.table(IDFileName)
#   n=dim(id)[1]
#   BinFile=file(BinFileName, "rb");
#   grm=readBin(BinFile, n=n*(n+1)/2, what=numeric(0), size=size)
#   NFile=file(NFileName, "rb");
#   if(AllN==T){
#     N=readBin(NFile, n=n*(n+1)/2, what=numeric(0), size=size)
#   }
#   else N=readBin(NFile, n=1, what=numeric(0), size=size)
#   i=sapply(1:n, sum_i)
#   return(list(diag=grm[i], off=grm[-i], id=id, N=N))
# }
# ReadGRMBin <- function(prefix, AllN = FALSE, size = 4) {
#   BinFileName <- paste(prefix, ".grm.bin", sep = "")
#   NFileName <- paste(prefix, ".grm.N.bin", sep = "")
#   IDFileName <- paste(prefix, ".grm.id", sep = "")
  
#   id <- read.table(IDFileName)
#   n <- dim(id)[1]
  
#   BinFile <- file(BinFileName, "rb")
#   grm <- readBin(BinFile, n = n * (n + 1) / 2, what = numeric(0), size = size)
  
#   NFile <- file(NFileName, "rb")
#   N <- if (AllN) {
#     readBin(NFile, n = n * (n + 1) / 2, what = numeric(0), size = size)
#   } else {
#     readBin(NFile, n = 1, what = numeric(0), size = size)
#   }
  
#   i <- cumsum(1:n)
  
#   return(list(diag = grm[i], off = grm[-i], id = id, N = N))
# }


# grm_path<-paste0(gcta_null_dir,"grm")

IDFileName=paste0(grm_path,".grm.id")
id = read.table(IDFileName)
# N_sample = dim(id)[1]
# grm_dense<-ReadGRMBin(grm_path)

# lower_triangular <- matrix(0, nrow = N_sample, ncol = N_sample)
# lower_triangular[lower.tri(lower_triangular, diag = FALSE)] <- grm_dense$off
# diag(lower_triangular)<-grm_dense$diag

# id%>%filter(V1%in%idx$FID)

# index_num<-match(idx$FID,id$V1)
# lower_triangular = lower_triangular[index_num,index_num]

# # Make the matrix symmetric
# symmetric_matrix <- lower_triangular + t(lower_triangular) - diag(diag(lower_triangular))


###symmetric_matrix is the GCTA constructed grm matrix, we can use GCTA computed relatedness matrix as input for GEMMA

##Subset symmtric_matrix to samples selected

#id%>%filter(V1%in%idx$FID)

index_num<-match(idx$FID,id$V1)
# grm_subset<-symmetric_matrix[index_num,index_num]

# pd = import("pandas")

# Skip rows not in index_num
# symmetric_matrix = pd$read_csv("/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/grm.csv.gz", 
#                                 header = NULL,
#                                 sep = " ",
#                                 usecols=index_num-1,
#                                 skiprows=seq(1, N_sample)[!seq(1, N_sample) %in% index_num] - 1)
print("Reading matrix...")
library(Rcpp)
sourceCpp("/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/matrix_reader.cpp")
symmetric_matrix = readAndFilterMatrix("/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/grm.csv",
                                      as.integer(index_num-1))

# symmetric_matrix = pd$read_csv("/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/grm.csv.gz", 
#                                 header = NULL,
#                                 sep = " ",
#                                 usecols=as.integer(index_num-1),
#                                 skiprows=as.integer(setdiff(seq(1, 50000), index_num) - 1),
#                                 engine = "c",
#                                 dtype = "float32")

# index_num=1:10
# symmetric_matrix = pd$read_csv("/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/grm.csv.gz", 
#                                 header = NULL,
#                                 sep = " ",
#                                 usecols=as.integer(index_num-1),
#                                 skiprows=as.integer(setdiff(seq(1, 50000), index_num) - 1),
#                                 engine = "c",
#                                 dtype = "float32")

grm_subset<-symmetric_matrix

gemma_grm_name<-paste0(subsample_data_dir,"GEMMA_","N_",N_subsample,"_rep_",replicate,".sXX.txt")
fwrite(grm_subset,gemma_grm_name,col.names=F,row.names=F,quote=F,sep=" ", nThread = 2*cores)

# Remove variables that are no longer needed to free up memory
rm(symmetric_matrix, grm_dense, grm_subset, id, index_num)

############################################################################################
#
#
#    GEMMA bed file
#       
#
############################################################################################
gemma_subset_bed<-paste0(subsample_data_dir,"GEMMA_","N_",N_subsample,"_rep_",replicate)
plink_cmd<-paste0("/net/fantasia/home/borang/software/plink2 --bfile ",paste0(geno_dir,"chr_all")," --keep ",idx_name," --extract ", GCTA_SNP_list_name," --threads ", cores," --make-bed --out ",gemma_subset_bed)
system(plink_cmd)
############################################################################################
#
#
#    GEMMA pheno file
#       
#
############################################################################################

pheno_c1_subset<-pheno_c1%>%filter(eid%in%idx$FID)
pheno_c1_subset<-pheno_c1_subset[match(idx$FID,pheno_c1_subset$eid),]
gemma_pheno_out<-scale(pheno_c1_subset[,3])

# Interpolate missing values with mean
gemma_pheno_out[is.na(gemma_pheno_out)] = mean(gemma_pheno_out, na.rm = TRUE)

gemma_pheno_out_name<-paste0(subsample_data_dir,"GEMMA_","N_",N_subsample,"_rep_",replicate,"_pheno.txt")
write.table(gemma_pheno_out,gemma_pheno_out_name,col.names=F,row.names=F,quote=F)

# Replace the 6th column in the fam file with the first column of the pheno file
fam_file = fread(paste0(gemma_subset_bed, ".fam"))
fam_file[,6] = gemma_pheno_out
fwrite(fam_file, paste0(gemma_subset_bed, ".fam"), col.names = F, row.names = F, quote = F, sep = " ")

############################################################################################
#
#
#    GEMMA pc file
#       
#
############################################################################################
if(N_covar > 0){
  pc_subset<-pc_file%>%filter(eid%in%idx$FID)
  pc_subset<-pc_subset[match(idx$FID,pc_subset$eid),]
  pc_subset_out<-pc_subset%>%select(-c(eid,Inferred.Gender))
  ###Add an intercept term to the covariate term is required
  pc_subset_out<-cbind(rep(1,nrow(pc_subset_out)),pc_subset_out)

  gemma_pc_out_name<-paste0(subsample_data_dir,"GEMMA_","N_",N_subsample,"_rep_",replicate,"_pc.txt")
  write.table(pc_subset_out,gemma_pc_out_name,col.names=F,row.names=F,quote=F)
}

############################################################################################
#
#
#    Run Gemma
#       
#
############################################################################################

setwd(subsample_data_dir)
gemma_res_out<-paste0("GEMMA_","N_",N_subsample,"_rep_",replicate)
#gemma_path<-"/net/fantasia/home/borang/software/gemma-0.98.3-linux-static"
#gemma_path = "/net/mulan/home/rlangefe/gemma_work/clean_gemma/GEMMA/bin/gemma"
gemma_path = "/net/fantasia/home/jiaqiang/shiquan_backup/Poisson_Mixed_Model/experiments/methods/LMM/gemma"

###Note that fit with covarite is computationally very slow, so recommend to regress out the covariates and fit with residuals instead, which is common in GWAS, for here I just ran without covariate
#gemma_cmd<-paste0(gemma_path, " -bfile ",gemma_subset_bed, " -p ",gemma_pheno_out_name, " -c ",gemma_pc_out_name, " -k ",gemma_grm_name ," -lmm 4 -o ",gemma_res_out)
#system(gemma_cmd)

# GEMMA failing because phenotype in fam file is -9
start = Sys.time()
if(N_covar > 0){
  gemma_cmd<-paste0(gemma_path, " -bfile ",gemma_subset_bed, " -c ",gemma_pc_out_name, " -k ",gemma_grm_name ," -lmm 1 -o ",gemma_res_out)
  #gemma_cmd<-paste0(gemma_path, " -bfile ",gemma_subset_bed, " -p ",gemma_pheno_out_name, " -c ",gemma_pc_out_name, " -k ",gemma_grm_name ," -lmm 1 -o ",gemma_res_out)
}else{
  gemma_cmd<-paste0(gemma_path, " -bfile ",gemma_subset_bed, " -k ",gemma_grm_name ," -lmm 1 -o ",gemma_res_out)
  #gemma_cmd<-paste0(gemma_path, " -bfile ",gemma_subset_bed, " -p ",gemma_pheno_out_name, " -k ",gemma_grm_name ," -lmm 1 -o ",gemma_res_out)
}
# system(gemma_cmd)

# # Time difference in seconds
# time_gemma = as.numeric(difftime(Sys.time(), start, units = "secs"))

time_gemma = time.command(gemma_cmd, start)


# Write times to csv with each column for different application
times = c(time_gcta, time_fastgwa, time_regenie, time_gemma)
times = as.data.frame(t(times))
colnames(times) = c("gcta", "fastgwa", "regenie", "gemma")
write.csv(times, "timing.csv", row.names = F)



############################################################################################
#
#
#     Check Res
#       
#
############################################################################################


# GCTA_res<-fread(paste0(subsample_data_dir,"N_",N_subsample,"_rep_",replicate,".mlma"))
# fastGWA_res<-fread(paste0(subsample_data_dir,"N_",N_subsample,"_rep_",replicate,".fastGWA"))
# regenie_res<-fread(paste0(subsample_data_dir,"regenie_N_",N_subsample,"_rep_",replicate,"_Height.regenie"))
# cor(-log10(fastGWA_res$P),regenie_res$LOG10P)
# gemma_res<-fread(paste0(subsample_data_dir,"output/",gemma_res_out,".assoc.txt"))