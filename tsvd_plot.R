library(ggplot2)
library(reshape2)

#Read in benchmark data
tsvd <- read.csv("/home/navdeep/tsvd-benchmarks/results/bench_results_random_data.csv")

#Collect small, medium, and big matrix cases
tsvd_small <- subset(tsvd, m < 50000) #For bigger number of rows
tsvd_small$tsvd_settings <- paste0(tsvd_small$m, "_", tsvd_small$n, "_", tsvd_small$k, "_", tsvd_small$data_precision)
tsvd_small$tsvd_settings <- as.factor(tsvd_small$tsvd_settings)
tsvd_sub_small <- tsvd_small[,c("tsvd_settings", "h2o4gpu_cusolver", "sklearn_arpack", "h2o4gpu_power", "sklearn_random")]
tsvd_sub_small_melt <- melt(tsvd_sub_small)

tsvd_medium <- subset(tsvd, m == 50000) #For bigger number of rows
tsvd_medium$tsvd_settings <- paste0(tsvd_medium$m, "_", tsvd_medium$n, "_", tsvd_medium$k, "_", tsvd_medium$data_precision)
tsvd_medium$tsvd_settings <- as.factor(tsvd_medium$tsvd_settings)
tsvd_sub_medium <- tsvd_medium[,c("tsvd_settings", "h2o4gpu_cusolver", "sklearn_arpack", "h2o4gpu_power", "sklearn_random")]
tsvd_sub_medium_melt <- melt(tsvd_sub_medium)

tsvd_large <- subset(tsvd, m > 50000) #For bigger number of rows
tsvd_large$tsvd_settings <- paste0(tsvd_large$m, "_", tsvd_large$n, "_", tsvd_large$k, "_", tsvd_large$data_precision)
tsvd_large$tsvd_settings <- as.factor(tsvd_large$tsvd_settings)
tsvd_sub_large <- tsvd_large[,c("tsvd_settings", "h2o4gpu_cusolver", "sklearn_arpack", "h2o4gpu_power", "sklearn_random")]
tsvd_sub_large_melt <- melt(tsvd_sub_large)

#Plot benchmarks for small datasets (<50K rows)
ggplot(tsvd_sub_small_melt,aes(x = tsvd_settings,y = value)) +
  geom_bar(aes(fill = variable),stat = "identity",position = "dodge") +
  labs(title = "Benchmark TSVD (<50K rows)", x = "Rows_Columns_k_precision", y = "Time") +
  coord_flip()

#Plot benchmarks for medium datasets (=50K rows)
ggplot(tsvd_sub_medium_melt,aes(x = tsvd_settings,y = value)) +
  geom_bar(aes(fill = variable),stat = "identity",position = "dodge") +
  labs(title = "Benchmark TSVD (=50K rows)", x = "Rows_Columns_k_precision", y = "Time") +
  coord_flip()

#Plot benchmarks for large size datasets (>50K rows)
ggplot(tsvd_sub_large_melt,aes(x = tsvd_settings,y = value)) +
  geom_bar(aes(fill = variable),stat = "identity",position = "dodge") +
  labs(title = "Benchmark TSVD on Large Data (>50K rows)", x = "Rows_Columns_k_precision", y = "Time") +
  coord_flip()


#Read in benchmark data
tsvd_higgs <- read.csv("/home/navdeep/tsvd-benchmarks/results/bench_results_real_data.csv")
tsvd_higgs$k = as.factor(tsvd_higgs$k)
tsvd_sub_higgs <- tsvd_higgs[,c("k", "h2o4gpu_cusolver", "sklearn_arpack", "h2o4gpu_power", "sklearn_random")]
tsvd_sub_higgs_melt<- melt(tsvd_sub_higgs)
#Plot benchmarks for large size datasets (>50K rows)
ggplot(tsvd_sub_higgs_melt,aes(x = k, y = value)) +
  geom_bar(aes(fill = variable),stat = "identity",position = "dodge") +
  labs(title = "Benchmark TSVD on Higgs Dataset (10,999,999 by 29)", x = "K", y = "Time") +
  coord_flip()