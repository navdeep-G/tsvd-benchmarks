library(ggplot2)
library(reshape2)

#Read in benchmark data
tsvd <- read.csv("/home/navdeep/tsvd-benchmarks/bench_results.csv")
tsvd$tsvd_settings <- paste0(tsvd$m, "_", tsvd$n, "_", tsvd$k, "_", tsvd$data_precision)
tsvd$tsvd_settings <- as.factor(tsvd$tsvd_settings)

#Plot data
tsvd_sub <- tsvd[,c("tsvd_settings", "h2o4gpu_cusolver", "sklearn_arpack", "h2o4gpu_power", "sklearn_random")]
tsvd_sub_melt <- melt(tsvd_sub)

ggplot(tsvd_sub_melt,aes(x = tsvd_settings,y = value)) +
  geom_bar(aes(fill = variable),stat = "identity",position = "dodge") +
  labs(title = "Benchmark TSVD", x = "Rows_Columns_k_precision", y = "Time") +
  coord_flip()