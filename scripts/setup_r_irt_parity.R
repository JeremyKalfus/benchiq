#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
cran_repo <- if (length(args) >= 1) args[[1]] else "https://cloud.r-project.org"
options(repos = c(CRAN = cran_repo))

if (!requireNamespace("mirt", quietly = TRUE)) {
  install.packages(
    "mirt",
    dependencies = c("Depends", "Imports", "LinkingTo")
  )
}

if (!requireNamespace("mirt", quietly = TRUE)) {
  stop("mirt is still unavailable after installation attempt")
}

cat(sprintf("r_version=%s\n", R.version.string))
cat(sprintf("mirt_version=%s\n", as.character(utils::packageVersion("mirt"))))
