packages <- c(
  "digest", "uuid", "base64enc", "rlang", "fastmap", "htmltools", "jsonlite",
  "lifecycle", "cli", "fansi", "glue", "utf8", "vctrs", "pillar", "repr",
  "IRdisplay", "crayon", "pbdZMQ", "IRkernel", "Rcpp", "R6", "magrittr",
  "pkgconfig", "tibble", "dplyr", "gtable", "testthat", "brio", "sessioninfo",
  "mirt", "scales", "munsell", "colorspace", "withr", "ggplot2", "parallel"
)

installed <- rownames(installed.packages())
to_install <- setdiff(packages, installed)

if (length(to_install) > 0) {
  install.packages(to_install)
}

IRkernel::installspec(name = "r", displayname = "R")