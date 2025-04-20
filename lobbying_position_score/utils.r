library(mirt)
library(ggplot2)
library(parallel)


load_embedding <- function(score_input_path, input_file_name){
    input_file_path <- paste0(score_input_path, "/", input_file_name, ".csv")
    IG<-read.csv(input_file_path)
    return(IG)
}


get_lob_names <- function(IG){
    IG <- as.matrix(IG)
    lob_names<-IG[,1]
    return(lob_names)
}


get_bill_ids <- function(IG){
    IG<-as.matrix(IG)
    bill_ids<-colnames(IG)[-1]
    return(bill_ids)
}


get_input_matrix <- function(IG, lob_names, bill_ids, itemtype){
    IG<-as.matrix(IG)
    IG_vote <-IG[,-c(1)]
    if (itemtype == 'graded') {
        label_order <- c(1, 3, 2) # support, engage, oppose
      } else {
        stop("Invalid itemtype. Please use 'nomial' or 'graded'.")
      }
    IG_vote <- matrix(label_order[match(IG_vote, 1:4)], nrow = nrow(IG_vote), ncol = ncol(IG_vote))
    nrow(IG_vote); ncol(IG_vote)
    head(IG_vote); dim(IG_vote)
    IG <- data.frame(lob_id = lob_names, IG_vote)
    
    resp_matrix <- matrix(as.numeric(IG_vote), nrow = length(lob_names), ncol = length(bill_ids))
    head(resp_matrix); dim(resp_matrix)
    resp_matrix[1:5,1:6]
    colnames(resp_matrix) <- bill_ids
    rownames(resp_matrix) <- lob_names
    resp_matrix[1:5,1:6]
    return(resp_matrix)
        }


get_mirt_model <- function(resp_matrix, dim, itemtype) {
    mirtCluster(64)
    model <- mirt(resp_matrix, dim, itemtype=itemtype)
    mirtCluster(remove = TRUE)
    return(model)
}


get_mirt_score_df <- function(model, lob_names) {
    scores <- fscores(model, full.scores.SE=TRUE)
    
    score_df <- as.data.frame(scores)
    score_df <- data.frame(lob_id = lob_names, ideal_points = score_df)
    return(score_df)
}


save_mirt_score_df <- function(lob_score_df, bill_score_df, score_output_path) {
    lob_file_name <- paste0(score_output_path, "/", "lobbying_position_score.csv")
    bill_file_name <- paste0(score_output_path, "/", "bill_latent_score.csv")    
    write.csv(lob_score_df, file = lob_file_name, row.names = TRUE)
    write.csv(bill_score_df, file = bill_file_name, row.names = TRUE)
}


pipeline_score_df <- function(score_input_path, input_file_name, score_output_path, dim, itemtype){
    IG <- load_embedding(score_input_path, input_file_name)
    lob_names <- get_lob_names(IG)
    bill_ids <- get_bill_ids(IG)
    resp_matrix <- get_input_matrix(IG, lob_names, bill_ids, itemtype)
    model <- get_mirt_model(resp_matrix, dim, itemtype)
    plot(model)
    
    lob_score_df <- get_mirt_score_df(model, lob_names)
    bill_score_df <- get_summary(model)$rotF
    save_mirt_score_df(lob_score_df, bill_score_df, score_output_path)
    return(model)
}


get_reliability <- function(model){
    fs <- fscores(model, method='EAP', full.scores.SE=TRUE)
    clean_fs <- fs[complete.cases(fs), ]  # Extract only the rows without NA values
    theta <- clean_fs[,1]
    se <- clean_fs[,2]   
    eap_reliability <- var(theta) / (var(theta) + mean(se^2))
    return(eap_reliability)
}


get_standard_error <- function(model){
    theta_range <- seq(-4, 4, 0.1)
    info_vals <- testinfo(model, Theta = theta_range)
    se_vals <- 1 / sqrt(info_vals)
    print(se_vals)
    plot(theta_range, se_vals, type="l", main="Standard Error by Theta", ylab="SE(θ)", xlab="Theta")
}


get_summary <- function(model){
    all_summary <- summary(model)
    return(all_summary)
}