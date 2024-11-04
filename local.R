library(httr2)
library(jsonlite)
library(dplyr)

call_local_llm <- function(system_prompt = "Always answer in rhymes.",
                           user_prompt = "Introduce yourself.",
                           model = "lmstudio-ai/gemma-2b-it-GGUF",
                           temperature = 0.7,
                           max_tokens = -1,
                           stream = TRUE,
                           debug = FALSE) {
  
  if(debug) {
    cat("\n=== Debug Info ===")
    cat("\nSystem prompt:", system_prompt)
    cat("\nUser prompt:", user_prompt)
    cat("\n================\n")
  }
  
  # Construct the API endpoint
  base_url <- "http://localhost:8000/v1/chat/completions"
  
  # Prepare the messages payload
  messages <- list(
    list(role = "system", content = system_prompt),
    list(role = "user", content = user_prompt)
  )
  
  # Construct the full request body
  body <- list(
    model = model,
    messages = messages,
    temperature = temperature,
    max_tokens = max_tokens,
    stream = stream
  )
  
  if(debug) {
    cat("\n=== Request Body ===\n")
    print(toJSON(body, auto_unbox = TRUE, pretty = TRUE))
    cat("\n==================\n")
  }
  
  # Make the request
  tryCatch({
    req <- request(base_url) %>%
      req_headers("Content-Type" = "application/json") %>%
      req_body_json(body, auto_unbox = TRUE)
    
    if (stream) {
      # Handle streaming response
      response_text <- ""
      
      req %>% req_perform_stream(function(x) {
        chunk <- rawToChar(x)
        
        # Process each line in the chunk
        for (line in strsplit(chunk, "\n")[[1]]) {
          if (line != "" && !grepl("^\\s*$", line)) {
            if (grepl("^data: ", line)) {
              line <- gsub("^data: ", "", line)
            }
            if (line == "[DONE]") next
            
            tryCatch({
              parsed <- fromJSON(line)
              content <- NULL
              
              if (!is.null(parsed$choices)) {
                if (!is.null(parsed$choices$delta$content)) {
                  content <- parsed$choices$delta$content
                } else if (!is.null(parsed$choices$text)) {
                  content <- parsed$choices$text
                } else if (!is.null(parsed$choices$content)) {
                  content <- parsed$choices$content
                } else if (is.character(parsed$choices)) {
                  content <- parsed$choices
                }
              }
              
              if (!is.null(content)) {
                response_text <<- paste0(response_text, content)
                cat(content)
              }
            }, error = function(e) {
              if(debug) cat("\nError parsing JSON: ", e$message, "\n")
            })
          }
        }
        TRUE
      })
      
      return(response_text)
      
    } else {
      response <- req %>% req_perform()
      resp_data <- response %>% resp_body_json()
      
      if (!is.null(resp_data$choices)) {
        if (!is.null(resp_data$choices[[1]]$message$content)) {
          return(resp_data$choices[[1]]$message$content)
        } else if (!is.null(resp_data$choices[[1]]$text)) {
          return(resp_data$choices[[1]]$text)
        }
      }
      return(resp_data)
    }
    
  }, error = function(e) {
    message("Error making API call: ", e$message)
    return(NULL)
  })
}

process_with_llm <- function(df, 
                             text_column = "text",
                             id_column = "id",
                             response_column = "llm_response",
                             prompt_template = NULL,
                             system_prompt = "Always answer in rhymes.",
                             n_rows = NULL,
                             model = "lmstudio-ai/gemma-2b-it-GGUF",
                             temperature = 0.7,
                             max_tokens = -1,
                             stream = TRUE,
                             debug = FALSE) {
  
  # Validate inputs
  if (!text_column %in% names(df)) {
    stop(paste("Text column", text_column, "not found in dataframe"))
  }
  if (!id_column %in% names(df)) {
    stop(paste("ID column", id_column, "not found in dataframe"))
  }
  
  # Determine number of rows to process
  total_rows <- nrow(df)
  rows_to_process <- if(is.null(n_rows)) total_rows else min(n_rows, total_rows)
  
  # Create copy of dataframe and add response column if it doesn't exist
  result_df <- df
  if (!response_column %in% names(result_df)) {
    result_df[[response_column]] <- NA_character_
  }
  
  # Initialize progress tracking
  cat(sprintf("\nProcessing %d out of %d rows\n", rows_to_process, total_rows))
  pb <- txtProgressBar(min = 0, max = rows_to_process, style = 3)
  
  # Process each row
  for (i in 1:rows_to_process) {
    # Get current row's text
    current_text <- df[[text_column]][i]
    current_id <- df[[id_column]][i]
    
    if(debug) {
      cat("\n=== Row", i, "===")
      cat("\nID:", current_id)
      cat("\nOriginal text:", current_text)
    }
    
    # Skip if text is NA or empty
    if (is.na(current_text) || current_text == "") {
      message(sprintf("\nSkipping row %d (ID: %s) - empty or NA text", i, current_id))
      next
    }
    
    # Skip if already processed
    if (!is.na(result_df[[response_column]][i])) {
      message(sprintf("\nSkipping row %d (ID: %s) - already processed", i, current_id))
      next
    }
    
    # Construct user prompt using template if provided
    user_prompt <- if (!is.null(prompt_template)) {
      # Use string interpolation instead of gsub
      prompt_text <- prompt_template
      prompt_text <- sprintf(gsub("\\{\\{text\\}\\}", "%s", prompt_text), current_text)
      if(debug) {
        cat("\nTemplate:", prompt_template)
        cat("\nAfter substitution:", prompt_text)
      }
      prompt_text
    } else {
      current_text
    }
    
    if(debug) {
      cat("\nFinal prompt:", user_prompt, "\n")
    }
    
    # Process with LLM
    tryCatch({
      response <- call_local_llm(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
        model = model,
        temperature = temperature,
        max_tokens = max_tokens,
        stream = stream,
        debug = debug
      )
      
      # Store response
      if (!is.null(response) && response != "") {
        result_df[[response_column]][i] <- response
      } else {
        message(sprintf("\nWarning: Empty response for row %d (ID: %s)", i, current_id))
      }
      
    }, error = function(e) {
      message(sprintf("\nError processing row %d (ID: %s): %s", i, current_id, e$message))
    })
    
    # Update progress bar
    setTxtProgressBar(pb, i)
    
    # Add a small delay to prevent overwhelming the server
    Sys.sleep(0.1)
  }
  
  # Close progress bar
  close(pb)
  
  # Print summary
  processed_count <- sum(!is.na(result_df[[response_column]][1:rows_to_process]))
  cat(sprintf("\n\nProcessing complete:\n"))
  cat(sprintf("- Rows processed: %d\n", rows_to_process))
  cat(sprintf("- Successful responses: %d\n", processed_count))
  cat(sprintf("- Failed/skipped: %d\n", rows_to_process - processed_count))
  
  return(result_df)
}

