data <- read.csv("witness.csv")

system_prompt <- "you are a helpful research asisstant who helps researchers work with witness statements"

prompt_template <- "extract the mentions of masks from {{text}}. Breifly summerise these mentions"

results <- process_with_llm(data, text_column = "statement_text", id_column = "witness_id", response_column = "llm", system_prompt = system_prompt, prompt_template = prompt_template, 10, debug = FALSE )