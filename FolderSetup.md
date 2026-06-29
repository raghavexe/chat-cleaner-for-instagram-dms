## Basic documentation for the src

- helpers/chat-cleaner.py :
  Cleans up the instagram DM messages and removes noise.It also concats consequitvely
  sent messages and changes "your name" to "me" and "sender" to "other"

- helpers/processor.py:
  iterates through your inbox folder and runs the chat-cleaner.py through every json
  file, cleans it up and saves the result in the clean-text folder.
  [*Run this to clean up data*]

- helpers/data_gen.py
  iterates through cleaned up files, forming conversation pairs and adding them to a new combined file
  which can be fed for AI training.

- run.py
  main pipeline to run the processor and the data_gen in a consequtive order to provide the cleaned-text and the final training_data.jsonl file

- secrets.txt
  additonal noise terms you wish to include to the chat_cleaner function that is not included in the main code
