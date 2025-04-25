# youtube-comment-filter
School project to do filtering on youtube comments, this project uses basic filtering and fintuned LLM to achive it. 

link of the finetune model based on LLAMA3.2-3b-instruct using the script in the repo: https://www.mediafire.com/file/m9frp2gdrpiw51k/youtube-moderator.gguf/file

To run the project, you need first to creates the databases by running the database.py script. Once that's done you can run the main.py script that threat the batch.csv file.

configuration for the ollama link is in ollamaapilink.py file. current confing is localhost and youtube-moderator model.
