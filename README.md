# GPT-2 (PyTorch)



A simple GPT-2 architecture implementation in PyTorch, based on Hugging Face's GPT-2. It includes token and position embeddings, causal multi-head attention, transformer blocks, and a language-model head with tied weights.



By default, it loads pretrained `openai-community/gpt2-medium` weights into the custom model, checks the forward-pass output shape, and runs interactive text generation. Test mode uses a smaller model with random weights to check the implementation.



## Run



```bash

pip install torch transformers colorama

python main.py          # GPT-2 Medium with pretrained weights

python main.py --test   # Small model with random weights

```



Type `exit` or `quit` to stop. Text generated in test mode will not be meaningful.

