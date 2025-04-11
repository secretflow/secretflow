python3 obfuscation_loop.py \
    --code "def unsafe(data): return eval('pickle.loads')(data)" \
    --gpt-model-obfuscate "gpt-4" \
    --gpt-model-detect "gpt-3.5-turbo" \
    --target-word "pickle" \
    --numpayloads 3 \
    --output-folder "avoid_pickle_3.5-turbo"
