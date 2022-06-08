from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification

from zeroshot_classifier.models.gpt2 import MODEL_NAME, HF_MODEL_NAME


if __name__ == '__main__':
    from stefutil import *

    tokenizer = GPT2TokenizerFast.from_pretrained(HF_MODEL_NAME)
    model = GPT2ForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    # Include `end-of-turn` token for sgd, cannot set `eos` for '<|endoftext|>' already defined in GPT2
    mic(tokenizer)
    mic(tokenizer.eos_token)
    mic(type(model))
