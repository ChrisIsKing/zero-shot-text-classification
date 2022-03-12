from zeroshot_encoder.bi_encoder.jskit.encoders.bi import (
    config as bi_enc_config, set_seed,
    tokenizer, model, train_model
)
# Cannot import like this cos `bi.py` already imported, could cause duplicate `config_setup` call, loading 2 models
# from jskit.encoders.utils.train import train_model


if __name__ == '__main__':
    import transformers
    from icecream import ic

    from zeroshot_encoder.util import *

    seed = config('random-seed')
    set_seed(seed)
    transformers.set_seed(seed)

    ic(config_parser2dict(bi_enc_config))
    ic(tokenizer, type(model))
    ic(train_model)

    # model = train_model(
    #     model_train=model,
    #     tokenizer=tokenizer,
    #     contexts=contexts,
    #     candidates=candidates,
    #     labels=labels,
    #     output_dir="model_output"
    # )
