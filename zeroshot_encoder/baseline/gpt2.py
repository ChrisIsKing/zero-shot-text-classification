"""
Implementation of NVIDIA-GPT2 approach.

[Zero-shot Text Classification With Generative Language Models](https://arxiv.org/abs/1912.10165)
"""

from warnings import warn

from torch import nn
import transformers
from transformers import BatchEncoding
from transformers import AutoConfig
from transformers import GPT2TokenizerFast
from transformers import GPT2Model, GPT2LMHeadModel  # LMHead for CLM training
from transformers import Trainer, TrainingArguments, SchedulerType, TrainerCallback
from transformers import DataCollatorForLanguageModeling
from transformers.training_args import OptimizerNames
from datasets import load_metric

from zeroshot_encoder.preprocess import *


PT_LOSS_PAD = -100  # Pytorch indicator value for ignoring loss, used in huggingface for padding tokens


class ZsGPT2Tokenizer(GPT2TokenizerFast):
    """
    A wrapper around GPT2 tokenizer for 0-shot classification tokenizing
    """
    SPEC_TOKS = OrderedDict([
        ('pref_ques', '<|question|>'),  # Word embeddings
        ('pref_text', '<|text|>'),
        ('pref_answ', '<|answer|>'),
        ('type_ques', '[QUES]'),  # Type embeddings
        ('type_text', '[TEXT]'),
        ('type_answ', '[ANSW]')
    ])

    class Cache(dict):
        """
        Wrapper around caching dict, that loads metadata on corresponding dataset
        """
        def __init__(self):
            super().__init__()

        def __getitem__(self, dataset_name):
            """
            Needed cos huggingface may load cached dataset, internal cache is gone
            """
            if dataset_name not in self:
                feats = load_dataset(dataset_name, split='train').features['label']  # Pick a split arbitrarily
                feat2feat_full = {
                    'World': 'World News',
                    'Sports': 'Sports',
                    'Business': 'Business',
                    'Sci/Tech': 'Science & Technology'
                }
                n_cls = feats.num_classes
                lb2feat_str: List[str] = [feat2feat_full[feats.names[i]] for i in range(n_cls)]  # Labels = range
                self[dataset_name] = dict(n_classes=n_cls, label2feature_str=lb2feat_str)
            return super().__getitem__(dataset_name)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pad token cannot be `self.eos_token`
        # cos otherwise `DataCollatorForLanguageModeling` would override normal eos tokens
        self.add_special_tokens(dict(
            pad_token='[PAD]', additional_special_tokens=list(ZsGPT2Tokenizer.SPEC_TOKS.values())
        ))

        self.templates = config('baselines.gpt2-nvidia.templates')
        # Mapping from dataset name to label for non-benchmark cases
        self.cache: Dict[str, Dict] = ZsGPT2Tokenizer.Cache()
        self.cache_bm: datasets.ClassLabel = None

        self.ques_token, self.text_token, self.answ_token = (
            ZsGPT2Tokenizer.SPEC_TOKS[k] for k in ('pref_ques', 'pref_text', 'pref_answ')
        )  # Special tokens
        self.ques_type_token, self.text_type_token, self.answ_type_token = (
            ZsGPT2Tokenizer.SPEC_TOKS[k] for k in ('type_ques', 'type_text', 'type_answ')
        )  # Type tokens

        self.warned_desc = set()  # Warning for each dataset happens once    @property

    @property
    def max_len_single_sentence(self) -> int:
        return self.model_max_length - 2 * 3  # 3 pairs of (special start token, eos token)

    def _call_paren(self, s: str, **kwargs) -> List[int]:
        return super().__call__(s, **kwargs)['input_ids']

    def enc_spec(self, tok: str) -> int:
        """
        Encode special tokens with sanity check
        """
        id_ = self.encode(tok)
        assert len(id_) == 1
        return id_[0]  # Intended for special tokens

    def __call__(self, samples: Dict[str, Union[List, str, int]], is_benchmark=False, **kwargs):
        """
        :param samples: Data sample(s) with keys [`dataset_name`, `label`, `text`]
            Each value an element or a list of elements
        """
        max_length = kwargs.get('max_length', None)
        is_batched = isinstance(samples['label'], (tuple, list))
        if max_length is None:
            max_length = self.model_max_length
        n_token = self.model_max_length  # Indented number of token positions as in the actual architecture

        ln = len(samples['label'])
        idxs_tpl = np.random.randint(len(self.templates), size=ln)

        def call_single(i, dataset_id: int, text: str, label: int):
            dset_nm = config('benchmark.dataset_id2name')[dataset_id]
            if is_benchmark:
                descs = config(f'benchmark.datasets.{dset_nm}.labels.train')  # Descriptions; TODO: Assume `train` split
                n_cls = len(descs)
                # `label` is shared across all datasets, map to local label within dataset
                if self.cache_bm is None:
                    self.cache_bm = datasets.load_from_disk(
                        os.path.join(get_output_base(), DIR_PROJ, DIR_DSET, 'processed', 'benchmark_joined')
                    )['train'].features['label']  # TODO: assume `train` split
                # The ordering indicates int<=>str label mapping, i.e., index is int label,
                # see `process_benchmark_dataset`

                def lb_int2desc(lb: int) -> str:
                    """
                    Map from local dataset label ordinal, in range(n_cls) to the descriptor
                    """
                    return descs[lb]
                answer = self.cache_bm.int2str(label)
            else:
                n_cls, label2feature_str = (self.cache[dset_nm][k] for k in ('n_classes', 'label2feature_str'))

                def lb_int2desc(lb: int) -> str:
                    return label2feature_str[lb]
                answer = label2feature_str[label]

            idx_lbs = np.arange(n_cls)
            np.random.shuffle(idx_lbs)
            strs_lb = ' , '.join(f'" {lb_int2desc(idx)} "' for idx in idx_lbs)
            question = self.templates[idxs_tpl[i]].format(strs_lb)

            ids_ques = self._call_paren(question, **kwargs)
            ids_text = self._call_paren(text, **kwargs)
            ids_answ = self._call_paren(answer, **kwargs)
            ln_q, ln_t, ln_a = len(ids_ques), len(ids_text), len(ids_answ)
            ln_total = ln_q + ln_t + ln_a
            if ln_total > self.max_len_single_sentence:
                # Crop the text portion, keep question and label intact, i.e., ensure no classification label is cropped
                ln_t_ = self.max_len_single_sentence - (ln_q + ln_a)
                assert ln_t_ > 0
                warn(f'Sample longer than model max sequence length: {ln_total+6} > {self.model_max_length}'
                     f' - Text portion cropped: {ln_t} > {ln_t_}')
                ids_text = ids_text[:ln_t_]
            # Number of contex tokens, up until answer token, inclusive
            n_ques, n_text, n_answ = (1 + len(ids_ques) + 1), (1 + len(ids_text) + 1), (1 + len(ids_answ) + 1)
            n_cont = n_ques + n_text + 1
            ids = [
                self.enc_spec(self.ques_token), *ids_ques, self.enc_spec(self.eos_token),
                self.enc_spec(self.text_token), *ids_text, self.enc_spec(self.eos_token),
                self.enc_spec(self.answ_token), *ids_answ, self.enc_spec(self.eos_token)
            ]
            tids = [self.enc_spec(self.ques_type_token)] * n_ques + \
                   [self.enc_spec(self.text_type_token)] * n_text + \
                   [self.enc_spec(self.answ_type_token)] * n_answ
            msks = [1] * len(ids)  # Encode ids are attended for CLM
            # Context position ids, followed by output position ids
            # adding `n_token` offset for the modified positional embeddings, see `ZsGPT2Model`
            pids = list(range(n_cont)) + [i + n_token for i in range(len(ids)-n_cont)]
            assert all(len(lst_ids) == len(ids) for lst_ids in (ids, tids, msks, pids))  # Sanity check

            def pad(ints: List[int], name) -> List[int]:
                """
                Pad to max_length, truncate if necessary
                """
                if name == 'attention_mask':
                    int_pad = 0  # Ignore in attention
                elif name == 'position_ids':
                    # Arbitrary, since will be ignored, but needs to be within `n_token` for embedding mapping
                    int_pad = 0
                else:
                    # `input_id`s set to `pad_token` will be ignored by `DataCollatorForLanguageModeling`
                    int_pad = self.enc_spec(self.pad_token)
                return ints[:max_length] if len(ints) > max_length else (ints + [int_pad] * (max_length - len(ints)))
            out = {k: pad(ints, k) for k, ints in ((
                ('input_ids', ids), ('attention_mask', msks), ('token_type_ids', tids), ('position_ids', pids)
            ))}
            out['dataset_id'] = dataset_id  # For computing zero-shot classification accuracy
            return out
        if is_batched:
            ds = [call_single(i, d_id, txt, lb) for i, (d_id, txt, lb) in enumerate(zip(
                *[samples[k] for k in ['dataset_id', 'text', 'label']]
            ))]
            return BatchEncoding({k: [d[k] for d in ds] for k in ds[0]})  # Stack all the ids
        else:
            return BatchEncoding(call_single(0, *[samples[k] for k in ['dataset_id', 'text', 'label']]))


class ZsGPT2Model(GPT2Model):
    """
    Modifying the `GPT2Model` for 0-shot classification paper
    """
    def __init__(self, config_):
        super().__init__(config_)
        # Override internal state, instead of adding internal state, so that forward pass stays untouched
        # Double the positional embedding matrix, as if stacking the context & output embedding matrices together
        # See positional id assignment in `ZsGPT2Tokenizer`
        self.wpe = nn.Embedding(config_.max_position_embeddings*2, self.embed_dim)


def pprint_gpt2_input(d: Dict[str, torch.Tensor]):
    """
    Prints to console the encoded ids, positional ids and type ids as sanity check
    """
    n_ct, n_dnm, n_wd = 3, 10, 13
    n_pad = n_ct + n_dnm + 3
    ids, pids, tids, dids = (d[k].detach() for k in ('input_ids', 'position_ids', 'token_type_ids', 'dataset_id'))
    pad = tkzer.enc_spec(tkzer.pad_token)
    id2name = config('benchmark.dataset_id2name')

    for i, (ids_, did, pids_, tids_) in enumerate(zip(ids, dids, pids, tids)):
        msk = (ids_ != pad)
        ids_, pids_, tids_ = ids_[msk], pids_[msk], tids_[msk]
        print(f'{i:>{n_ct}}: {id2name[did.item()]:>{n_dnm}}', end=' ')
        for id_ in ids_:
            tok = tkzer.decode(id_)
            print(f'{tok:>{n_wd}}', end='')
        print()

        print(' ' * n_pad, end='')
        for pid in pids_:
            print(f'{pid.item():>{n_wd}}', end='')
        print()
        print(' ' * n_pad, end='')
        for tid in tids_:
            print(f'{tkzer.decode(tid):>{n_wd}}', end='')
        print()


class ZsGPT2LMHeadModel(GPT2LMHeadModel):
    """
    So that `ZsGPT2Model` is loaded
    """
    def __init__(self, config_):
        super().__init__(config_)
        self.transformer = ZsGPT2Model(config_)  # Override internal state

    def forward(self, dataset_id=None, **kwargs):
        # Function override to ignore `dataset_id`, not need in learning; Just need to pass value for evaluation
        # pprint_gpt2_input(kwargs | dict(dataset_id=dataset_id))
        # exit(1)
        return super().forward(**kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        md_ = super().from_pretrained(*args, **kwargs)  # Loads the GPT2LMHeadModel while ignoring `wpe.weight`
        md_ori = GPT2LMHeadModel.from_pretrained(*args, **kwargs)
        weight_pretrained = md_ori.transformer.wpe.state_dict()['weight']
        # Check `vars(md_ori.transformer.wpe)`, weight is the only parameter
        del md_ori

        with torch.no_grad():  # Crude loading the pretrained weights, to each half of the doubled positional embedding
            n_tok = md_.transformer.wpe.weight.shape[0]
            if n_tok == 1024 * 2:
                md_.transformer.wpe.weight[:1024, :] = weight_pretrained
                md_.transformer.wpe.weight[1024:, :] = weight_pretrained
            else:
                warn('Wrong model size, positional not loaded. This is expected in debugging')
        return md_


def tokenize_func(tokenizer_: ZsGPT2Tokenizer, dataset_name='ag_news', max_length=None):
    def _tokenize_func(sample: Dict[str, List]):
        """
        :param sample: A batch of data samples
        """
        if dataset_name != 'benchmark_joined':
            sample['dataset_id'] = [config('benchmark.dataset_name2id')[dataset_name]] * len(sample['label'])
        # Otherwise, `dataset_id` already part of input
        return tokenizer_(sample, is_benchmark=dataset_name == 'benchmark_joined', max_length=max_length)
    return _tokenize_func


def get_model_n_tokenizer(model_name='gpt2') -> Tuple[
    ZsGPT2LMHeadModel, ZsGPT2Tokenizer, DataCollatorForLanguageModeling
]:
    """
    :param model_name: Model name, one of [`debug`, `gpt2`, `gpt2-medium`]
    """
    pretrained_model_name = 'gpt2'

    if 'debug' in model_name:  # Try a smaller model for training sanity check
        if 'large' in model_name:
            n_token = 128
        else:
            n_token = 4
        conf = AutoConfig.from_pretrained('gpt2')
        # If using cpu, must be debugging and hence no `gradient_checkpointing`, see `get_train_setup`
        conf.update(dict(n_ctx=n_token, n_positions=n_token, use_cache=not torch.cuda.is_available()))
        model_ = ZsGPT2LMHeadModel.from_pretrained(pretrained_model_name, config=conf, ignore_mismatched_sizes=True)
        model_max_length = n_token
    else:
        model_max_length = 1024  # Keep max seq len of 1024, instead of 512 in paper, for longer texts & more labels
        conf = AutoConfig.from_pretrained(model_name)
        conf.update(dict(use_cache=False))  # For enabling `gradient_checkpointing`, see `get_train_setup`
        # Keep the 1024 token length, reducing to 512 tokens involves loading part of pretrained weights, complicated
        model_ = ZsGPT2LMHeadModel.from_pretrained(model_name, config=conf, ignore_mismatched_sizes=True)

    tokenizer_ = ZsGPT2Tokenizer.from_pretrained(
        pretrained_model_name, use_fast=True, model_max_length=model_max_length
    )
    model_.resize_token_embeddings(len(tokenizer_))

    return model_, tokenizer_, DataCollatorForLanguageModeling(tokenizer=tokenizer_, mlm=False)


def get_train_setup(model_name='gpt2', do_eval=True) -> TrainingArguments:
    name_ = model_name
    if name_ == 'debug-gpt-ori':
        name_ = 'gpt2'

    d_train_args = {
        'debug': dict(
            learning_rate=1e-4,
            batch_size=4,
            weight_decay=1e-2,
            num_train_epochs=4,
            lr_scheduler_type=SchedulerType.CONSTANT,
        ),
        'debug-large': dict(
            learning_rate=5e-5,
            batch_size=4,
            weight_decay=1e-2,
            num_train_epochs=40,
            lr_scheduler_type=SchedulerType.CONSTANT,
        ),
        'gpt2': dict(
            learning_rate=3e-5,
            batch_size=32,
            weight_decay=1e-2,
            num_train_epochs=5,
            lr_scheduler_type=SchedulerType.COSINE,
        ),
        'gpt2-medium': dict(
            learning_rate=4e-5,
            batch_size=8,  # TODO: low memory on machine now
            gradient_accumulation_steps=16,  # Effectively batch size 128 as in paper, to fit in memory
            weight_decay=1e-2,
            num_train_epochs=50,
            lr_scheduler_type=SchedulerType.COSINE,
        )
    }
    lr, bsz, decay, n_ep, sch, gas = (d_train_args[name_].get(k, None) for k in [
        'learning_rate', 'batch_size', 'weight_decay',
        'num_train_epochs', 'lr_scheduler_type', 'gradient_accumulation_steps'
    ])
    args = dict(
        output_dir=os.path.join(get_output_base(), DIR_PROJ, DIR_MDL, 'gpt2', model_name, now(sep='-')),
        do_train=True,
        do_eval=do_eval,
        evaluation_strategy='steps' if do_eval else 'no',
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        gradient_accumulation_steps=gas,
        eval_accumulation_steps=16,  # Saves GPU memory
        # Adam's beta1, beta2, epsilon taken from the GPT2 config in
        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=1,
        num_train_epochs=n_ep,
        lr_scheduler_type=sch,
        warmup_ratio=1e-2,
        log_level='warning',
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='epoch',
        fp16=torch.cuda.is_available(),
        fp16_full_eval=torch.cuda.is_available(),
        optim=OptimizerNames.ADAMW_TORCH,
        disable_tqdm=True,
        # Pass dataset name information down to `compute_loss` for computing text classification accuracy
        remove_unused_columns=False,
        report_to='none',
        # Set to True on CPU gives warning; Enable for fitting in `clarity1` memory
        gradient_checkpointing=torch.cuda.is_available()
    )
    args = {k: v for k, v in args.items() if v is not None}
    return TrainingArguments(**args)


def compute_metrics(eval_pred):
    """
    :param eval_pred: 2-tuple of (greedy prediction **ids**, labels)
        Intended to work with `CustomTrainer.prediction_step`
    """
    if not hasattr(compute_metrics, 'metric'):
        compute_metrics.metric = load_metric('accuracy')
    predictions, labels = eval_pred  # `argmax` performed already, see `CustomTrainer.prediction_step`
    predictions, labels = predictions[:, :-1], labels[:, 1:]  # For CLM
    labels, predictions = labels.flatten(), predictions.flatten()  # Original 2D tensor gives error
    msk_non_pad = (labels != PT_LOSS_PAD)
    labels, predictions = labels[msk_non_pad], predictions[msk_non_pad]
    return compute_metrics.metric.compute(predictions=predictions, references=labels)


def get_all_setup(
        model_name, dataset_name: str = 'ag_news', n_sample=None, random_seed=None, do_eval=True, custom_logging=True
) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast, datasets.Dataset, datasets.Dataset, Trainer]:
    if model_name == 'debug-gpt-ori':  # Sanity check: As if keep training GPT-2, with padding for simplicity
        conf = AutoConfig.from_pretrained('gpt2')
        conf.update(dict(use_cache=False))
        model_ = GPT2LMHeadModel.from_pretrained('gpt2', config=conf)
        tokenizer_ = GPT2TokenizerFast.from_pretrained('gpt2')
        data_collator_ = None
        train_args_ = get_train_setup(model_name, do_eval=do_eval)

        def group_texts(examples):
            examples = tokenizer_(examples['text'])
            # Taken from
            # https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb
            # block_size = tokenizer_.model_max_length
            block_size = 512  # To fit in memory
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result['labels'] = result['input_ids'].copy()
            return result
        map_func = group_texts
    else:
        model_, tokenizer_, data_collator_ = get_model_n_tokenizer(model_name)
        train_args_ = get_train_setup(model_name, do_eval=do_eval)
        map_func = tokenize_func(tokenizer_, dataset_name=dataset_name)

    dset_tr_, dset_vl_ = get_dset(
        dataset_name=dataset_name, map_func=map_func, remove_columns=['label', 'text'],
        n_sample=n_sample, random_seed=random_seed,
        fast='debug' not in model_name
    )
    trainer_args = dict(
        model=model_, args=train_args_, data_collator=data_collator_,
        train_dataset=dset_tr_, eval_dataset=dset_vl_, compute_metrics=compute_metrics
    )
    trainer_ = CustomTrainer(
        tokenizer=tokenizer_, custom_logging=custom_logging, compute_cls_acc=model_name != 'debug-gpt-ori',
        **trainer_args
    )
    return model_, tokenizer_, dset_tr_, dset_vl_, trainer_


class TrainPlot:
    """
    An interactive matplotlib graph to log metrics during training
    """
    def __init__(
            self,
            title='Transformer Training', train_args: TrainingArguments = None, out_dir: str = None, meta: Dict = None,
            interactive=True, save_plot=True
    ):
        assert train_args is not None and out_dir is not None
        self.title = title
        self.axes = None
        self.lines = []
        self.first = True

        self.interactive = interactive
        self.save_plot = save_plot
        self.colors = sns.color_palette(palette='husl', n_colors=7)
        self.c_tr, self.c_vl = self.colors[0], self.colors[3]

        self.train_args = train_args
        self.meta = meta
        n_data, md_sz, lr, bsz, n_ep, n_step = (
            meta[k] for k in ('#data', 'model size', 'learning rate', 'batch shape', '#epochs', '#steps')
        )

        self.out_dir = out_dir
        self.title_plot = rf'{title}, $n={n_data}$, #position = ${md_sz}$ ' \
                          + rf'$\alpha = {lr}$, batch shape=${bsz}$, #epochs=${n_ep}$, #steps=${n_step}$'
        self.title_save = f'{title}, n={n_data}, l={md_sz}, a={lr}, bsz={bsz}, ' \
                          f'n_ep={n_ep}, n_stp={n_step}, {now(sep="-")}'

    def make_plot(self):
        fig, self.axes = plt.subplots(3, 1, figsize=(16, 9))
        fig.suptitle(self.title_plot)
        for ax in self.axes:
            ax.set_xlabel('Step')
        self.axes[0].set_ylabel('Loss')
        self.axes[1].set_ylabel('Accuracy (%)')
        self.axes[2].set_ylabel('Classification Accuracy (%)')
        if self.interactive:
            plt.ion()

    def update(self, stats: List[Dict]):
        """
        Updates the plot with a new data point

        :param stats: List of training step stats
        """
        df = pd.DataFrame(stats)
        step, tr_acc, tr_loss, vl_acc, vl_loss, tr_acc_cls, vl_acc_cls = (
            (df[k] if k in df.columns else None) for k in (
                'step', 'train_acc', 'train_loss', 'eval_acc', 'eval_loss', 'train_acc_cls', 'eval_acc_cls'
            )
        )
        ax1, ax2, ax3 = self.axes
        # Re-plot, since x and y lim may change
        while ax1.lines:
            ax1.lines[-1].remove()
        while ax2.lines:
            ax2.lines[-1].remove()
        ax1.plot(step, tr_loss, label='Training Loss', c=self.c_tr, **LN_KWARGS)
        if vl_loss is not None:
            ax1.plot(step, vl_loss, label='Validation Loss', c=self.c_vl, **LN_KWARGS)
        ax2.plot(step, tr_acc * 100, label='Training Accuracy', c=self.c_tr, **LN_KWARGS)
        if vl_acc is not None:
            ax2.plot(step, vl_acc * 100, label='Validation Accuracy', c=self.c_vl, **LN_KWARGS)
        if tr_acc_cls is not None:
            ax3.plot(step, tr_acc_cls * 100, label='Training Classification Accuracy', c=self.c_tr, **LN_KWARGS)
        if vl_acc_cls is not None:
            ax3.plot(step, vl_acc_cls * 100, label='Training Classification Accuracy', c=self.c_tr, **LN_KWARGS)
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.draw()  # Needed for `ion`

    def plot_single(self, stats):
        """
        Make single static plot
        """
        self.make_plot()
        self.update(stats)
        self.finish()

    def finish(self):
        plt.ioff()  # Keep the plot window
        if self.save_plot:
            self.save()
        plt.show()

    def save(self):
        plt.savefig(os.path.join(self.out_dir, f'{self.title_save}.png'), dpi=300)


class MyLoggingCallback(TrainerCallback):
    """
    Requires
        - Tuple of (custom compute_loss log, internal training log, internal validation log) for each step
            - Intended for coupled training and evaluation
        - Accuracy as a metric is passed to `Trainer` and training metric computed in `compute_loss` and logged
    """
    def __init__(
            self, parent_trainer: Trainer, do_eval=True,
            name='Zero-shot GPT-2 Training', mode='train', interactive=True, save_plot=True
    ):
        """
        :param parent_trainer: The parent Trainer
        :param name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        had_handler = False
        hd_attr_nm = 'name_for_my_logging'  # Use a name that's unlikely to have collisions
        for hd in self.logger.handlers:  # Set console output logging
            if hasattr(hd, hd_attr_nm) and getattr(hd, hd_attr_nm) == name:
                had_handler = True
        if not had_handler:
            handler = logging.StreamHandler(stream=sys.stdout)  # For my own coloring
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(MyFormatter(with_color=True))
            # For ipython compatibility, potentially update it instead of adding new handler
            setattr(handler, hd_attr_nm, name)
            self.logger.addHandler(handler)
        self.logger_fl = logging.getLogger('trainer-file-write')  # Write out to file
        self.logger_fl.setLevel(logging.DEBUG)
        self.fl_handler = None

        self.out_dict = None
        self.out_dict_tr: Dict[str, Union[str, int, float, List]] = None
        self.is_compute_loss_on_train = True
        self.k_acc = 'acc_meta'
        self.k_cls = 'classification_acc_meta'  # See `CustomTrainer`
        self.k_cls_eval = f'{self.k_cls}_eval'

        self.parent_trainer = parent_trainer
        self.do_eval = do_eval
        args, dset_tr__, dset_vl_, md_, tokzer = (
            getattr(parent_trainer, k) for k in ['args', 'train_dataset', 'eval_dataset', 'model', 'tokenizer']
        )
        self.n_eval = len(dset_vl_)
        lr, n_ep = args.learning_rate, args.num_train_epochs
        self.bsz = args.per_device_train_batch_size * args.gradient_accumulation_steps
        seq_max_len = len(dset_tr__[0]['input_ids'])
        n_data, md_sz = len(dset_tr__), md_.config.n_positions
        self.n_step = max(math.ceil(len(dset_tr__) // self.bsz), 1) * n_ep  # #step/epoch at least 1
        self.train_meta = OrderedDict([
            ('#data', n_data), ('model size', md_sz),
            ('learning rate', lr), ('batch shape', (self.bsz, seq_max_len)), ('#epochs', n_ep), ('#steps', self.n_step)
        ])
        self.called_val_init = False
        self.log_hist: List[Dict] = []

        self.log_fnm_tpl = f'{name}, n={n_data}, l={md_sz}, a={lr}, bsz={self.bsz}, n_ep={n_ep}, {{}}'
        self.log_fnm = None  # Current logging file name template & file instance during training
        paths_ = self.parent_trainer.args.output_dir.split(os.sep)
        path_proj = paths_[paths_.index(DIR_PROJ):]
        self.out_dir = os.path.join(PATH_BASE, *path_proj)  # Keep the logging & plotting inside project directory

        self.mode = mode
        self.train_begin, self.train_end = None, None
        self.t_strt, self.t_end = None, None

        self.interactive = interactive
        self.plot = TrainPlot(
            title=name, train_args=parent_trainer.args, out_dir=self.out_dir, meta=self.train_meta, save_plot=save_plot
        )

    def set_mode(self, mode: str):
        """
        :param mode: One of ['train', 'eval']
        """
        self.mode = mode

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        self.logger_fl.removeHandler(self.fl_handler)  # Remove prior `FileHandler`, prep for next potential run
        self.fl_handler = None

        self.log_fnm = self.log_fnm_tpl.format(now(sep="-"))
        # Set file write logging
        os.makedirs(self.out_dir, exist_ok=True)
        self.fl_handler = logging.FileHandler(os.path.join(self.out_dir, f'{self.log_fnm}.log'))
        self.fl_handler.setLevel(logging.DEBUG)
        self.fl_handler.setFormatter(MyFormatter(with_color=False))
        self.logger_fl.addHandler(self.fl_handler)

        self.logger.info(f'Training started with {log_dict(self.train_meta)}')
        self.logger_fl.info(f'Training started with {log_dict(self.train_meta, with_color=False)}')
        self.t_strt = datetime.datetime.now()

        self.mode = 'train'
        self.train_begin = True
        if self.interactive:
            self.plot.make_plot()

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        if self.train_begin:
            self.train_begin = False
            self.train_end = True

            self.t_end = datetime.datetime.now()
            t = fmt_dt(self.t_end - self.t_strt)
            self.logger.info(f'Training completed in {logi(t)} ')
            self.logger_fl.info(f'Training completed in {t} ')

            if self.interactive:
                self.plot.finish()
            else:  # If didn't show plot before
                self.plot.plot_single(self.log_hist)
        self.mode = 'eval'

    def on_log(self, args: TrainingArguments, state, control, logs: Dict = None, **kwargs):
        def out_dict2str(d: Dict, return_wo_color: bool = False):
            keys_ = [
                'step', 'epoch', 'train_loss', 'eval_loss', 'train_acc', 'eval_acc',
                'train_acc_cls', 'eval_acc_cls', 'train_acc_mis', 'eval_acc_mis',
                'learning_rate'
            ]
            fmt = [
                f':>{len(str(self.n_step))}', ':6.2f', ':7.4f', ':7.4f', ':6.2f', ':6.2f',
                ':6.2f', ':6.2f', ':6.2f', ':6.2f', ':.2e'
            ]
            s_fmts = [f'{{{k}{fmt_}}}' for k, fmt_ in zip(keys_, fmt)]  # Enforce ordering

            d = {k: (
                    ('loss' in k and round(v, 4)) or
                    ('acc' in k and round(v*100, 4)) or
                    ('learning_rate' in k and round(v, 6)) or
                    v
                ) for k, v in d.items()
            }
            s_outs = [(k, fmt_.format(**{k: d[k]})) for fmt_, k in zip(s_fmts, keys_) if k in d]
            out_ = ', '.join(f'{k}={logi(s)}' for (k, s) in s_outs)
            if return_wo_color:
                out_ = out_, ', '.join(f'{k}={s}' for (k, s) in s_outs)
            return out_

        def log_update(d_out):
            out_console, out_write = out_dict2str(d_out, return_wo_color=True)
            self.logger.info(out_console)
            self.logger_fl.info(out_write)
            self.log_hist.append(d_out)
            if self.interactive:
                self.plot.update(self.log_hist)

        def cls_stats2dict(out_dict: Dict, n_sample: int, prefix: str) -> Dict:
            """
            Convert `acc_meta`, `classification_acc_meta` dict to stats for logging
            """
            stats_acc: pd.Series = pd.DataFrame(out_dict[self.k_acc]).sum(axis=0)
            stats_acc_cls: pd.Series = pd.DataFrame(out_dict[self.k_cls]).sum(axis=0)
            assert stats_acc_cls.n_missing/n_sample  == 0
            return {
                f'{prefix}_acc': stats_acc.n_acc / stats_acc.n_total,
                f'{prefix}_acc_cls': (stats_acc_cls.n_acc/stats_acc_cls.n_total) if stats_acc_cls.n_total != 0 else 0,
                # f'{prefix}_acc_mis': stats_acc_cls.n_missing/n_sample  # As a fraction
            }

        def set_eval_cls_acc():  # TODO: support gradient accumulation
            stats_cls_acc: List[Dict] = self.out_dict.pop(self.k_cls_eval)
            stats_cls_acc: Dict = {k: sum(d[k] for d in stats_cls_acc) for k in stats_cls_acc[0]}
            self.out_dict = {
                **self.out_dict,
                **cls_stats2dict(stats_cls_acc, self.n_eval, prefix='eval')
            }

        def log_default(d_stats: Dict):
            self.logger.info(log_dict(d_stats) if isinstance(d_stats, dict) else d_stats)
            self.logger_fl.info(log_dict(d_stats, with_color=False) if isinstance(d_stats, dict) else d_stats)

        if state.is_local_process_zero:
            if self.mode == 'train':
                step = state.global_step
                if self.do_eval:
                    if 'src' in logs and logs['src'] == 'compute_loss':  # Custom added metric computation
                        if step == 0:  # Before model runs, initial call
                            if not self.called_val_init:  # Prevents circular logging call, see Trainer.evaluate()
                                # Got to here, cos the 1st, training compute_loss logging
                                assert self.is_compute_loss_on_train
                                self.called_val_init = True
                                tr_acc, tr_loss, n_ep = (logs[k] for k in ('acc', 'loss', 'epoch'))
                                self.out_dict: Dict[str, Union[str, int, float, List]] = {
                                    **dict(step=step, epoch=0, train_acc=tr_acc, train_loss=tr_loss,),
                                    **cls_stats2dict(logs[self.k_cls], self.bsz, prefix='train')
                                }

                                # Prep for Trainer internal evaluation call
                                self.is_compute_loss_on_train = False
                                out: Dict = self.parent_trainer.evaluate()  # Expanded to branch below then comes back
                                # Disable, seems like an edge case 1st training step, not sure ow
                                # self.is_compute_loss_on_train = True
                                n_ep_, vl_acc, vl_loss = (out.get(k, None) for k in (
                                    'epoch', 'eval_accuracy', 'eval_loss'
                                ))
                                assert all(elm is not None for elm in (n_ep, vl_acc, vl_loss))
                                assert n_ep == n_ep_ and n_ep == 0
                                # python3.6 compatibility
                                self.out_dict.update(dict(eval_acc=vl_acc, eval_loss=vl_loss))

                                set_eval_cls_acc()
                                log_update(self.out_dict)
                            elif not self.is_compute_loss_on_train:  # `compute_loss` ran on evaluation set
                                # => Keep track of the batch-wise classification accuracy
                                if self.k_cls_eval not in self.out_dict:
                                    self.out_dict[self.k_cls_eval] = [logs[self.k_cls]]
                                else:
                                    self.out_dict[self.k_cls_eval].append(logs[self.k_cls])
                        else:  # Need to look for the accuracy calculated for the training batch
                            # Heuristic: 1st call to `compute_loss` corresponds to training
                            if self.is_compute_loss_on_train:
                                self.is_compute_loss_on_train = False
                                acc, loss = logs.get('acc', None), logs.get('loss', None)
                                assert acc is not None and loss is not None
                                if self.out_dict is None:
                                    # Now is the 1st call, after logging for last batch completes
                                    self.out_dict = {
                                        **dict(step=step, train_acc=acc, train_loss=loss),
                                        **cls_stats2dict(logs[self.k_cls], self.bsz, prefix='train')
                                    }
                            else:  # On eval set, keep track like above
                                if self.k_cls_eval not in self.out_dict:
                                    self.out_dict[self.k_cls_eval] = [logs[self.k_cls]]
                                else:
                                    self.out_dict[self.k_cls_eval].append(logs[self.k_cls])
                    elif 'loss' in logs:  # Internal training log
                        # Edge case step = 1: Before training start, i.e. step=1, stats for training already logged,
                        # But log anyway, for after gradient update, evaluation loss changes
                        tr_loss, lr, n_ep = (logs.get(k, None) for k in ('loss', 'learning_rate', 'epoch'))
                        assert all(elm is not None for elm in (tr_loss, lr, n_ep))
                        tr_loss_compute: int = self.out_dict.get('train_loss', None)
                        # Without overriding `_maybe_log_save_evaluate`,
                        # can only get the training loss with 4 decimal place
                        assert round(tr_loss_compute, 4) == tr_loss
                        # See Trainer.train(); compute_loss executes before step increments
                        assert self.out_dict['step'] == step-1  # Override step & loss
                        self.out_dict.update(dict(step=step, train_loss=tr_loss, lr=lr, epoch=n_ep))
                    elif 'eval_loss' in logs:
                        if step != 0:
                            vl_loss, vl_acc, n_ep_ = (
                                logs.get(k, None) for k in ('eval_loss', 'eval_accuracy', 'epoch')
                            )
                            assert all(elm is not None for elm in (vl_loss, vl_acc, n_ep_))
                            assert step == self.out_dict['step']
                            assert n_ep_ == self.out_dict['epoch']
                            # python3.6 compatibility
                            self.out_dict.update(dict(eval_loss=vl_loss, eval_acc=vl_acc))

                            set_eval_cls_acc()
                            log_update(self.out_dict)
                            self.out_dict = None
                            self.is_compute_loss_on_train = True
                    elif any('runtime' in k for k in logs.keys()):
                        log_default(logs)
                    else:
                        print('unhandled case', logs)
                        exit(1)
                else:  # Only training without evaluation supported
                    if 'src' in logs and logs['src'] == 'compute_loss':
                        # For gradient_accumulation, many batches of `compute_loss` may be called,
                        # before going into train logging
                        # Loss here is per batch, not per gradient update, ignore
                        if self.out_dict_tr is None:
                            n_ep = logs['epoch']
                            self.out_dict_tr = {'step': step, 'epoch': n_ep, self.k_acc: [logs[self.k_acc]]}
                            # Aggregate accuracy & classification accuracy counts
                            if self.parent_trainer.compute_cls_acc:
                                self.out_dict_tr[self.k_cls] = [logs[self.k_cls]]
                        else:  # Later batch in the same gradient accumulation
                            step_, n_ep = self.out_dict_tr['step'], self.out_dict_tr['epoch']
                            n_ep_ = logs['epoch']
                            assert step_ == step and n_ep_ == n_ep
                            self.out_dict_tr[self.k_acc].append(logs[self.k_acc])
                            if self.parent_trainer.compute_cls_acc:
                                self.out_dict_tr[self.k_cls].append(logs[self.k_cls])
                    elif 'loss' in logs:  # The Trainer default training loss logging
                        # Take the averaging by parent `Trainer` for granted
                        self.out_dict_tr.update(cls_stats2dict(self.out_dict_tr, self.bsz, prefix='train'))
                        del self.out_dict_tr[self.k_acc]
                        del self.out_dict_tr[self.k_cls]
                        self.out_dict_tr['learning_rate'] = logs['learning_rate']
                        self.out_dict_tr['train_loss'] = logs['loss']
                        log_update(self.out_dict_tr)
                        self.out_dict_tr = None  # Rest for next global step
                    elif any('runtime' in k for k in logs.keys()):
                        self.logger.info(log_dict(logs) if isinstance(logs, dict) else logs)
                    else:
                        print('unhandled case', logs)
                        exit(1)
            else:
                if 'src' not in logs:  # Skip custom compute_loss logging
                    log_default(logs)


class ColoredPrinterCallback(TrainerCallback):
    def __init__(self, name='Zero-shot GPT-2 Training'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        had_handler = False
        hd_attr_nm = 'name_for_my_logging'
        for hd in self.logger.handlers:
            if hasattr(hd, hd_attr_nm) and getattr(hd, hd_attr_nm) == name:
                had_handler = True
        if not had_handler:
            handler = logging.StreamHandler(stream=sys.stdout)  # For my own coloring
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(MyFormatter())
            setattr(handler, hd_attr_nm, name)
            self.logger.addHandler(handler)

    def on_log(self, args, state, control, logs_=None, **kwargs):
        if state.is_local_process_zero:
            self.logger.info(log_dict(logs_) if isinstance(logs_, dict) else logs_)


class CustomTrainer(Trainer):
    def __init__(self, tokenizer: ZsGPT2Tokenizer = None, custom_logging=True, compute_cls_acc=True, **kwargs):
        super().__init__(**kwargs)
        assert 'args' in kwargs
        self.custom_logging = custom_logging
        self.compute_cls_acc = compute_cls_acc

        self.tokenizer = tokenizer  # TODO: generalize to more tokenizers?
        self.post_init()

    def post_init(self):
        callbacks = self.callback_handler.callbacks
        self.callback_handler.callbacks = [  # Remove internal callback
            c for c in callbacks if str(c.__class__) != "<class 'transformers.trainer_callback.PrinterCallback'>"
        ]

        if self.custom_logging:
            self.add_callback(MyLoggingCallback(self, do_eval=self.args.do_eval, interactive=False))
        else:
            self.add_callback(ColoredPrinterCallback())

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override `Trainer.compute_loss` for logging accuracy
            - Note that both training and validation calls `compute_loss`
                => Further logic needs to determine accuracy for which dataset

        Modified from https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/4?u=stefanh
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # ========================== Begin of added ==========================
        inputs: Dict[str, torch.Tensor]
        d_log = None
        if self.custom_logging and 'labels' in inputs:
            preds = outputs.logits.detach().argmax(axis=-1)
            labels_ = inputs['labels'].detach()
            # CLM, predicting the next token given current, so shift
            # Last prediction is not part of input label, 1st input is fed into model & not predicted
            preds, labels_ = preds[:, :-1], labels_[:, 1:]
            mask_non_pad = labels_ != PT_LOSS_PAD  # Consider only the actual tokens for accuracy
            preds_non_pad, labels_non_pad = preds[mask_non_pad], labels_[mask_non_pad]
            matches: torch.Tensor = (preds_non_pad == labels_non_pad)
            d_log = dict(src='compute_loss', acc_meta=dict(n_acc=matches.sum().item(), n_total=preds_non_pad.numel()))

            if self.compute_cls_acc:
                token_type_ids, dataset_id = inputs['token_type_ids'].detach(), inputs['dataset_id'].detach()

                id_att = self.tokenizer.enc_spec(self.tokenizer.answ_type_token)
                id_answ = self.tokenizer.enc_spec(self.tokenizer.answ_token)
                id_eos = self.tokenizer.enc_spec(self.tokenizer.eos_token)
                sample2idxs: Dict[int, List[int]] = {
                    i_sample: (row == id_att).nonzero().flatten().tolist()
                    for i_sample, row in enumerate(token_type_ids[:, 1:])  # Also shift by 1
                }

                # For each unique row/sample with answer tokens present, check if it forms a classification label string
                def get_labels(i, idxs_):
                    if idxs_:
                        lbs = labels_[i, idxs_].tolist()  # Inputs are labels
                        assert lbs[0] == id_answ  # Remove answer special prefix token & potentially the ending token
                        idxs_, lbs = idxs_[1:], lbs[1:]
                        if lbs:  # Labels are still available
                            if lbs[-1] == id_eos:
                                idxs_, lbs = idxs_[:-1], lbs[:-1]

                            dset_id = dataset_id[i].item()
                            dnm_ = config('benchmark.dataset_id2name')[dset_id]
                            descs = config(f'benchmark.datasets.{dnm_}.labels.train')  # TODO: assume `train` mode
                            if self.tokenizer.decode(lbs) in descs:
                                return dict(idxs=idxs_, labels=lbs)

                sample2idxs_n_lbs = {i_sample: get_labels(i_sample, idxs) for i_sample, idxs in sample2idxs.items()}
                sample2idxs_n_lbs = {k: v for k, v in sample2idxs_n_lbs.items() if v is not None}
                d_log['classification_acc_meta'] = dict(
                    # prediction ids match label ids
                    n_acc=sum(
                        preds[i_sample, d['idxs']].tolist() == d['labels']
                        for i_sample, d in sample2idxs_n_lbs.items()
                    ),
                    n_total=len(sample2idxs_n_lbs),  # Number of samples with complete label
                    n_missing=dataset_id.numel()-len(sample2idxs_n_lbs)
                )
        # ========================== End of added ==========================

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # ========================== Begin of added ==========================
        if self.custom_logging and 'labels' in inputs:
            d_log['loss'] = loss.detach().item()
            self.log(d_log)
        # ========================== End of added ==========================

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """
        Override `Trainer.prediction_step` for reducing memory footprint
        """
        # ========================== Begin of added =========================
        from transformers.file_utils import is_sagemaker_mp_enabled
        from transformers.trainer_pt_utils import nested_detach
        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat
        # ========================== End of added =========================

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        # ========================== Begin of added =========================
        # Compute the labels right away,
        # instead of potentially concatenating the original evaluation matrix of shape (#eval, #model size, #vocab)
        logits = logits.argmax(dim=-1)
        # ========================== End of added =========================
        return loss, logits, labels


if __name__ == '__main__':
    from icecream import ic

    from zeroshot_encoder.util import *

    seed = config('random-seed')
    transformers.set_seed(seed)

    # dnm = 'ag_news'
    dnm = 'benchmark_joined'

    # nm = 'debug'
    # nm = 'debug-gpt-ori'
    # nm = 'debug-large'
    # nm = 'gpt2'
    nm = 'gpt2-medium'

    # n = 1
    # n = 1024
    n = None
    md, tkzer, dset_tr, dset_vl, trainer = get_all_setup(
        nm, dnm, do_eval=False, custom_logging=True, n_sample=n, random_seed=seed
    )
    # ic(trainer.args)
    trainer.train()
    trainer.save_model(os.path.join(trainer.args.output_dir))
    trainer.evaluate()
