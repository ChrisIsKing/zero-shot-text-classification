from typing import NamedTuple

from transformers import TrainingArguments

from zeroshot_encoder.util import *


PT_LOSS_PAD = -100  # Pytorch indicator value for ignoring loss, used in huggingface for padding tokens


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


class MyEvalPrediction(NamedTuple):
    """
    Support `dataset_id`, see `compute_metrics` and `CustomTrainer.prediction_step`
    """
    # TODO: wouldn't work if subclass `EvalPrediction`; see https://github.com/python/mypy/issues/11721
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    dataset_ids: Union[np.ndarray, Tuple[np.ndarray]]
