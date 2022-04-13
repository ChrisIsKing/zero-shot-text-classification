from zeroshot_encoder.util.util import *


class CheckArg:
    """
    Raise errors when common arguments don't match the expected values
    """
    model_names = ['binary-bert', 'bert-nli', 'bi-encoder', 'dual-bi-encoder', 'gpt2-nvidia']
    domains = ['in', 'out']
    sampling_strategies = ['rand', 'vect', 'none', 'NA']
    training_strategies = ['vanilla', 'implicit', 'explicit']

    @staticmethod
    def check_mismatch(arg_type: str, arg_value: str, expected_values: List[str]):
        if arg_value not in expected_values:
            raise ValueError(f'Unexpected {logi(arg_type)}: '
                             f'expect one of {logi(expected_values)}, got {logi(arg_value)}')

    @staticmethod
    def check_model_name(model_name: str):
        CheckArg.check_mismatch('model name', model_name, CheckArg.model_names)

    @staticmethod
    def check_domain(domain: str):
        CheckArg.check_mismatch('domain', domain, CheckArg.domains)

    @staticmethod
    def check_sampling_strategy(sampling_strategy: str):
        CheckArg.check_mismatch('sampling strategy', sampling_strategy, CheckArg.sampling_strategies)

    @staticmethod
    def check_training_strategy(training_strategy: str):
        CheckArg.check_mismatch('training strategy', training_strategy, CheckArg.training_strategies)

    def __init__(self):
        self.d_name2func = dict(
            model_name=CheckArg.check_model_name,
            domain=CheckArg.check_domain,
            sampling_strategy=CheckArg.check_sampling_strategy,
            training_strategy=CheckArg.check_training_strategy
        )

    def __call__(self, **kwargs):
        for k in kwargs:
            self.d_name2func[k](kwargs[k])


ca = CheckArg()


if __name__ == '__main__':

    md_nm, dm, samp_strat, tr_strat = 'bert-nli', 'in', 'rand', 'implicit'
    ca(model_name=md_nm, domain=dm, sampling_strategy=samp_strat, training_strategy=tr_strat)
