from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aggregation_method',default='multirun', type=str, help='how many test examples to run [oneshot | multi-run]')
        parser.add_argument("--num_runs", type=int, default=3, help="Number of runs per example when using multi-run aggregation")
        parser.add_argument("--dataset_name", type=str, help="chooses what prompt to use. [HaluEval | ArenaPosition | Arena | BiasBio]")
        parser.add_argument("--role", type=str,required=True,  help="Choose which role is running. Affects input / output folder [generator | judge].")
        parser.add_argument("--max_tokens", type=int, default=128)
        parser.add_argument("--temperature", type=float, default=0.3)
        parser.add_argument("--test_limit", type=int, default=2000, help="use to limit the number of test examples.")
        parser.add_argument("--mode", type=str, default="generator", help="difines the mode of testing, which can be 'generator' , or 'judge'. This model will be the same since dependes on the server.")

        
        return parser