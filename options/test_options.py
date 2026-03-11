from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument("--dataset_name", type=str, default="HaluEval", help="chooses what prompt to use. [HaluEval | Arena_position | Arena_length | Arena_Preference | Bias_Bio]")
        parser.add_argument("--role", type=str, default="generator", help="Choose which role is running. Affects output folder [generator | judge].")
        parser.add_argument("--max_tokens", type=int, default=128)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--test_limit", type=int, default=5, help="use to limit the number of test examples.")
        parser.add_argument("--mode", type=str, default="generator", help="difines the mode of testing, which can be 'generator' , or 'judge'. This model will be the same since dependes on the server.")

        
        return parser