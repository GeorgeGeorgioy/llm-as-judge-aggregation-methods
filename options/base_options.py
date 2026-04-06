import argparse
from pathlib import Path
import torch
import models
import data


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        #self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument("--promptroot", required=True, help="path to prompts (should have subfolders trainA, trainB, valA, valB, etc)")
        parser.add_argument("--experiment_name", type=str, default="memory optimizing", help="name of the experiment. It decides where to store samples and models")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here when train")
        # model parameters
        parser.add_argument("--model_name", type=str, required=True, help="specify which model to use.[qwen7 |mistral7 | llama8] ")
        parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for the model weights and activations..| [half| bfloat16 | float32]")
        parser.add_argument("--max-model-len", type=int, default=2048, help="maximum length of the model, prompt tokens + generated tokens")

        # connection parameters
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=18017)
        parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 1")
        parser.add_argument("--start_server", action="store_true", help="assume server already running")
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.6, help="vLLM GPU memory utilization when starting server")
        parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size when starting server")

        # dataset parameters
        #parser.add_argument("--serial_batches", action="store_true", help="if true, takes images in order to make batches, otherwise takes them randomly")
       # parser.add_argument("--num_threads", default=4, type=int, help="# threads for loading data")
       # parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
       # parser.add_argument("--max_dataset_size", type=int, default=float("inf"), help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.")
       # parser.add_argument("--preprocess", type=str, default="resize_and_crop", help="scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]")
        # additional parameters
       # parser.add_argument("--epoch", type=str, default="latest", help="which epoch to load? set to latest to use latest cached model")
      #  parser.add_argument("--load_iter", type=int, default="0", help="which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]")
       # parser.add_argument("--verbose", action="store_true", help="if specified, print more debugging information")
       # parser.add_argument("--suffix", default="", type=str, help="customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}")
        # wandb parameters
        #self.initialized = True
        return parser
    

    
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        

        # process opt.suffix
        # if opt.suffix:
        #     suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
        #     opt.name = opt.name + suffix

        self.print_options(opt)
        self.opt = opt
        return self.opt

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
       # if not self.initialized:  # check if it has been initialized
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # # modify model-related parser options
        # model_name = opt.model                                        #    We read the model name that was provided via CLI.
        # model_option_setter = models.get_option_setter(model_name)    #    This function dynamically loads the corresponding model class (e.g. models/pix2pix_model.py) and returns its static method `modify_commandline_options`.

        # parser = model_option_setter(parser, self.isTrain)            #  We call that method, giving it the current parser. Internally, the model will call parser.add_argument(...) to add model-specific flags.  This mutates the same parser object by enriching it.

        #    Since new arguments were just added, we re-parse the command
        #    line so that:
        #       - any model-specific CLI values are now recognized
        #       - model-specific default values are applied


        # opt, _ = parser.parse_known_args()  # parse again with new defaults

        # # modify dataset-related parser options
        # dataset_name = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_name)
        # parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = Path(opt.checkpoints_dir) / opt.experiment_name
        expr_dir.mkdir(parents=True, exist_ok=True)
        file_name = expr_dir / "opt.txt"  # f"{opt.phase}
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")
