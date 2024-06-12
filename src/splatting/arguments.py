import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        print("There")
        group = parser.add_argument_group(name)
        print("group: ", group)
        print("group.__dict__: ", group.__dict__)
        for key, value in vars(self).items():
            print("key: ", key)
            print("value: ", value)

            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0001
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005  # before 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        print("Here")
        super().__init__(parser, "Optimization Parameters")


# def get_combined_args(parser: ArgumentParser):
#     cmdlne_string = sys.argv[1:]
#     cfgfile_string = "Namespace()"
#     args_cmdline = parser.parse_args(cmdlne_string)

#     try:
#         cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
#         print("Looking for config file in", cfgfilepath)
#         with open(cfgfilepath) as cfg_file:
#             print("Config file found: {}".format(cfgfilepath))
#             cfgfile_string = cfg_file.read()
#     except TypeError:
#         print("Config file not found at")
#         pass
#     args_cfgfile = eval(cfgfile_string)

#     merged_dict = vars(args_cfgfile).copy()
#     for k, v in vars(args_cmdline).items():
#         if v is not None:
#             merged_dict[k] = v
#     return Namespace(**merged_dict)


if __name__ == "__main__":
    parser = ArgumentParser(description="Test")
    opt = OptimizationParams(parser)
    args = parser.parse_args()
    breakpoint()
    # print(opt_params.extract(args))
    # print(get_combined_args(parser))