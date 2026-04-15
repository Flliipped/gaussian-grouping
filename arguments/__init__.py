# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 100 
        self.random_init = False
        self.train_split = False
        self._object_path = "object_mask"
        self.num_classes = 200
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.reg3d_interval = 2
        self.reg3d_k = 5
        self.reg3d_lambda_val = 2
        self.reg3d_max_points = 300000
        self.reg3d_sample_size = 1000
        
        self.geo_start_iter = 17_000
        self.geo_interval = 10
        self.geo_knn_k = 8
        self.geo_weight_lambda = 1.0
        self.geo_warmup_iters = 4000
        self.geo_lambda_pos = 1.0
        self.geo_lambda_neg = 1.0
        self.geo_smooth_lambda = 0.1
        self.geo_max_points = 200000
        self.geo_sample_size = 800
        self.geo_plane_tau = 0.01
        self.geo_neg_plane_tau = 0.02
        self.geo_spatial_pos_scale = 0.75
        self.geo_normal_pos_tau = 0.75
        self.geo_normal_neg_tau = 0.4
        self.geo_neg_margin = 0.8
        self.geo_margin = 0.3
        self.geo_hard_neg_k = 2
        self.geo_normal_source = "surface_axis"
        self.geo_use_multiview_semantics = False
        self.geo_support_views = 3
        self.geo_sem_pos_ratio = 0.7
        self.geo_sem_min_views = 2
        self.geo_sem_conf_tau = 0.7
        self.geo_sem_ignore_label = -1
        self.geo_normal_weight_lambda = 5.0
        self.geo_sem_same_boost = 1.0
        self.geo_sem_neg_boost = 1.0
        self.geo_sem_conflict_penalty = 0.75
        self.geo_confidence_enable = False
        self.geo_confidence_mode = "sugar"
        self.geo_ignore_band = 0.1
        self.geo_gate_sharpen_start = 17_000
        self.geo_gate_sharpen_end = 21_000
        self.geo_multiscale_knn = 8
        self.geo_boundary_persistence_enable = False
        self.proto_enable = False
        self.proto_weight_lambda = 0.05
        self.geo_alpha = 2.0
        self.geo_beta = 1.0
        self.geo_gamma = 1.0
        self.geo_weight_power = 2.0

        self.distill_enable = False
        self.distill_weight_lambda = 0.1
        self.distill_feature_dir = "dino_feature"
        self.distill_loss_type = "cosine"
        self.distill_mask_eps = 1e-6

        self.sugar_start_iter = 15_000
        self.sugar_interval = 10
        self.sugar_warmup_iters = 2000
        self.sugar_weight_lambda = 0.2
        self.sugar_lambda_axis = 1.0
        self.sugar_lambda_plane = 0.5
        self.sugar_lambda_flat = 0.1
        self.sugar_knn_k = 8
        self.sugar_max_points = 200000
        self.sugar_sample_size = 800
        self.sugar_lambda_opacity_entropy = 0.0
        self.sugar_opacity_entropy_start_iter = 15_000
        self.sugar_opacity_entropy_end_iter = 17_000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
