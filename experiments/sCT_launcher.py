from .tmux_launcher import Options, TmuxLauncher
import time
import torch

class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0 - nifti_sCT_CUT_cut
            Options(
                dataroot="data/model_test_nifti/train",
                name="nifti_sCT_CUT_cut",
                CUT_mode="CUT",
                input_nc=1,
                output_nc=1,
            )
        ]

    def commands(self):
        return ["nice -n 19 python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        return ["python test.py " + str(opt.set(phase='train')) for opt in self.common_options()]
