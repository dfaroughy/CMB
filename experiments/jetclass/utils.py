import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class PlotModule:
    def __init__(self, generated_sample, test_sample, postprocess=False):
        self.gen = generated_sample
        self.test = test_sample
        self.mask_gen = (generated_sample.mask > 0).squeeze()
        self.mask_test = (test_sample.mask > 0).squeeze()
        self.postprocess = postprocess

    def particle_kinematics(self, label, save_path=None):
        args_gen = {
            "stat": "density",
            "log_scale": (False, True),
            "fill": False,
            "color": "darkred",
            "lw": 0.75,
            "label": "generated (t=1)",
        }
        args_tar = {
            "stat": "density",
            "log_scale": (False, True),
            "fill": True,
            "color": "k",
            "lw": 0.3,
            "alpha": 0.2,
            "label": label,
        }

        _, ax = plt.subplots(1, 3, figsize=(10, 3))

        if self.postprocess:
            binrange, binwidth = (-10, 750), 10
        else:
            binrange, binwidth = (-2, 30), 0.1

        self.test.histplot(
            "pt",
            mask=self.mask_test,
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $p_t$ [GeV]",
            ax=ax[0],
            **args_tar,
        )
        self.gen.histplot(
            "pt",
            mask=self.mask_gen,
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $p_t$ [GeV]",
            ax=ax[0],
            **args_gen,
        )

        if self.postprocess:
            binrange, binwidth = (-1.5, 1.5), 0.02
        else:
            binrange, binwidth = (-5, 5), 0.05

        self.test.histplot(
            "eta_rel",
            mask=self.mask_test,
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $\Delta \eta$",
            ax=ax[1],
            **args_tar,
        )
        self.gen.histplot(
            "eta_rel",
            mask=self.mask_gen,
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $\Delta \eta$",
            ax=ax[1],
            **args_gen,
        )
        self.test.histplot(
            "phi_rel",
            mask=self.mask_test,
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $\Delta \phi$",
            ax=ax[2],
            **args_tar,
        )
        self.gen.histplot(
            "phi_rel",
            mask=self.mask_gen,
            binrange=binrange,
            binwidth=binwidth,
            xlabel=r"particle $\Delta \phi$",
            ax=ax[2],
            **args_gen,
        )

        ax[0].legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path / "particle_level_kinematics.png")
        else:
            plt.show()

    def jet_kinematics(self, label, save_path=None):
        args_gen = {
            "stat": "density",
            "log_scale": (False, True),
            "fill": False,
            "color": "darkred",
            "lw": 0.75,
            "label": "generated (t=1)",
        }
        args_tar = {
            "stat": "density",
            "log_scale": (False, True),
            "fill": True,
            "color": "k",
            "lw": 0.3,
            "alpha": 0.2,
            "label": label,
        }
        pass

    def flavor_fractions(self, label, save_path=None):
        args_sam = {
            "stat": "density",
            "log_scale": (False, True),
            "fill": False,
            "color": "darkred",
            "lw": 0.75,
            "label": "generated (t=1)",
        }
        args_tar = {
            "stat": "density",
            "log_scale": (False, True),
            "fill": True,
            "color": "k",
            "lw": 0.3,
            "alpha": 0.2,
            "label": label,
        }

        _, ax = plt.subplots(1, 1, figsize=(3, 3))

        sns.histplot(
            self.test.discrete[self.mask_test].squeeze(),
            binrange=(-0.1, 7.1),
            element="step",
            discrete=True,
            **args_tar,
        )
        sns.histplot(
            self.gen.discrete[self.mask_gen].squeeze(),
            binrange=(-0.1, 7.1),
            element="step",
            discrete=True,
            **args_sam,
        )

        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("Particle Flavor")
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(
            [
                r"$\gamma$",
                r"$h^0$",
                r"$h^-$",
                r"$h^+$",
                r"$e^-$",
                r"$e^+$",
                r"$\mu^-$",
                r"$\mu^+$",
            ]
        )
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path / "flavor_fractions.png")
        else:
            plt.show()

    def flavor_multiplicities(self, label, save_path=None):
        args_gen = {
            "stat": "density",
            "log_scale": (False, False),
            "fill": False,
            "color": "darkred",
            "lw": 0.75,
            "label": "generated (t=1)",
        }
        args_tar = {
            "stat": "density",
            "log_scale": (False, False),
            "fill": True,
            "color": "k",
            "lw": 0.3,
            "alpha": 0.2,
            "label": label,
        }
        _, ax = plt.subplots(2, 4, figsize=(11, 5))
        dic = {
            0: r"$\gamma$",
            1: r"$h^0$",
            2: r"$h^-$",
            3: r"$h^+$",
            4: r"$e^-$",
            5: r"$e^+$",
            6: r"$\mu^-$",
            7: r"$\mu^+$",
        }

        for n in [0, 1, 2, 3]:
            gen_counts = (self.gen.discrete == n) * self.mask_gen.unsqueeze(-1)
            gen_counts = gen_counts.sum(dim=1)
            target_counts = (self.test.discrete == n) * self.mask_test.unsqueeze(-1)
            target_counts = target_counts.sum(dim=1)
            sns.histplot(
                target_counts.squeeze(),
                discrete=True,
                ax=ax[0, n],
                element="step",
                **args_tar,
            )  # black for target
            sns.histplot(
                gen_counts.squeeze(),
                discrete=True,
                ax=ax[0, n],
                element="step",
                **args_gen,
            )  # darkred for sample

            ax[0, n].set_xlabel(f"{dic[n]} multiplicities")

        for n in [0, 1, 2, 3]:
            gen_counts = (self.gen.discrete == 4 + n).sum(dim=1)
            target_counts = (self.test.discrete == 4 + n).sum(dim=1)

            sns.histplot(
                target_counts.squeeze(),
                discrete=True,
                ax=ax[1, n],
                element="step",
                **args_tar,
            )  # black for target
            sns.histplot(
                gen_counts.squeeze(),
                discrete=True,
                ax=ax[1, n],
                element="step",
                **args_gen,
            )  # darkred for sample

            ax[1, n].set_xlabel(f"{dic[4 + n]} multiplicities")
            ax[1, n].set_xlim(0, 5)
            ax[1, n].set_ylim(0, 1.0)
            ax[1, n].set_xticks(np.arange(6))

        ax[0, 0].legend(loc="upper right", fontsize=6)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path / "flavor_multiplicities.png")
        else:
            plt.show()

    def jet_total_Q(self, label, save_path=None):
        args_gen = {
            "stat": "density",
            "log_scale": (False, False),
            "fill": False,
            "color": "darkred",
            "lw": 0.75,
            "label": "generated (t=1)",
        }
        args_tar = {
            "stat": "density",
            "log_scale": (False, False),
            "fill": True,
            "color": "k",
            "lw": 0.3,
            "alpha": 0.2,
            "label": label,
        }

        sample_total_charge = (self.gen.discrete == 2).sum(dim=1) - (
            self.gen.discrete == 3
        ).sum(dim=1)
        sample_total_charge += (self.gen.discrete == 4).sum(dim=1) - (
            self.gen.discrete == 5
        ).sum(dim=1)
        sample_total_charge += (self.gen.discrete == 6).sum(dim=1) - (
            self.gen.discrete == 7
        ).sum(dim=1)

        target_total_charge = (self.test.discrete == 2).sum(dim=1) - (
            self.test.discrete == 3
        ).sum(dim=1)
        target_total_charge += (self.test.discrete == 4).sum(dim=1) - (
            self.test.discrete == 5
        ).sum(dim=1)
        target_total_charge += (self.test.discrete == 6).sum(dim=1) - (
            self.test.discrete == 7
        ).sum(dim=1)

        _, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        sns.histplot(
            target_total_charge.squeeze(), discrete=True, element="step", **args_tar
        )
        sns.histplot(
            sample_total_charge.squeeze(), discrete=True, element="step", **args_gen
        )
        ax.legend(loc="upper right", fontsize=6)
        ax.set_xlabel(r"jet $Q_{\rm tot}$")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path / "jet_total_charge.png")
        else:
            plt.show()
