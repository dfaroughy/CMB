import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from cmb.models.trainers import GenerativeDynamicsModule
from cmb.datasets.jetclass import JetDataclass, ParticleClouds

#######################################################################################################################################################
path = "/home/df630/CMB/results/runs/aoj_generation_CMB_final/GaussNoise_to_AspenOpenJets_FlowMatching_TelegraphProcess_HybridEPiC_2024.11.09_19h15_6392"
#######################################################################################################################################################

cmb = GenerativeDynamicsModule(config=path + "/config.yaml", device="cuda:0")
cmb.load(checkpoint="best")

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

num_jets = 10000
cmb.config.pipeline.num_timesteps = 1000
cmb.config.pipeline.time_eps = 0.001

#####


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_1.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_2.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_3.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_4.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_5.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_6.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_7.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_8.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_9.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_10.pt",
)

################################################################################
################################################################################

test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_11.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_12.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_13.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_14.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_15.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_16.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_17.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_18.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_19.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_20.pt",
)


################################################################################
################################################################################


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_21.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_22.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_23.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_24.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_25.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_26.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_27.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_28.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_29.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_30.pt",
)


################################################################################
################################################################################


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_31.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_32.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_33.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_34.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_35.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_36.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_37.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_38.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_39.pt",
)


test = JetDataclass(cmb.config.data, task="test", num_jets=num_jets)
test.source.preprocess(
    output_continuous=cmb.config.data.preprocess.continuous,
    output_discrete=cmb.config.data.preprocess.discrete,
)

cmb.generate(
    source_continuous=test.source.continuous,
    source_discrete=test.source.discrete,
    mask=test.source.mask,
    dataclass=ParticleClouds,
    output_history=False,
    save_to=path + "/generated_sample_40.pt",
)
