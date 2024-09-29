
from cmb.models.architectures.deep_nets import (MLP, 
                                                HybridMLP, 
                                                ClassifierMLP)

from cmb.models.architectures.epic import (EPiC, 
                                           HybridEPiC)

# from cmb.models.architectures.transformer import (ParticleTransformer, 
#                                                   HybridParticleTransformer)

models = {'MLP': MLP,
          'ClassifierMLP': ClassifierMLP,
          'HybridMLP': HybridMLP,
          'EPiC': EPiC,
          'HybridEPiC': HybridEPiC,
        #   'ParticleTransformer': ParticleTransformer,
        #   'HybridParticleTransformer': HybridParticleTransformer
          }