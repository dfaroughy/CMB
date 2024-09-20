import numpy as np
import awkward as ak
import fastjet
import vector
import uproot
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

class Tokens:       
    def __init__(self, data, max_num_particles=128):
        self.tokens = self.zeropad(data, max_num_particles)
        self.mask = self.tokens > 0
        self.shape = self.tokens.shape

    def histogram(self, features='pt', num_bins=100, density=True, use_quantiles=False):
            mask = self.mask > 0
            x = getattr(self, features)[mask]
            bins = np.quantile(x, np.linspace(0.001, 0.999, num_bins)) if use_quantiles else num_bins
            return np.histogram(x, density=density, bins=bins)[0]

    def KLmetric1D(self, feature, reference, num_bins=100, use_quantiles=True):
        h1 = self.histogram(feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles) + 1e-8
        h2 = reference.constituents.histogram(feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles) + 1e-8
        return scipy.stats.entropy(h1, h2)  

    def Wasserstein1D(self, feature, reference):
        mask = self.mask > 0
        x = getattr(self, feature)[mask]
        y = getattr(reference.constituents, feature)[mask]
        return scipy.stats.wasserstein_distance(x, y)

    def histplot(self, features='tokens', xlim=None, ylim=None, xlabel=None, ylabel=None, idx=None, figsize=(3,3), ax=None, **kwargs):
        mask = self.mask > 0
        if ax is None: _, ax = plt.subplots(figsize=figsize)   
        if isinstance(features, tuple):
            x = getattr(self, features[0])[mask] if idx is None else getattr(self, features[0])[:,idx]
            y = getattr(self, features[1])[mask] if idx is None else getattr(self, features[1])[:,idx]
            sns.histplot(x=x, y=y, ax=ax, **kwargs)
            ax.set_xlabel(features[0] if xlabel is None else xlabel)
            ax.set_ylabel(features[1] if ylabel is None else ylabel)
        else:
            x = getattr(self, features)[mask] if idx is None else getattr(self, features)[:,idx]
            sns.histplot(x=x, element="step", ax=ax, **kwargs) 
            ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def zeropad(self, x, maxlen, pad_val=0, dtype='int64'):
        jets = []
        for i in x:
            pad = pad_val * np.ones(maxlen - len(i[1:-1]))
            jet = np.concatenate([i[1:-1], pad.astype(int)])
            jets.append(jet)
        return np.array(jets, dtype=dtype)
        

class JetTokens:       
    def __init__(self, filepath, max_num_particles=128):
        data = ak.from_parquet(filepath)
        self.constituents = Tokens(data,  max_num_particles=max_num_particles)
        self.shape = self.constituents.shape
        self.num_constituents = np.sum(self.constituents.mask, axis=1)
        self.token_rank = np.argsort(self.constituents)[::-1]

    def token_co_occurence(self, num_jets=1000):
        vocab_size = max(max(doc) for doc in self.constituents.tokens[:num_jets]) + 1
        co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        for jet in self.constituents.tokens[:num_jets]:
            non_zero_tokens = [tk for tk in jet if tk != 0]
            for i in range(len(non_zero_tokens)):
                for j in range(i + 1, len(non_zero_tokens)):
                    token_i = non_zero_tokens[i]
                    token_j = non_zero_tokens[j]
                    co_occurrence_matrix[token_i, token_j] += 1
        return co_occurrence_matrix + co_occurrence_matrix.T

    def histogram(self, features='pt', density=True, num_bins=100, use_quantiles=False):
        x = getattr(self, features)
        bins = np.quantile(x, np.linspace(0.001, 0.999, num_bins)) if use_quantiles else num_bins
        return np.histogram(x, density=density, bins=bins)[0]

    def KLmetric1D(self, feature, reference, num_bins=100, use_quantiles=True):
        h1 = self.histogram(feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles) + 1e-8
        h2 = reference.histogram(feature, density=True, num_bins=num_bins, use_quantiles=use_quantiles) + 1e-8
        return scipy.stats.entropy(h1, h2)  

    def Wassertein1D(self, feature, reference):
        x = getattr(self, feature)
        y = getattr(reference, feature)
        return scipy.stats.wasserstein_distance(x, y)

    def histplot(self, features='pt', xlim=None, ylim=None, xlabel=None, ylabel=None, figsize=(3,3), ax=None, **kwargs):
        if ax is None: _, ax = plt.subplots(figsize=figsize)   
        if isinstance(features, tuple):
            x, y = getattr(self, features[0]), getattr(self, features[1])
            sns.histplot(x=x, y=y, ax=ax, **kwargs)
            ax.set_xlabel(features[0] if xlabel is None else xlabel)
            ax.set_ylabel(features[1] if ylabel is None else ylabel)
        else:
            x = getattr(self, features)
            sns.histplot(x=x, element="step", ax=ax, **kwargs) 
            ax.set_xlabel(features if xlabel is None else xlabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)