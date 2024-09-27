import numpy as np
import torch
import awkward as ak
import fastjet
import vector
import uproot

def extract_features(dataset, min_num=0, max_num=128, num_jets=None):
    if isinstance(dataset, list):
        all_data = []
        for data in dataset:
            assert  '.root' in data, 'Input should be a path to a .root file or a tensor'
            data = read_root_file(data)
            features = ['part_ptrel', 'part_etarel', 'part_phirel', 'part_isPhoton', 'part_isNeutralHadron', 'part_isChargedHadron', 'part_isElectron', 'part_isMuon', 'part_charge', 'part_isGhost']       
            data = torch.tensor(np.stack([ak.to_numpy(pad(data[feat], min_num=min_num, max_num=max_num)) for feat in features] , axis=1))
            data = torch.permute(data, (0,2,1))   
            continuous = data[...,0:3] 
            flavor = data[...,3:8].long()
            charge = data[...,8].long()
            mask = data[...,-1].long()
            discrete = flavor_representation(flavor, charge, rep='states') # (isPhoton, isNeutralHadron, isNegHadron, isPosHadron, isElectron, isAntiElectron, isMuon, isAntiMuon) 
            discrete = discrete.unsqueeze(-1) * mask.unsqueeze(-1)
            data = torch.cat([continuous, discrete.long(), mask.unsqueeze(-1)], dim=-1)
            all_data.append(data)   
        data = torch.cat(all_data, dim=0)   
        data = data[:num_jets] if num_jets is not None else data
    else:
        assert isinstance(dataset, torch.Tensor), 'Input should be a path to a .root file or a tensor'
        data = dataset[:num_jets] if num_jets is not None else dataset
        mask = data[...,-1].unsqueeze(-1)
        data *= mask
    return data


def read_root_file(filepath):
    
    """Loads a single .root file from the JetClass dataset.
    """
    x = uproot.open(filepath)['tree'].arrays()
    x['part_pt'] = np.hypot(x['part_px'], x['part_py'])
    x['part_pt_log'] = np.log(x['part_pt'])
    x['part_ptrel'] = x['part_pt'] / x['jet_pt']
    x['part_deltaR'] = np.hypot(x['part_deta'], x['part_dphi'])

    p4 = vector.zip({'px': x['part_px'],
                        'py': x['part_py'],
                        'pz': x['part_pz'],
                        'energy': x['part_energy']})

    x['part_eta'] = p4.eta
    x['part_phi'] = p4.phi
    x['part_etarel'] = p4.eta - x['jet_eta'] 
    x['part_phirel'] = (p4.phi - x['jet_phi'] + np.pi) % (2 * np.pi) - np.pi
    x['part_isGhost'] = np.ones_like(x['part_energy']) 
    return x


def flavor_representation(flavor_tensor, charge_tensor, rep='states'):
    ''' inputs: 
            - flavor in one-hot (isPhoton, isNeutralHadron, isChargedHadron, isElectron, isMuon)
            - charge (-1, 0, +1)
        outputs: 
            - 8-dim discrete feature vector (isPhoton:1, isNeutralHadron:2, isNegHadron:3, isPosHadron:4, isElectron:5, isAntiElectron:6, isMuon:7, isAntiMuon:8)  

    '''
    neutrals = flavor_tensor[...,:2].clone()
    charged = flavor_tensor[...,2:].clone() * charge_tensor.unsqueeze(-1)
    charged = charged.repeat_interleave(2, dim=-1)
    for idx in [0,2,4]:
        pos = charged[..., idx] == 1
        neg = charged[..., idx+1] == -1
        charged[..., idx][pos]=0
        charged[..., idx + 1][neg]=0
        charged[..., idx][neg]=1
    one_hot = torch.cat([neutrals, charged], dim=-1)
    if rep=='one-hot': 
        return one_hot.long()
    elif rep=='states': 
        state = torch.argmax(one_hot, dim=-1)
        state += 1
        return state
    
    
def pad(a, min_num, max_num, value=0, dtype='float32'):
    assert max_num >= min_num, 'max_num must be >= min_num'
    assert isinstance(a, ak.Array), 'Input must be an awkward array'
    a = a[ak.num(a) >= min_num]
    a = ak.fill_none(ak.pad_none(a, max_num, clip=True), value)
    return ak.values_astype(a, dtype)
