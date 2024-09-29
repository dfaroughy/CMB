from cmb.dynamics.cmb import (ConditionalMarkovBridge, 
                              BatchOTCMB, 
                              BatchRegOTCMB)

from cmb.dynamics.cfm import (ConditionalFlowMatching,
                              BatchOTCFM,
                              BatchRegOTCFM)

# from cmb.dynamics.cjb import ConditionalJumpBridge

dynamics = {'ConditionalMarkovBridge': ConditionalMarkovBridge,
            'ConditionalFlowMatching': ConditionalFlowMatching,
            # 'ConditionalJumpBridge': ConditionalJumpBridge,
            'BatchOTCMB': BatchOTCMB,
            'BatchOTCFM': BatchOTCFM,
            'BatchRegOTCMB': BatchRegOTCMB,
            'BatchRegOTCFM': BatchRegOTCFM
             }