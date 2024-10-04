from cmb.dynamics.cmb import (ConditionalMarkovBridge, 
                              BatchOTCMB, 
                              BatchEntropicOTCMB)


# from cmb.dynamics.cjb import ConditionalJumpBridge

dynamics = {'ConditionalMarkovBridge': ConditionalMarkovBridge,
            'BatchOTCMB': BatchOTCMB,
            'BatchEntropicOTCMB': BatchEntropicOTCMB,
             }