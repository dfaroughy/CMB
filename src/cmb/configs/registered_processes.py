from cmb.dynamics.processes import (TelegraphProcess, 
                                    FlowMatching, 
                                    SchrodingerBridge)

processes = {'continuous': 
                {'FlowMatching': FlowMatching,
                 'SchrodingerBridge': SchrodingerBridge},
             'discrete': 
                {'TelegraphProcess': TelegraphProcess}
             }