from cmb.dynamics.processes import (TelegraphProcess, 
                                    FlowMatching, 
                                    SchrodingerBridge)

from cmb.dynamics.solvers import (EulerSolver, 
                                  TauLeapingSolver, 
                                  EulerMaruyamaSolver, 
                                  EulerLeapingSolver, 
                                  EulerMaruyamaLeapingSolver)

from cmb.models.architectures.epic import EPiC, HybridEPiC

processes = {
             'continuous': {'FlowMatching': FlowMatching,
                            'SchrodingerBridge': SchrodingerBridge},
             'discrete': {'TelegraphProcess': TelegraphProcess}
             }

solvers = {'EulerSolver': EulerSolver,
           'EulerMaruyamaSolver': EulerMaruyamaSolver,
           'TauLeapingSolver': TauLeapingSolver,
           'EulerLeapingSolver': EulerLeapingSolver,
           'EulerMaruyamaLeapingSolver': EulerMaruyamaLeapingSolver}

models = {'HybridEPiC': HybridEPiC,}