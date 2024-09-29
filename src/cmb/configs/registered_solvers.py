from cmb.dynamics.solvers import (EulerSolver, 
                                  TauLeapingSolver, 
                                  EulerMaruyamaSolver, 
                                  EulerLeapingSolver, 
                                  EulerMaruyamaLeapingSolver)

solvers = {'EulerSolver': EulerSolver,
           'EulerMaruyamaSolver': EulerMaruyamaSolver,
           'TauLeapingSolver': TauLeapingSolver,
           'EulerLeapingSolver': EulerLeapingSolver,
           'EulerMaruyamaLeapingSolver': EulerMaruyamaLeapingSolver}
