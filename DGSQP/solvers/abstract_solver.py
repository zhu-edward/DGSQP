#!/usr/bin python3

from abc import ABC, abstractmethod
import pathlib
import shutil

from DGSQP.types import VehicleState, VehiclePrediction

class AbstractSolver(ABC):
    '''
    Base class for solvers
    Controllers may differ widely in terms of algorithm runtime and setup however
      for interchangeability controllers should implement a set of standard runtime methods:
    '''

    needs_env_state = False

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def solve(self):
        pass


    @abstractmethod
    def step(self, estimated_state: VehicleState, env_state):
        '''
        Update the controller on the estimated state of the vehicle and its environment.
        solve() may look different for different controllers but this method must be identical

        The controller should modify the control fields of estimated_state directly and return a status that indicates any errors of the controller.
        '''

        #TODO: Can implement basic common controller function here, or in a higher level class
        # these would be things like checking if a controller is initialized, ready, and error-free, and choosing / switching between various controllers
        return


    #These are methods that return empty data structures by default and should be overriden by controllers that implement them.
    def get_prediction(self):
        return VehiclePrediction()

    # TODO: change to get_safe_set
    def get_ss(self):
        return VehiclePrediction()

    # Method for installing generated solver files
    def install(self, path='~/.mpclab_controllers/', verbose=False):
        src_path = pathlib.Path.cwd().joinpath(self.solver_name)
        if src_path.exists():
            dest_path = pathlib.Path(path).expanduser()
            if not dest_path.exists():
                dest_path.mkdir(parents=True)
            if dest_path.joinpath(self.solver_name).exists():
                if verbose:
                    print('- Existing installation found, removing...')
                shutil.rmtree(dest_path.joinpath(self.solver_name))
            shutil.move(str(src_path), str(dest_path))
            if verbose:
                print('- Installed files from source: %s to destination: %s' % (str(src_path), str(dest_path.joinpath(self.solver_name))))
            return dest_path.joinpath(self.solver_name)
        else:
            if verbose:
                print('- The source directory %s does not exist, did not install' % str(src_path))
            return None
