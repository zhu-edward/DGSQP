#!/usr/bin python3

from abc import ABC, abstractmethod
import pathlib
import shutil
import os

import casadi as ca

from DGSQP.dynamics.model_types import ModelConfig

class AbstractModel(ABC):
    '''
    Base class for models
    Controllers may differ widely in terms of algorithm runtime and setup however
      for interchangeability controllers should implement a set of standard runtime methods:
    '''
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

        self.code_gen = model_config.code_gen
        self.jit = model_config.jit

        if not model_config.enable_jacobians:
            jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        else:
            jac_opts = dict()

        if self.code_gen and not self.jit:
            self.c_file_name = self.model_config.model_name + '.c'
            self.so_file_name = self.model_config.model_name + '.so'
            self.generator = ca.CodeGenerator(self.c_file_name)
            self.options = lambda fn_name: dict(jit=False, **jac_opts)
        elif self.code_gen and self.jit:
            self.generator = None
            self.options = lambda fn_name: dict(jit=True,
                                                    jit_name=fn_name,
                                                    compiler='shell',
                                                    jit_options=dict(compiler='gcc', flags=['-%s' % self.model_config.opt_flag], verbose=self.model_config.verbose),
                                                    **jac_opts)
        else:
            self.generator = None
            self.options = lambda fn_name: dict(jit=False, **jac_opts)

    @abstractmethod
    def step(self):
        pass

    # Method for generating C code and building a shared object from it
    def build_shared_object(self, fns) -> pathlib.Path:
        for f in fns:
            self.generator.add(f)

        # Set up paths
        cur_dir = pathlib.Path.cwd()
        gen_path = cur_dir.joinpath(self.model_config.model_name)
        c_path = gen_path.joinpath(self.c_file_name)
        if gen_path.exists():
            shutil.rmtree(gen_path)
        gen_path.mkdir(parents=True)

        # Switch directory to gen_path and generate C code
        # (we do this because the CasADi CodeGenerator can't be initialized with a path)
        os.chdir(gen_path)
        if self.model_config.verbose:
            print('- Generating C code for model %s at %s' % (self.model_config.model_name, str(gen_path)))
        self.generator.generate()

        # Compile into shared object
        so_path = gen_path.joinpath(self.so_file_name)
        if self.model_config.verbose:
            print('- Compiling shared object %s from %s with optimization flag -%s' % (so_path, c_path, self.model_config.opt_flag))
        os.system('gcc -fPIC -shared -%s %s -o %s' % (self.model_config.opt_flag, c_path, so_path))

        # Swtich back to working directory
        os.chdir(cur_dir)

        # Install generated C code
        if self.model_config.install:
            install_path = self.install(verbose=self.model_config.verbose)
            return install_path.joinpath(self.so_file_name)
        else:
            return so_path

    # Method for installing generated files
    def install(self, dest_dir: str=None, src_dir: str=None, verbose=False):
        # If no target directory is provided, try to install a directory with
        # the same name as the model name in the current directory
        if src_dir is None:
            src_path = pathlib.Path.cwd().joinpath(self.model_config.model_name)
        else:
            src_path = pathlib.Path(src_dir).expanduser()

        if dest_dir is None:
            if self.model_config.install_dir is None:
                if verbose:
                    print('- No destination directory provided, did not install')
                return None
            dest_path = pathlib.Path(self.model_config.install_dir).expanduser()
        else:
            dest_path = pathlib.Path(dest_dir).expanduser()

        if src_path.exists():
            if not dest_path.exists():
                dest_path.mkdir(parents=True)
            # If directory with same name as model already exists, delete
            if dest_path.joinpath(self.model_config.model_name).exists():
                if verbose:
                    print('- Existing installation found, removing...')
                shutil.rmtree(dest_path.joinpath(self.model_config.model_name))
            shutil.move(str(src_path), str(dest_path))
            if verbose:
                print('- Installed files from source: %s to destination: %s' % (str(src_path), str(dest_path.joinpath(self.model_config.model_name))))
            return dest_path.joinpath(self.model_config.model_name)
        else:
            if verbose:
                print('- The source directory %s does not exist, did not install' % str(src_path))
            return None
