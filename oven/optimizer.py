import os
from collections import defaultdict
from pprint import pformat
from typing import Callable

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from scipy.optimize import minimize
from tqdm import tqdm
import logging

def select_optimizer(cls, x0, optimizer_type : str, bounds = None, nlp_options : dict = None, use_jax : bool = True, logger = None):

    methods = {func : getattr(cls, func) for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("__")}

    if optimizer_type == "sgd":
        _cls = type(cls.__name__, (SGD, ), methods)
        nlp = _cls(nlp_options, logger) # initialize the class
    
    elif optimizer_type == "scipy":
        ainst = cls()
        nlp = ScipyMinimize(
            fun = ainst.objective, # callable,
            jac = ainst.gradient if use_jax else None, # callable
            hess = ainst.hessian if use_jax else None, #  callable
            constraints = ainst.constraints(), # dictionary
            bounds = bounds, # list
            tol = nlp_options.pop("tol", 1e-6),
            method = ainst.method(),
            options = nlp_options # ipopt options
        )
    else :
        assert False, f"Optimizer {optimizer_type} not defined"
        
    return nlp.solve(x0)

"""
class TqdmLoggingHandler(logging.Handler):
    # https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634
    def __init__(self, level = logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try : 
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record) 
"""


class SGD:
    # stochastic gradient descent algorithm (adam optimizer)

    def __init__(self, nlp_options, logger):
        self.nlp_options = nlp_options
        # logger.addHandler(TqdmLoggingHandler())
        self.logger = logger
        self.divider = "--"*50
        self.results = defaultdict(list)
        self.logger.info(f"{self.divider} \nNLP Options {pformat(self.nlp_options)}")

    def solve(self, x0):
        
        epochs = self.nlp_options.pop("epochs", 100)
        print_every_epoch = self.nlp_options.pop("print_every_epoch", 10)
        opt_state = self.nlp_options.pop("opt_state", None)
        aux_args = self.nlp_options.pop("aux_args", None)
        learning_rate = self.nlp_options.pop("step_size")

        _opt_init, _opt_update, _get_params = optimizers.adam(learning_rate, **self.nlp_options)
        if opt_state is None:
            opt_state = _opt_init(x0)
        
        progress_bar = tqdm(range(epochs), file = open(os.devnull, "w"))
        self.logger.info(f"{self.divider} \nStarted Optimization")
        for epoch in progress_bar:
            aux_args = self.update_args(aux_args)
            x = _get_params(opt_state)
            training_loss = self.objective(x, aux_args)
            gradients = self.gradient(x, aux_args)
            opt_state =  _opt_update(epoch, gradients, opt_state)
            
            # store losses 
            testing_loss = self.intermediate(x, epoch, aux_args)
            self.results["training_loss"].append(training_loss)
            self.results["testing_loss"].append(testing_loss)
            self.results["learning_rate"].append(learning_rate(epoch) if isinstance(learning_rate, Callable) else learning_rate)

            if epoch % print_every_epoch == 0 :
                if self.logger is not None : 
                    self.logger.info(str(progress_bar))
                    self.logger.info(f"epoch {epoch}, training loss {training_loss}, gradients norm {jax.tree_util.tree_map(jnp.linalg.norm, gradients)}, testing loss {testing_loss}")
                else :
                    print(str(progress_bar))
                    print(f"epoch {epoch}, training loss {training_loss}, gradients norm {jax.tree_util.tree_map(jnp.linalg.norm, gradients)}, testing loss {testing_loss}")

        self.logger.info(f"Finished Optimization \n {self.divider}")
        return x, {"opt_state" : opt_state, "result" : self.results}


class ScipyMinimize:

    def __init__(self, **kwargs):
        self.minimize = lambda x0 : minimize(x0 = x0, **kwargs)

    def solve(self, x0):
        res = self.minimize(x0)
        return res.x, res
