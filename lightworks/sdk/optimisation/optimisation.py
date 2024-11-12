# Copyright 2024 Aegiq Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contains routines to perform optimisation with a parameterized circuit.
"""

import warnings
from numbers import Number
from types import FunctionType, NoneType
from typing import Any, Callable

import zoopt
from bayes_opt import BayesianOptimization
from scipy.optimize import basinhopping, minimize

from ...emulator import Detector, Sampler, Simulator, Source
from ...emulator.results import SamplingResult, SimulationResult
from ..circuit import Circuit, ParameterDict
from ..state import State


class Optimisation:
    """
    Provides a generic interface for performing optimisation of circuits
    according to a set figure of merit function.

    Args:

        opt_circuit (ParameterizedCircuit) : The parameterized circuit that is
            to be optimised.

        parameters (ParameterDict) : A parameter dictionary containing all of
            the parameters used in the circuit. If the parameters have bounds
            then these will be used as the bounds for the optimisation if
            supported.

        input_state (State) : The input photon configuration to use with the
            optimisation.

    """

    def __init__(
        self,
        opt_circuit: Circuit,
        parameters: ParameterDict,
        input_state: State,
    ) -> None:
        self.opt_circuit = opt_circuit
        self.parameters = parameters
        self.input_state = input_state

        # Check all provided parameters are included in the circuit
        circ_params = self.opt_circuit.get_all_params()
        all_in = True
        for k in parameters:
            if parameters[k] not in circ_params:
                all_in = False
        # If not then raise a warning - use warning instead of error as in
        # principle it should still be possible to optimise
        if not all_in:
            warnings.warn(  # noqa: B028
                "One or more of the parameters in the ParameterDict are not "
                "used in the optimisation circuit.",
                Warning,
            )

        return

    @property
    def opt_circuit(self) -> Circuit:
        """The circuit to be used for optimisation."""
        return self.__opt_circuit

    @opt_circuit.setter
    def opt_circuit(self, value: Circuit) -> None:
        if not isinstance(value, Circuit):
            raise TypeError("opt_circuit should be a Circuit.")
        self.__opt_circuit = value

    @property
    def parameters(self) -> ParameterDict:
        """The circuit to be used for optimisation."""
        return self.__parameters

    @parameters.setter
    def parameters(self, value: ParameterDict) -> None:
        if not isinstance(value, ParameterDict):
            raise TypeError("parameters should be a ParameterDict.")
        self.__parameters = value
        self.__x0 = [v for k, v in value.items()]

    @property
    def input_state(self) -> State:
        """The input state to use for the optimisation."""
        return self.__input_state

    @input_state.setter
    def input_state(self, value: Any) -> None:
        if not isinstance(value, State):
            raise TypeError(
                "Optimisation input should be a single State object."
            )
        self.__input_state = value

    def set_optimisation_func(
        self, function: Callable, args: list | None = None
    ) -> None:
        """
        Sets the function that should be used to calculate the figure of merit
        for the optimisation. This function should take a Result object and
        return a single real numerical value.

        Args:

            function (Callable) : The function to use as part of the
            optimisation.

            args (list | None) : For providing any additional arguments
                required as part of the optimisation function. This is only
                required if the function takes more than a single argument.

        """
        if not isinstance(function, FunctionType):
            raise TypeError(
                "optimisation function should be a callable function."
            )
        if function.__code__.co_argcount > 1:
            if args is None:
                raise ValueError("args not specified.")
            if len(args) != function.__code__.co_argcount - 1:
                raise ValueError(
                    "args has an incorrect number of items for the function."
                )
        else:
            args = []

        self.__opt_func = function
        self.__func_args = args
        # Check optimisation function is valid
        self._test_opt_func()

        return

    def set_processing_type(
        self, processor: str, processor_args: dict | None = None
    ) -> None:
        """
        Set the processor type for the optimisation and optional arguments that
        can be specified.

        Args:

            processor (str) : Define the processor type to use.

            processor_args (dict | None) : A dictionary of arguments to use
                with the chosen processor. If an argument is invalid an error
                will be raised.

        """
        if not isinstance(processor_args, (dict, NoneType)):
            raise TypeError("processor_args should be a dictionary.")
        if processor == "simulator":
            if processor_args is not None:
                raise ValueError(
                    "No processor arguments available when using simulator."
                )
            args = {}
        elif processor == "sampler":
            args = {
                "samples": 10000,
                "source": Source(),
                "detector": Detector(),
                "post_select": None,
                "min_detection": 0,
            }
            if processor_args is not None:
                for p, v in processor_args.items():
                    if p in args:
                        args[p] = v
                    else:
                        options = ", ".join(list(args.keys()))
                        msg = (
                            "Incorrect sampler argument entered, valid "
                            f"options are '{options}'."
                        )
                        raise ValueError(msg)
        else:
            raise ValueError("Processor type not valid.")
        self.__processor = processor
        self.__processor_args = args

        return

    def scipy_optimise(self, minimise: bool = True) -> dict:
        """
        Perform an optimisation using the scipy minimize routine.

        Args:

            minimise (bool, optional) : Sets whether the provided optimisation
                function should be minimised or maximised. Defaults to True
                (minimisation).

        Returns:

            dict : The determined optimal parameters using the keys from the
                provided parameter dictionary.

        """
        self._opt_checks()
        self.__invert_fom = not minimise
        bounds = None
        if self.parameters.has_bounds():
            bounds = list(self.parameters.get_bounds().values())
        res = minimize(self._fom, x0=self.__x0, bounds=bounds)
        self.__opt_results = dict(zip(self.parameters.keys(), res.x))
        return self.__opt_results

    def scipy_basinhopping_optimise(
        self, minimise: bool = True, n_iter: int = 20
    ) -> dict:
        """
        Perform an optimisation using the scipy basinhopping optimisation
        routine.

        Args:

            minimise (bool, optional) : Sets whether the provided optimisation
                function should be minimised or
                                        maximised. Defaults to True
                                        (minimisation).

            n_iter (int, optional) : Set the number of iterations to use in
                                     the optimisation, defaults to 20. Note
                                     that one iteration will consist of many
                                     function evaluations.

        Returns:

            dict : The determined optimal parameters using the keys from the
                   provided parameter dictionary.

        """
        self._opt_checks()
        self.__invert_fom = not minimise
        if self.parameters.has_bounds():
            raise ValueError(
                "Bounds not supported by basinhopping optimise method."
            )
        res = basinhopping(self._fom, x0=self.__x0, niter=n_iter)
        self.__opt_results = dict(zip(self.parameters.keys(), res.x))
        return self.__opt_results

    def bayesian_optimise(
        self,
        minimise: bool = True,
        display_progress: bool = False,
        init_points: int = 10,
        n_iter: int = 50,
    ) -> dict:
        """
        Perform an optimisation using a Bayesian optimisation process.

        Args:

            minimise (bool, optional) : Sets whether the provided optimisation
                function should be minimised or maximised. Defaults to True
                (minimisation).

            display_progress (bool, optional) : Optionally includes a display
                of the optimisation progression. Defaults to False.

            init_points (int, optional) : Set the number of initial points to
                use as part of the optimisation. Defaults to 10.

            n_iter (int, optional) : Set the total number of iterations to use
                within the Bayesian optimisation. Defaults to 50.

        Returns:

            dict : The determined optimal parameters using the keys from the
                provided parameter dictionary.

        """
        self._opt_checks()
        self.__invert_fom = minimise
        pbounds = self.parameters.get_bounds()
        verbose = 2 if display_progress else 0
        optimizer = BayesianOptimization(
            self._kwargs_fom,
            pbounds=pbounds,
            allow_duplicate_points=True,
            verbose=verbose,
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        best = optimizer.max
        self.__opt_results = best["params"]  # type:ignore
        return best["params"]  # type:ignore

    def zero_order_optimise(
        self, minimise: bool = True, budget: int = 200
    ) -> dict:
        """
        Performs an optimisation using the Zeroth-Order Optimization (ZOOpt)
        python package, intended for large and noisy parameter spaces,

        Args:

            minimise (bool, optional) : Sets whether the provided optimisation
                function should be minimised or maximised. Defaults to True
                (minimisation).

            budget (int, optional) : Sets the number of figure of merit
                evaluations that the optimiser is allowed to make.

        Returns:

            dict : The determined optimal parameters using the keys from the
                provided parameter dictionary.

        """
        self.__invert_fom = not minimise
        dim = zoopt.Dimension(
            len(self.parameters),
            list(self.parameters.get_bounds().values()),
            [True] * len(self.parameters),
        )
        objective = zoopt.Objective(self._fom_zoopt, dim)
        parameter = zoopt.Parameter(budget=budget)
        sol = zoopt.Opt.min(objective, parameter)
        self.__opt_results = dict(zip(self.parameters.keys(), sol.get_x()))
        return self.__opt_results

    def get_optimal_circuit(self) -> Circuit:
        """Creates a circuit using the optimal parameters and returns it."""
        for p in self.parameters:
            self.parameters[p] = self.__opt_results[p]
        return self.opt_circuit

    def test_optimal_circuit(self) -> SamplingResult | SimulationResult:
        """
        Finds the results produced using an optimal circuit and returns
        this as Result object.
        """
        for p in self.parameters:
            self.parameters[p] = self.__opt_results[p]
        return self._process()

    def _opt_checks(self) -> None:
        """
        Checks that all required components have been configured before an
        optimisation takes place.
        """
        m1 = " should be defined with "
        m2 = " method before optimisation takes place."
        requirements = [
            ("__opt_func", "Optimisation function", "set_optimisation_func"),
            ("__processor", "Processor", "set_processing_type"),
        ]
        pre_attr = "_Optimisation"
        for attr, name, func in requirements:
            if not hasattr(self, pre_attr + attr):
                raise RuntimeError(name + m1 + func + m2)

    def _fom(self, params: list) -> float:
        """Takes a list of parameters and calculates the fom."""
        for i, p in enumerate(self.parameters.keys()):
            self.parameters[p] = params[i]
        # Get results using selected processor
        results = self._process()
        # Find fom using user provided function and return
        fom = self.__opt_func(results, *self.__func_args)
        return -fom if self.__invert_fom else fom

    def _fom_zoopt(self, params: Any) -> float:
        """Takes a parameter object and calculates the fom."""
        params = params.get_x()
        for i, p in enumerate(self.parameters.keys()):
            self.parameters[p] = params[i]
        # Get results using selected processor
        results = self._process()
        # Find fom using user provided function and return
        fom = self.__opt_func(results, *self.__func_args)
        return -fom if self.__invert_fom else fom

    def _kwargs_fom(self, **kwargs: Any) -> float:
        """Takes optimisation parameters as kwargs and finds fom."""
        for p in self.parameters:
            self.parameters[p] = kwargs[p]
        # Get results using selected processor
        results = self._process()
        # Find fom using user provided function and return
        fom = self.__opt_func(results, *self.__func_args)
        return -fom if self.__invert_fom else fom

    def _process(self) -> SamplingResult | SimulationResult:
        """Get the results from the chosen processor type."""
        if self.__processor == "sampler":
            sampler = Sampler(
                self.opt_circuit,
                self.input_state,
                source=self.__processor_args["source"],  # type: ignore
                detector=self.__processor_args["detector"],  # type: ignore
            )
            return sampler.sample_N_inputs(
                self.__processor_args["samples"],  # type: ignore
                post_select=self.__processor_args["post_select"],  # type: ignore
                min_detection=(
                    self.__processor_args["min_detection"]  # type: ignore
                ),
            )
        if self.__processor == "simulator":
            simulator = Simulator(self.opt_circuit)
            return simulator.simulate(self.input_state)
        raise ValueError("Processor type not recognised.")

    def _test_opt_func(self) -> None:
        """
        Confirms a provided optimisation function should work with a test
        input and returns a numeric output.
        """
        if not hasattr(self, "_Optimisation__opt_func"):
            raise RuntimeError(
                "Optimisation function needs to be specified first."
            )
        n = self.opt_circuit.input_modes
        results = SamplingResult(
            {State([0] * n): 0.5, State([1] + [0] * (n - 1)): 0.5},
            input=self.input_state,
        )
        try:
            return_val = self.__opt_func(results, *self.__func_args)
        except Exception as e:
            raise RuntimeError(
                "An error occurred while trying to test the provided "
                "optimisation function."
            ) from e
        if not isinstance(return_val, Number):
            raise TypeError("Function return value should be a number.")
