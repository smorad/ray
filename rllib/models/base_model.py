# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model base classes.

RLlib models all inherit from the recurrent base class, which can be chained together
with other models.

The models input and output TensorDicts. Which keys the models read/write to
and the desired tensor shapes must be defined in input_spec, output_spec,
prev_state_spec, and next_state_spec.

The `unroll` function gets the model inputs and previous recurrent state, and outputs
the model outputs and next recurrent state. Note all ins/outs must match the specs.
Users should override `_unroll` rather than `unroll`.

`initial_state` returns the "next" state for the first recurrent iteration. Again,
users should override `_initial_state` instead.

For non-recurrent models, users may instead override `_forward` which does not
make use of recurrent states.
"""


import abc
from typing import Optional, Tuple
from ray.rllib.models.temp_spec_classes import TensorDict, SpecDict, ModelConfig


ForwardOutputType = TensorDict
# [Output, Recurrent State(s)]
UnrollOutputType = Tuple[TensorDict, TensorDict]


class RecurrentModel(abc.ABC):
    """The base model all other models are based on.

    Args:
        name: An optional name for the module

    Examples:
        >>> named_class = NamedClass() # Inherits from RecurrentModel
        >>> named_class.get_name() # NamedClass
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Returns the name of this module."""
        return self._name

    @property
    @abc.abstractmethod
    def input_spec(self) -> SpecDict:
        """Returns the spec of the input of this module."""

    @property
    @abc.abstractmethod
    def prev_state_spec(self) -> SpecDict:
        """Returns the spec of the prev_state of this module."""

    @property
    @abc.abstractmethod
    def output_spec(self) -> SpecDict:
        """Returns the spec of the output of this module."""

    @property
    @abc.abstractmethod
    def next_state_spec(self) -> SpecDict:
        """Returns the spec of the next_state of this module."""

    @abc.abstractmethod
    def _initial_state(self) -> TensorDict:
        """Initial state of the component.
        If this component returns a next_state in its unroll function, then
        this function provides the initial state.
        Subclasses should override this function instead of `initial_state`, which
        adds additional checks.

        Returns:
            A TensorDict containing the state before the first step.
        """

    def initial_state(self) -> TensorDict:
        """Initial state of the component.
        If this component returns a next_state in its unroll function, then
        this function provides the initial state.

        Returns:
            A TensorDict containing the state before the first step.

        Examples:
            >>> state = model.initial_state()
            >>> state # TensorDict(...)
        """
        initial_state = self._initial_state()
        self.next_state_spec.validate(initial_state)
        return initial_state

    @abc.abstractmethod
    def _unroll(
        self, inputs: TensorDict, prev_state: TensorDict, **kwargs
    ) -> UnrollOutputType:
        """Computes the output of the module over unroll_len timesteps.
        Subclasses should override this function instead of `unroll`, which
        adds additional checks.

        Args:
            inputs: A TensorDict of inputs
            prev_state: A TensorDict containing the next_state of the last
                timestep of the previous unroll.

        Returns:
            outputs: A TensorDict of outputs
            next_state: A dict containing the state to be passed
                as the first state of the next rollout.
        """

    def unroll(
        self, inputs: TensorDict, prev_state: TensorDict, **kwargs
    ) -> UnrollOutputType:
        """Computes the output of the module over unroll_len timesteps.

        Args:
            inputs: A TensorDict containing inputs to the model
            prev_state: A TensorDict containing containing the
                next_state of the last timestep of the previous unroll.

        Returns:
            outputs: A TensorDict containing model outputs
            next_state: A TensorDict containing the
                state to be passed as the first state of the next rollout.

        Examples:
            >>> output, state = model.unroll(TensorDict(...), TensorDict(...))
            >>> output # TensorDict(...)
            >>> state # TensorDict(...)

        """
        self.input_spec.validate(inputs)
        self.prev_state_spec.validate(prev_state)
        # We hide inputs not specified in input_spec to prevent accidental use.
        inputs = inputs.filter(self.input_spec)
        prev_state = prev_state.filter(self.prev_state_spec)
        inputs, prev_state = self._check_inputs_and_prev_state(inputs, prev_state)
        outputs, next_state = self._unroll(inputs, prev_state, **kwargs)
        self.output_spec.validate(outputs)
        self.next_state_spec.validate(next_state)
        outputs, next_state = self._check_outputs_and_next_state(outputs, next_state)
        return outputs, next_state

    def _check_inputs_and_prev_state(
        self, inputs: TensorDict, prev_state: TensorDict
    ) -> Tuple[TensorDict, TensorDict]:
        """Override this function to add additional checks on inputs.

        Args:
            inputs: TensorDict containing inputs to the model
            prev_state: The previous recurrent state

        Returns:
            inputs: Potentially modified inputs
            prev_state: Potentially modified recurrent state
        """
        return inputs, prev_state

    def _check_outputs_and_next_state(
        self, outputs: TensorDict, next_state: TensorDict
    ) -> Tuple[TensorDict, TensorDict]:
        """Override this function to add additional checks on outputs.

        Args:
            outputs: TensorDict output by the model
            next_state: Recurrent state output by the model

        Returns:
            outputs: Potentially modified TensorDict output by the model
            next_state: Potentially modified recurrent state output by the model
        """
        return outputs, next_state


class Model(RecurrentModel):
    """A RecurrentModel made non-recurrent by ignoring
    the input/output states.

    Args:
        name: An optional name for the module

    Examples:
        >>> named_class = NamedClass() # Inherits from Model
        >>> named_class.get_name() # NamedClass
    """

    @property
    def prev_state_spec(self) -> SpecDict:
        return SpecDict()

    @property
    def next_state_spec(self) -> SpecDict:
        return SpecDict()

    def _initial_state(self) -> TensorDict:
        return TensorDict()

    def _check_inputs_and_prev_state(
        self, inputs: TensorDict, prev_state: TensorDict
    ) -> Tuple[TensorDict, TensorDict]:
        inputs = self._check_inputs(inputs)
        return inputs, prev_state

    def _check_inputs(self, inputs: TensorDict) -> TensorDict:
        """Override this function to add additional checks on inputs.

        Args:
            inputs: TensorDict containing inputs to the model

        Returns:
            inputs: Potentially modified inputs
        """
        return inputs

    def _check_outputs_and_next_state(
        self, outputs: TensorDict, next_state: TensorDict
    ) -> Tuple[TensorDict, TensorDict]:
        outputs = self._check_outputs(outputs)
        return outputs, next_state

    def _check_outputs(self, outputs: TensorDict) -> TensorDict:
        """Override this function to add additional checks on outputs.

        Args:
            outputs: TensorDict output by the model

        Returns:
            outputs: Potentially modified TensorDict output by the model
        """
        return outputs

    def _unroll(
        self, inputs: TensorDict, prev_state: TensorDict, **kwargs
    ) -> UnrollOutputType:
        del prev_state
        outputs = self._forward(inputs, **kwargs)
        return outputs, TensorDict()

    @abc.abstractmethod
    def _forward(self, inputs: TensorDict, **kwargs) -> ForwardOutputType:
        """Computes the output of this module for each timestep.

        Args:
            inputs: A TensorDict containing model inputs

        Returns:
            outputs: A TensorDict containing model outputs

        Examples:
            # This is abstract, see the torch/tf/jax implementations
            >>> out = model._forward(TensorDict({"in": np.arange(10)}))
            >>> out # TensorDict(...)
        """


class ModelIO(abc.ABC):
    """Abstract class defining how to save and load model weights

    Args:
        config: The ModelConfig passed to the underlying model
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def config(self) -> ModelConfig:
        return self._config

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to a path

        Args:
            path: The path on disk where weights are to be saved

        Examples:
            model.save("/tmp/model_path.cpt")
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str) -> RecurrentModel:
        """Load model weights from a path

        Args:
            path: The path on disk where to load weights from

        Examples:
            model.load("/tmp/model_path.cpt")
        """
        raise NotImplementedError
