import torch
from torch import nn
import tree

from ray.rllib.utils.annotations import (
    DeveloperAPI,
    override,
)
from ray.rllib.models.temp_spec_classes import TensorDict, ModelConfig
from ray.rllib.models.base_model import RecurrentModel, Model, ModelIO


class TorchModelIO(ModelIO):
    """Save/Load mixin for torch models

    Examples:
        >>> model.save("/tmp/model_weights.cpt")
        >>> model.load("/tmp/model_weights.cpt")
    """

    @DeveloperAPI
    @override(ModelIO)
    def save(self, path: str) -> None:
        """Saves the state dict to the specified path

        Args:
            path: Path on disk the checkpoint is saved to

        """
        torch.save(self.state_dict(), path)

    @DeveloperAPI
    @override(ModelIO)
    def load(self, path: str) -> RecurrentModel:
        """Loads the state dict from the specified path

        Args:
            path: Path on disk to load the checkpoint from
        """
        self.load_state_dict(torch.load(path))


class TorchRecurrentModel(RecurrentModel, nn.Module, TorchModelIO):
    """The base class for recurrent pytorch models.

    If implementing a custom recurrent model, you likely want to inherit
    from this model.

    Args:
        config: The config used to construct the model

    Required Attributes:
        input_spec: SpecDict: Denotes the input keys and shapes passed to `unroll`
        output_spec: SpecDict: Denotes the output keys and shapes returned from
            `unroll`
        prev_state_spec: SpecDict: Denotes the keys and shapes for the input
            recurrent states to the model
        next_state_spec: SpecDict: Denotes the keys and shapes for the
            recurrent states output by the model

    Required Overrides:
        # Define unrolling (forward pass) over a sequence of inputs
        _unroll(self, inputs: TensorDict, prev_state: TensorDict, **kwargs)
            -> Tuple[TensorDict, TensorDict]

    Optional Overrides:
        # Define the initial state, if a zero tensor is insufficient
        # the returned TensorDict must match the prev_state_spec
        _initial_state(self) -> TensorDict

        # Save model weights to path
        save(self, path: str) -> None

        # Load model weights from path
        load(self, path: str) -> None

    Examples:
        >>> class MyCustomModel(TorchRecurrentModel):
        ...     def __init__(self, input_size, output_size, recurrent_size):
        ...         self.input_spec = SpecDict(
        ...             {"obs": "batch, time, hidden"}, hidden=input_size
        ...        )
        ...         self.output_spec = SpecDict(
        ...             {"logits": "batch, time, logits"}, logits=output_size
        ...         )
        ...         self.prev_state_spec = SpecDict(
        ...             {"input_state": "batch, recur"}, recur=recurrent_size
        ...         )
        ...         self.next_state_spec = SpecDict(
        ...             {"output_state": "batch, recur"}, recur=recurrent_size
        ...         )
        ...
        ...         self.lstm = nn.LSTM(
        ...             input_size, recurrent_size, batch_first=True
        ...         )
        ...         self.project = nn.Linear(recurrent_size, output_size)
        ...
        ...     def _unroll(self, inputs, prev_state, **kwargs):
        ...         output, state = self.lstm(inputs["obs"], prev_state["input_state"])
        ...         output = self.project(output)
        ...         return TensorDict(
        ...             {"logits": output}), TensorDict({"output_state": state}
        ...         )

    """

    def __init__(self, config: ModelConfig) -> None:
        RecurrentModel.__init__(self)
        nn.Module.__init__(self)
        TorchModelIO.__init__(self, config)

    @override(RecurrentModel)
    def _initial_state(self) -> TensorDict:
        """Returns the initial recurrent state

        This defaults to all zeros and can be overidden to return
        nonzero tensors.

        Returns:
            A TensorDict that matches the initial_state_spec
        """
        return TensorDict(
            tree.map_structure(
                lambda spec: torch.zeros(spec.shape, dtype=spec.dtype),
                self.initial_state_spec,
            )
        )


class TorchModel(Model, nn.Module, TorchModelIO):
    """The base class for non-recurrent pytorch models.

    If implementing a custom pytorch model, you likely want to
    inherit from this class.

    Args:
        config: The config used to construct the model

    Required Attributes:
        input_spec: SpecDict: Denotes the input keys and shapes passed to `_forward`
        output_spec: SpecDict: Denotes the output keys and shapes returned from
            `_forward`

    Required Overrides:
        # Define unrolling (forward pass) over a sequence of inputs
        _forward(self, inputs: TensorDict, **kwargs)
            -> TensorDict

    Optional Overrides:
        # Save model weights to path
        save(self, path: str) -> None

        # Load model weights from path
        load(self, path: str) -> None

    Examples:
        >>> class MyCustomModel(TorchModel):
        ...     def __init__(self, input_size, hidden_size, output_size):
        ...         self.input_spec = SpecDict(
        ...             {"obs": "batch, time, hidden"}, hidden=input_size
        ...         )
        ...         self.output_spec = SpecDict(
        ...         {"logits": "batch, time, logits"}, logits=output_size
        ...         )
        ...
        ...         self.mlp = nn.Sequential(
        ...             nn.Linear(input_size, hidden_size),
        ...             nn.ReLU(),
        ...             nn.Linear(hidden_size, hidden_size),
        ...             nn.ReLU(),
        ...             nn.Linear(hidden_size, output_size)
        ...         )
        ...
        ...     def _forward(self, inputs, **kwargs):
        ...         output = self.mlp(inputs["obs"])
        ...         return TensorDict({"logits": output})

    """

    def __init__(self, config: ModelConfig) -> None:
        Model.__init__(self)
        nn.Module.__init__(self)
        TorchModelIO.__init__(self, config)
