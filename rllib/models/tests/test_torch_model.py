import unittest
import os
import tempfile
import torch
from torch import nn

from ray.rllib.utils.test_utils import check
from ray.rllib.utils.annotations import override
from ray.rllib.models.temp_spec_classes import TensorDict, ModelConfig
from ray.rllib.models.specs.specs_dict import ModelSpecDict
from ray.rllib.models.specs.specs_torch import TorchSpecs
from ray.rllib.models.torch.model import TorchRecurrentModel, TorchModel

B, T = 6, 8


class SimpleRecurrentModel(TorchRecurrentModel):
    @property
    @override(TorchRecurrentModel)
    def input_spec(self) -> ModelSpecDict:
        return ModelSpecDict({"in": TorchSpecs("b, t, h", h=2)})

    @property
    @override(TorchRecurrentModel)
    def output_spec(self) -> ModelSpecDict:
        return ModelSpecDict({"out": TorchSpecs("b, t, h", h=3)})

    @property
    @override(TorchRecurrentModel)
    def prev_state_spec(self) -> ModelSpecDict:
        return ModelSpecDict({"in": TorchSpecs("b, h", h=4)})

    @property
    @override(TorchRecurrentModel)
    def next_state_spec(self) -> ModelSpecDict:
        return ModelSpecDict({"out": TorchSpecs("b, h", h=5)})

    @override(TorchRecurrentModel)
    def _unroll(self, input, prev_state):
        assert torch.all(input["in"] == torch.arange(B * T * 2).reshape(B, T, 2))
        assert torch.all(prev_state["in"] == torch.arange(B * 4).reshape(B, 4))
        assert "bork" not in input.keys()
        assert "bork" not in prev_state.keys()

        return (
            TensorDict({"out": torch.arange(B * T * 3).reshape(B, T, 3)}),
            TensorDict({"out": torch.arange(B * 5).reshape(B, 5)}),
        )


class SimpleModel(TorchModel):
    @property
    @override(TorchRecurrentModel)
    def input_spec(self) -> ModelSpecDict:
        return ModelSpecDict({"in": TorchSpecs("b, h", h=2)})

    @property
    @override(TorchRecurrentModel)
    def output_spec(self) -> ModelSpecDict:
        return ModelSpecDict({"out": TorchSpecs("b, h", h=3)})

    @override(TorchModel)
    def _forward(self, input):
        assert torch.all(input["in"] == torch.arange(B * 2).reshape(B, 2))
        return TensorDict({"out": torch.arange(B * 3).reshape(B, 3)})


class IOTorchModel(SimpleModel):
    def __init__(self, value):
        super().__init__(config=ModelConfig())
        self.weights = nn.Parameter(torch.tensor([value]))

    @override(SimpleModel)
    def _forward(self, input):
        pass


class TestTorchModel(unittest.TestCase):
    def test_save_load(self):
        """Test saving/restoring model weights"""
        with tempfile.TemporaryDirectory("test_torch_model.cpt") as d:
            path = os.path.join(d, "bork")
            m = IOTorchModel(value=1.0)
            m.save(path)
            lo = IOTorchModel(value=2.0)
            lo.load(path)
            check(m.weights, lo.weights)

    def test_recurrent_init(self):
        SimpleRecurrentModel(config=ModelConfig())

    def test_recurrent_unroll_and_filter(self):
        """Ensures unused inputs are filtered out before _unroll and that
        outputs are correct."""
        inputs = TensorDict(
            {
                "in": torch.arange(B * T * 2).reshape(B, T, 2),
                "bork": torch.arange(5 * 4).reshape(5, 4),
            }
        )
        states = TensorDict(
            {
                "in": torch.arange(B * 4).reshape(B, 4),
                "bork": torch.arange(5 * 4).reshape(5, 4),
            }
        )
        outputs, out_states = SimpleRecurrentModel(ModelConfig()).unroll(inputs, states)
        desired = TensorDict({"out": torch.arange(B * T * 3).reshape(B, T, 3)})
        desired_states = TensorDict({"out": torch.arange(B * 5).reshape(B, 5)})

        for k in outputs.shallow_keys() | desired.shallow_keys():
            check(outputs[k], desired[k])

        for k in out_states.shallow_keys() | desired_states.shallow_keys():
            check(out_states[k], desired_states[k])

    def test_model_init(self):
        SimpleModel(config=ModelConfig())

    def test_model_fwd_and_filter(self):
        """Ensures unused inputs are filtered out before forward and that
        outputs are correct."""
        inputs = TensorDict(
            {
                "in": torch.arange(B * 2).reshape(B, 2),
                "bork": torch.arange(5 * 4).reshape(5, 4),
            }
        )
        outputs, _ = SimpleModel(ModelConfig()).unroll(inputs, TensorDict())
        desired = TensorDict({"out": torch.arange(B * 3).reshape(B, 3)})

        for k in outputs.shallow_keys() | desired.shallow_keys():
            check(outputs[k], desired[k])


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
