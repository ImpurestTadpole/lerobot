# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import abc
import builtins
import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.utils.hub import HubMixin

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


def _resolve_local_pretrained_path(pretrained_name_or_path: str | Path) -> Path:
    p = Path(pretrained_name_or_path).expanduser()
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _looks_like_filesystem_checkpoint_path(s: str) -> bool:
    """Heuristic: avoid sending local-style paths to Hugging Face Hub (invalid repo id)."""
    s_norm = s.replace("\\", "/").strip()
    if not s_norm:
        return False
    if s_norm.startswith(("/", "./", "../")):
        return True
    markers = (
        "pretrained_model",
        "outputs/train",
        "/train/",
        "/checkpoints/",
        "checkpoints/",
    )
    return any(m in s_norm for m in markers)


def _infer_policy_config_type(config: dict[str, Any]) -> str | None:
    """
    Infer draccus ChoiceRegistry ``type`` for JSON that omits it.

    ``PreTrainedConfig.type`` is a property (not a dumped field), so older checkpoints
    may lack the top-level ``type`` key required by ``draccus.parse``.
    """
    if "annotation_mode" in config and "clip_batch_size" in config:
        return "sarm"
    return None


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):  # type: ignore[misc,name-defined] #TODO: draccus issue
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
    """

    n_obs_steps: int = 1

    # `input_features` can be set to None/null in order to infer those values from the dataset.
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    # Whether the policy employed PEFT for training.
    use_peft: bool = False

    push_to_hub: bool = True  # type: ignore[assignment] # TODO: use a different name to avoid override
    repo_id: str | None = None

    # Upload on private repository on the Hugging Face hub.
    private: bool | None = None
    # Add tags to your policy on the hub.
    tags: list[str] | None = None
    # Add tags to your policy on the hub.
    license: str | None = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    pretrained_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg] #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        # ``type`` is required for ``from_pretrained`` / draccus ChoiceRegistry but is a
        # ``@property``, not a dataclass field, so draccus.dump omits it unless merged in.
        buf = io.StringIO()
        with draccus.config_type("json"):
            draccus.dump(self, buf, indent=4)
        data = json.loads(buf.getvalue())
        data["type"] = self.type
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        resolved_local = _resolve_local_pretrained_path(model_id)
        config_file: str | None = None
        if resolved_local.is_dir():
            if CONFIG_NAME in os.listdir(resolved_local):
                config_file = str(resolved_local / CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {resolved_local}")
        elif _looks_like_filesystem_checkpoint_path(model_id):
            raise FileNotFoundError(
                f"Local policy path is not an existing directory: {resolved_local}. "
                "If you meant a Hugging Face Hub repo, use 'namespace/repo_name'. "
                "Otherwise fix the path (checkpoint step folder may be missing, e.g. 008000 vs 005000), "
                "run from the repo root, or pass an absolute path to pretrained_model."
            )
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            config = json.load(f)

        if "type" not in config:
            inferred = _infer_policy_config_type(config)
            if inferred is not None:
                config["type"] = inferred
            else:
                raise ValueError(
                    f"Policy {CONFIG_NAME} at {config_file!r} has no top-level 'type' key and "
                    "could not infer the policy class. Add \"type\": \"<policy_name>\" (e.g. "
                    '"sarm") or re-save the checkpoint with a current lerobot version.'
                )

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            temp_for_choice = f.name

        # HACK: Parse the original config to get the config subclass, so that we can
        # apply cli overrides.
        # This is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        with draccus.config_type("json"):
            orig_config = draccus.parse(cls, temp_for_choice, args=[])

        # ChoiceRegistry dispatch requires top-level ``type``; concrete config dataclasses
        # (e.g. SARMConfig) do not define that field — strip before the second parse.
        config.pop("type", None)
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            temp_for_subclass = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(orig_config.__class__, temp_for_subclass, args=cli_overrides)