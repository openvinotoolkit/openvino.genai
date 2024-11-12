import openvino._pyopenvino
import os
from typing import Callable, ClassVar, Iterator, overload


class Adapter:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.Adapter) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.Adapter, path: os.PathLike) -> None


                    Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
                    path (str): Path to adapter file in safetensors format.

        """

    @overload
    def __init__(self, path: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.Adapter) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.Adapter, path: os.PathLike) -> None


                    Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
                    path (str): Path to adapter file in safetensors format.

        """

    def __bool__(self) -> bool:
        """__bool__(self: openvino_genai.py_openvino_genai.Adapter) -> bool"""


class AdapterConfig:
    class Mode:
        __members__: ClassVar[dict] = ...  # read-only
        MODE_AUTO: ClassVar[AdapterConfig.Mode] = ...
        MODE_DYNAMIC: ClassVar[AdapterConfig.Mode] = ...
        MODE_FUSE: ClassVar[AdapterConfig.Mode] = ...
        MODE_STATIC: ClassVar[AdapterConfig.Mode] = ...
        MODE_STATIC_RANK: ClassVar[AdapterConfig.Mode] = ...
        __entries: ClassVar[dict] = ...

        def __init__(self, value: int) -> None:
            """__init__(self: openvino_genai.py_openvino_genai.AdapterConfig.Mode, value: int) -> None"""

        def __eq__(self, other: object) -> bool:
            """__eq__(self: object, other: object) -> bool"""

        def __hash__(self) -> int:
            """__hash__(self: object) -> int"""

        def __index__(self) -> int:
            """__index__(self: openvino_genai.py_openvino_genai.AdapterConfig.Mode) -> int"""

        def __int__(self) -> int:
            """__int__(self: openvino_genai.py_openvino_genai.AdapterConfig.Mode) -> int"""

        def __ne__(self, other: object) -> bool:
            """__ne__(self: object, other: object) -> bool"""

        @property
        def name(self) -> str: ...

        @property
        def value(self) -> int: ...

    @overload
    def __init__(self, mode: AdapterConfig.Mode = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, alpha: float, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        3. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        4. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[openvino_genai.py_openvino_genai.Adapter], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        5. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[tuple[openvino_genai.py_openvino_genai.Adapter, float]], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None
        """

    @overload
    def __init__(self, adapter: Adapter, alpha: float, mode: AdapterConfig.Mode = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, alpha: float, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        3. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        4. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[openvino_genai.py_openvino_genai.Adapter], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        5. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[tuple[openvino_genai.py_openvino_genai.Adapter, float]], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None
        """

    @overload
    def __init__(self, adapter: Adapter, mode: AdapterConfig.Mode = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, alpha: float, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        3. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        4. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[openvino_genai.py_openvino_genai.Adapter], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        5. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[tuple[openvino_genai.py_openvino_genai.Adapter, float]], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None
        """

    @overload
    def __init__(self, adapters: list[Adapter], mode: AdapterConfig.Mode = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, alpha: float, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        3. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        4. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[openvino_genai.py_openvino_genai.Adapter], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        5. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[tuple[openvino_genai.py_openvino_genai.Adapter, float]], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None
        """

    @overload
    def __init__(self, adapters: list[tuple[Adapter, float]], mode: AdapterConfig.Mode = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, alpha: float, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        3. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        4. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[openvino_genai.py_openvino_genai.Adapter], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None

        5. __init__(self: openvino_genai.py_openvino_genai.AdapterConfig, adapters: list[tuple[openvino_genai.py_openvino_genai.Adapter, float]], mode: openvino_genai.py_openvino_genai.AdapterConfig.Mode = AdapterConfig.Mode.MODE_AUTO) -> None
        """

    @overload
    def add(self, adapter: Adapter, alpha: float) -> AdapterConfig:
        """add(*args, **kwargs)
        Overloaded function.

        1. add(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, alpha: float) -> openvino_genai.py_openvino_genai.AdapterConfig

        2. add(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter) -> openvino_genai.py_openvino_genai.AdapterConfig
        """

    @overload
    def add(self, adapter: Adapter) -> AdapterConfig:
        """add(*args, **kwargs)
        Overloaded function.

        1. add(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, alpha: float) -> openvino_genai.py_openvino_genai.AdapterConfig

        2. add(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter) -> openvino_genai.py_openvino_genai.AdapterConfig
        """

    def get_adapters(self) -> list[Adapter]:
        """get_adapters(self: openvino_genai.py_openvino_genai.AdapterConfig) -> list[openvino_genai.py_openvino_genai.Adapter]"""

    def get_alpha(self, adapter: Adapter) -> float:
        """get_alpha(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter) -> float"""

    def remove(self, adapter: Adapter) -> AdapterConfig:
        """remove(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter) -> openvino_genai.py_openvino_genai.AdapterConfig"""

    def set_alpha(self, adapter: Adapter, alpha: float) -> AdapterConfig:
        """set_alpha(self: openvino_genai.py_openvino_genai.AdapterConfig, adapter: openvino_genai.py_openvino_genai.Adapter, alpha: float) -> openvino_genai.py_openvino_genai.AdapterConfig"""

    def __bool__(self) -> bool:
        """__bool__(self: openvino_genai.py_openvino_genai.AdapterConfig) -> bool"""


class AggregationMode:
    __members__: ClassVar[dict] = ...  # read-only
    NORM_SUM: ClassVar[AggregationMode] = ...
    SUM: ClassVar[AggregationMode] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.AggregationMode, value: int) -> None"""

    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""

    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""

    def __index__(self) -> int:
        """__index__(self: openvino_genai.py_openvino_genai.AggregationMode) -> int"""

    def __int__(self) -> int:
        """__int__(self: openvino_genai.py_openvino_genai.AggregationMode) -> int"""

    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class AutoencoderKL:
    class Config:
        block_out_channels: list[int]
        in_channels: int
        latent_channels: int
        out_channels: int
        scaling_factor: float

        def __init__(self, config_path: os.PathLike) -> None:
            """__init__(self: openvino_genai.py_openvino_genai.AutoencoderKL.Config, config_path: os.PathLike) -> None"""

    @overload
    def __init__(self, vae_decoder_path: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.


        2. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.


        3. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        4. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        5. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, model: openvino_genai.py_openvino_genai.AutoencoderKL) -> None

        AutoencoderKL model
                    AutoencoderKL class.
                    model (AutoencoderKL): AutoencoderKL model.

        """

    @overload
    def __init__(self, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.


        2. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.


        3. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        4. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        5. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, model: openvino_genai.py_openvino_genai.AutoencoderKL) -> None

        AutoencoderKL model
                    AutoencoderKL class.
                    model (AutoencoderKL): AutoencoderKL model.

        """

    @overload
    def __init__(self, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.


        2. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.


        3. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        4. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        5. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, model: openvino_genai.py_openvino_genai.AutoencoderKL) -> None

        AutoencoderKL model
                    AutoencoderKL class.
                    model (AutoencoderKL): AutoencoderKL model.

        """

    @overload
    def __init__(self, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.


        2. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.


        3. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        4. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        5. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, model: openvino_genai.py_openvino_genai.AutoencoderKL) -> None

        AutoencoderKL model
                    AutoencoderKL class.
                    model (AutoencoderKL): AutoencoderKL model.

        """

    @overload
    def __init__(self, model: AutoencoderKL) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.


        2. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike) -> None


                    AutoencoderKL class initialized with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.


        3. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        4. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None


                    AutoencoderKL class initialized only with both encoder and decoder models.
                    vae_encoder_path (str): VAE encoder directory.
                    vae_decoder_path (str): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        5. __init__(self: openvino_genai.py_openvino_genai.AutoencoderKL, model: openvino_genai.py_openvino_genai.AutoencoderKL) -> None

        AutoencoderKL model
                    AutoencoderKL class.
                    model (AutoencoderKL): AutoencoderKL model.

        """

    def compile(self, device: str, **kwargs) -> None:
        """compile(self: openvino_genai.py_openvino_genai.AutoencoderKL, device: str, **kwargs) -> None

        device on which inference will be done
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.

        """

    def decode(self, latent: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        """decode(self: openvino_genai.py_openvino_genai.AutoencoderKL, latent: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor"""

    def encode(self, image: openvino._pyopenvino.Tensor, generator: Generator) -> openvino._pyopenvino.Tensor:
        """encode(self: openvino_genai.py_openvino_genai.AutoencoderKL, image: openvino._pyopenvino.Tensor, generator: openvino_genai.py_openvino_genai.Generator) -> openvino._pyopenvino.Tensor"""

    def get_config(self) -> AutoencoderKL.Config:
        """get_config(self: openvino_genai.py_openvino_genai.AutoencoderKL) -> openvino_genai.py_openvino_genai.AutoencoderKL.Config"""

    def get_vae_scale_factor(self) -> int:
        """get_vae_scale_factor(self: openvino_genai.py_openvino_genai.AutoencoderKL) -> int"""

    def reshape(self, batch_size: int, height: int, width: int) -> AutoencoderKL:
        """reshape(self: openvino_genai.py_openvino_genai.AutoencoderKL, batch_size: int, height: int, width: int) -> openvino_genai.py_openvino_genai.AutoencoderKL"""


class CLIPTextModel:
    class Config:
        max_position_embeddings: int
        num_hidden_layers: int

        def __init__(self, config_path: str) -> None:
            """__init__(self: openvino_genai.py_openvino_genai.CLIPTextModel.Config, config_path: str) -> None"""

    @overload
    def __init__(self, root_dir: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, root_dir: os.PathLike) -> None


                    CLIPTextModel class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, root_dir: os.PathLike, device: str, **kwargs) -> None


                    CLIPTextModel class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, model: openvino_genai.py_openvino_genai.CLIPTextModel) -> None

        CLIPText model
                    CLIPTextModel class
                    model (CLIPTextModel): CLIPText model

        """

    @overload
    def __init__(self, root_dir: os.PathLike, device: str, **kwargs) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, root_dir: os.PathLike) -> None


                    CLIPTextModel class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, root_dir: os.PathLike, device: str, **kwargs) -> None


                    CLIPTextModel class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, model: openvino_genai.py_openvino_genai.CLIPTextModel) -> None

        CLIPText model
                    CLIPTextModel class
                    model (CLIPTextModel): CLIPText model

        """

    @overload
    def __init__(self, model: CLIPTextModel) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, root_dir: os.PathLike) -> None


                    CLIPTextModel class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, root_dir: os.PathLike, device: str, **kwargs) -> None


                    CLIPTextModel class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModel, model: openvino_genai.py_openvino_genai.CLIPTextModel) -> None

        CLIPText model
                    CLIPTextModel class
                    model (CLIPTextModel): CLIPText model

        """

    def compile(self, device: str, **kwargs) -> None:
        """compile(self: openvino_genai.py_openvino_genai.CLIPTextModel, device: str, **kwargs) -> None


                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.

        """

    def get_config(self) -> CLIPTextModel.Config:
        """get_config(self: openvino_genai.py_openvino_genai.CLIPTextModel) -> openvino_genai.py_openvino_genai.CLIPTextModel.Config"""

    def get_output_tensor(self, idx: int) -> openvino._pyopenvino.Tensor:
        """get_output_tensor(self: openvino_genai.py_openvino_genai.CLIPTextModel, idx: int) -> openvino._pyopenvino.Tensor"""

    def infer(self, pos_prompt: str, neg_prompt: str, do_classifier_free_guidance: bool) -> openvino._pyopenvino.Tensor:
        """infer(self: openvino_genai.py_openvino_genai.CLIPTextModel, pos_prompt: str, neg_prompt: str, do_classifier_free_guidance: bool) -> openvino._pyopenvino.Tensor"""

    def reshape(self, batch_size: int) -> CLIPTextModel:
        """reshape(self: openvino_genai.py_openvino_genai.CLIPTextModel, batch_size: int) -> openvino_genai.py_openvino_genai.CLIPTextModel"""

    def set_adapters(self, adapters: AdapterConfig | None) -> None:
        """set_adapters(self: openvino_genai.py_openvino_genai.CLIPTextModel, adapters: Optional[openvino_genai.py_openvino_genai.AdapterConfig]) -> None"""


class CLIPTextModelWithProjection:
    class Config:
        max_position_embeddings: int
        num_hidden_layers: int

        def __init__(self, config_path: os.PathLike) -> None:
            """__init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection.Config, config_path: os.PathLike) -> None"""

    @overload
    def __init__(self, root_dir: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, root_dir: os.PathLike) -> None


                    CLIPTextModelWithProjection class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, root_dir: os.PathLike, device: str, **kwargs) -> None


                    CLIPTextModelWithProjection class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, model: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection) -> None

        CLIPTextModelWithProjection model
                    CLIPTextModelWithProjection class
                    model (CLIPTextModelWithProjection): CLIPTextModelWithProjection model

        """

    @overload
    def __init__(self, root_dir: os.PathLike, device: str, **kwargs) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, root_dir: os.PathLike) -> None


                    CLIPTextModelWithProjection class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, root_dir: os.PathLike, device: str, **kwargs) -> None


                    CLIPTextModelWithProjection class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, model: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection) -> None

        CLIPTextModelWithProjection model
                    CLIPTextModelWithProjection class
                    model (CLIPTextModelWithProjection): CLIPTextModelWithProjection model

        """

    @overload
    def __init__(self, model: CLIPTextModelWithProjection) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, root_dir: os.PathLike) -> None


                    CLIPTextModelWithProjection class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, root_dir: os.PathLike, device: str, **kwargs) -> None


                    CLIPTextModelWithProjection class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, model: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection) -> None

        CLIPTextModelWithProjection model
                    CLIPTextModelWithProjection class
                    model (CLIPTextModelWithProjection): CLIPTextModelWithProjection model

        """

    def compile(self, device: str, **kwargs) -> None:
        """compile(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, device: str, **kwargs) -> None


                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.

        """

    def get_config(self) -> CLIPTextModelWithProjection.Config:
        """get_config(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection) -> openvino_genai.py_openvino_genai.CLIPTextModelWithProjection.Config"""

    def get_output_tensor(self, idx: int) -> openvino._pyopenvino.Tensor:
        """get_output_tensor(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, idx: int) -> openvino._pyopenvino.Tensor"""

    def infer(self, pos_prompt: str, neg_prompt: str, do_classifier_free_guidance: bool) -> openvino._pyopenvino.Tensor:
        """infer(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, pos_prompt: str, neg_prompt: str, do_classifier_free_guidance: bool) -> openvino._pyopenvino.Tensor"""

    def reshape(self, batch_size: int) -> CLIPTextModelWithProjection:
        """reshape(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, batch_size: int) -> openvino_genai.py_openvino_genai.CLIPTextModelWithProjection"""

    def set_adapters(self, adapters: AdapterConfig | None) -> None:
        """set_adapters(self: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, adapters: Optional[openvino_genai.py_openvino_genai.AdapterConfig]) -> None"""


class CacheEvictionConfig:
    aggregation_mode: AggregationMode

    def __init__(self, start_size: int, recent_size: int, max_cache_size: int,
                 aggregation_mode: AggregationMode) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.CacheEvictionConfig, start_size: int, recent_size: int, max_cache_size: int, aggregation_mode: openvino_genai.py_openvino_genai.AggregationMode) -> None"""

    def get_evictable_size(self) -> int:
        """get_evictable_size(self: openvino_genai.py_openvino_genai.CacheEvictionConfig) -> int"""

    def get_max_cache_size(self) -> int:
        """get_max_cache_size(self: openvino_genai.py_openvino_genai.CacheEvictionConfig) -> int"""

    def get_recent_size(self) -> int:
        """get_recent_size(self: openvino_genai.py_openvino_genai.CacheEvictionConfig) -> int"""

    def get_start_size(self) -> int:
        """get_start_size(self: openvino_genai.py_openvino_genai.CacheEvictionConfig) -> int"""


class ChunkStreamerBase:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.ChunkStreamerBase) -> None"""

    def end(self) -> None:
        """end(self: openvino_genai.py_openvino_genai.ChunkStreamerBase) -> None

        End is called at the end of generation. It can be used to flush cache if your own streamer has one
        """

    def put(self, arg0: int) -> bool:
        """put(self: openvino_genai.py_openvino_genai.ChunkStreamerBase, arg0: int) -> bool

        Put is called every time new token is generated. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """

    def put_chunk(self, arg0: list[int]) -> bool:
        """put_chunk(self: openvino_genai.py_openvino_genai.ChunkStreamerBase, arg0: list[int]) -> bool

        Put is called every time new token chunk is generated. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """


class ContinuousBatchingPipeline:
    @overload
    def __init__(self, models_path: str, scheduler_config: SchedulerConfig, device: str,
                 properties: dict[str, object] = ..., tokenizer_properties: dict[str, object] = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, models_path: str, scheduler_config: openvino_genai.py_openvino_genai.SchedulerConfig, device: str, properties: dict[str, object] = {}, tokenizer_properties: dict[str, object] = {}) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, models_path: str, tokenizer: openvino_genai.py_openvino_genai.Tokenizer, scheduler_config: openvino_genai.py_openvino_genai.SchedulerConfig, device: str, properties: dict[str, object] = {}) -> None
        """

    @overload
    def __init__(self, models_path: str, tokenizer: Tokenizer, scheduler_config: SchedulerConfig, device: str,
                 properties: dict[str, object] = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, models_path: str, scheduler_config: openvino_genai.py_openvino_genai.SchedulerConfig, device: str, properties: dict[str, object] = {}, tokenizer_properties: dict[str, object] = {}) -> None

        2. __init__(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, models_path: str, tokenizer: openvino_genai.py_openvino_genai.Tokenizer, scheduler_config: openvino_genai.py_openvino_genai.SchedulerConfig, device: str, properties: dict[str, object] = {}) -> None
        """

    @overload
    def add_request(self, request_id: int, input_ids: openvino._pyopenvino.Tensor,
                    sampling_params: GenerationConfig) -> GenerationHandle:
        """add_request(*args, **kwargs)
        Overloaded function.

        1. add_request(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, request_id: int, input_ids: openvino._pyopenvino.Tensor, sampling_params: openvino_genai.py_openvino_genai.GenerationConfig) -> openvino_genai.py_openvino_genai.GenerationHandle

        2. add_request(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, request_id: int, prompt: str, sampling_params: openvino_genai.py_openvino_genai.GenerationConfig) -> openvino_genai.py_openvino_genai.GenerationHandle
        """

    @overload
    def add_request(self, request_id: int, prompt: str, sampling_params: GenerationConfig) -> GenerationHandle:
        """add_request(*args, **kwargs)
        Overloaded function.

        1. add_request(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, request_id: int, input_ids: openvino._pyopenvino.Tensor, sampling_params: openvino_genai.py_openvino_genai.GenerationConfig) -> openvino_genai.py_openvino_genai.GenerationHandle

        2. add_request(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, request_id: int, prompt: str, sampling_params: openvino_genai.py_openvino_genai.GenerationConfig) -> openvino_genai.py_openvino_genai.GenerationHandle
        """

    @overload
    def generate(self, input_ids: list[openvino._pyopenvino.Tensor], sampling_params: list[GenerationConfig],
                 streamer: Callable[[str], bool] | StreamerBase | None = ...) -> list[EncodedGenerationResult]:
        """generate(*args, **kwargs)
        Overloaded function.

        1. generate(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, input_ids: list[openvino._pyopenvino.Tensor], sampling_params: list[openvino_genai.py_openvino_genai.GenerationConfig], streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None) -> list[openvino_genai.py_openvino_genai.EncodedGenerationResult]

        2. generate(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, prompts: list[str], sampling_params: list[openvino_genai.py_openvino_genai.GenerationConfig], streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None) -> list[openvino_genai.py_openvino_genai.GenerationResult]
        """

    @overload
    def generate(self, prompts: list[str], sampling_params: list[GenerationConfig],
                 streamer: Callable[[str], bool] | StreamerBase | None = ...) -> list[GenerationResult]:
        """generate(*args, **kwargs)
        Overloaded function.

        1. generate(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, input_ids: list[openvino._pyopenvino.Tensor], sampling_params: list[openvino_genai.py_openvino_genai.GenerationConfig], streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None) -> list[openvino_genai.py_openvino_genai.EncodedGenerationResult]

        2. generate(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline, prompts: list[str], sampling_params: list[openvino_genai.py_openvino_genai.GenerationConfig], streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None) -> list[openvino_genai.py_openvino_genai.GenerationResult]
        """

    def get_config(self) -> GenerationConfig:
        """get_config(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline) -> openvino_genai.py_openvino_genai.GenerationConfig"""

    def get_metrics(self) -> PipelineMetrics:
        """get_metrics(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline) -> openvino_genai.py_openvino_genai.PipelineMetrics"""

    def get_tokenizer(self) -> Tokenizer:
        """get_tokenizer(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline) -> openvino_genai.py_openvino_genai.Tokenizer"""

    def has_non_finished_requests(self) -> bool:
        """has_non_finished_requests(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline) -> bool"""

    def step(self) -> None:
        """step(self: openvino_genai.py_openvino_genai.ContinuousBatchingPipeline) -> None"""


class CppStdGenerator(Generator):
    def __init__(self, seed: int) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.CppStdGenerator, seed: int) -> None"""

    def next(self) -> float:
        """next(self: openvino_genai.py_openvino_genai.CppStdGenerator) -> float"""

    def randn_tensor(self, shape: openvino._pyopenvino.Shape) -> openvino._pyopenvino.Tensor:
        """randn_tensor(self: openvino_genai.py_openvino_genai.CppStdGenerator, shape: openvino._pyopenvino.Shape) -> openvino._pyopenvino.Tensor"""


class DecodedResults:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.DecodedResults) -> None"""

    @property
    def perf_metrics(self) -> PerfMetrics: ...

    @property
    def scores(self) -> list[float]: ...

    @property
    def texts(self) -> list[str]: ...


class EncodedGenerationResult:
    m_generation_ids: list[list[int]]
    m_scores: list[float]

    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.EncodedGenerationResult) -> None"""

    @property
    def m_request_id(self) -> int: ...


class EncodedResults:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

    @property
    def perf_metrics(self) -> PerfMetrics: ...

    @property
    def scores(self) -> list[float]: ...

    @property
    def tokens(self) -> list[list[int]]: ...


class GenerationConfig:
    adapters: AdapterConfig | None
    assistant_confidence_threshold: float
    diversity_penalty: float
    do_sample: bool
    echo: bool
    eos_token_id: int
    frequency_penalty: float
    ignore_eos: bool
    include_stop_str_in_output: bool
    length_penalty: float
    logprobs: int
    max_length: int
    max_new_tokens: int
    min_new_tokens: int
    no_repeat_ngram_size: int
    num_assistant_tokens: int
    num_beam_groups: int
    num_beams: int
    num_return_sequences: int
    presence_penalty: float
    repetition_penalty: float
    rng_seed: int
    stop_criteria: StopCriteria
    stop_strings: set[str]
    stop_token_ids: set[int]
    temperature: float
    top_k: int
    top_p: float

    @overload
    def __init__(self, json_path: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.GenerationConfig, json_path: os.PathLike) -> None

        path where generation_config.json is stored

        2. __init__(self: openvino_genai.py_openvino_genai.GenerationConfig, **kwargs) -> None
        """

    @overload
    def __init__(self, **kwargs) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.GenerationConfig, json_path: os.PathLike) -> None

        path where generation_config.json is stored

        2. __init__(self: openvino_genai.py_openvino_genai.GenerationConfig, **kwargs) -> None
        """

    def is_beam_search(self) -> bool:
        """is_beam_search(self: openvino_genai.py_openvino_genai.GenerationConfig) -> bool"""

    def is_greedy_decoding(self) -> bool:
        """is_greedy_decoding(self: openvino_genai.py_openvino_genai.GenerationConfig) -> bool"""

    def is_speculative_decoding(self) -> bool:
        """is_speculative_decoding(self: openvino_genai.py_openvino_genai.GenerationConfig) -> bool"""

    def set_eos_token_id(self, tokenizer_eos_token_id: int) -> None:
        """set_eos_token_id(self: openvino_genai.py_openvino_genai.GenerationConfig, tokenizer_eos_token_id: int) -> None"""

    def update_generation_config(self, config_map: dict[str, openvino._pyopenvino.OVAny]) -> None:
        """update_generation_config(self: openvino_genai.py_openvino_genai.GenerationConfig, config_map: dict[str, openvino._pyopenvino.OVAny]) -> None"""


class GenerationFinishReason:
    __members__: ClassVar[dict] = ...  # read-only
    LENGTH: ClassVar[GenerationFinishReason] = ...
    NONE: ClassVar[GenerationFinishReason] = ...
    STOP: ClassVar[GenerationFinishReason] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.GenerationFinishReason, value: int) -> None"""

    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""

    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""

    def __index__(self) -> int:
        """__index__(self: openvino_genai.py_openvino_genai.GenerationFinishReason) -> int"""

    def __int__(self) -> int:
        """__int__(self: openvino_genai.py_openvino_genai.GenerationFinishReason) -> int"""

    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class GenerationHandle:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

    def back(self) -> dict[int, GenerationOutput]:
        """back(self: openvino_genai.py_openvino_genai.GenerationHandle) -> dict[int, openvino_genai.py_openvino_genai.GenerationOutput]"""

    def can_read(self) -> bool:
        """can_read(self: openvino_genai.py_openvino_genai.GenerationHandle) -> bool"""

    def drop(self) -> None:
        """drop(self: openvino_genai.py_openvino_genai.GenerationHandle) -> None"""

    def get_status(self) -> GenerationStatus:
        """get_status(self: openvino_genai.py_openvino_genai.GenerationHandle) -> openvino_genai.py_openvino_genai.GenerationStatus"""

    def read(self) -> dict[int, GenerationOutput]:
        """read(self: openvino_genai.py_openvino_genai.GenerationHandle) -> dict[int, openvino_genai.py_openvino_genai.GenerationOutput]"""

    def read_all(self) -> list[GenerationOutput]:
        """read_all(self: openvino_genai.py_openvino_genai.GenerationHandle) -> list[openvino_genai.py_openvino_genai.GenerationOutput]"""


class GenerationOutput:
    finish_reason: GenerationFinishReason
    generated_ids: list[int]
    generated_log_probs: list[float]
    score: float

    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""


class GenerationResult:
    m_generation_ids: list[str]
    m_scores: list[float]

    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.GenerationResult) -> None"""

    def get_generation_ids(self) -> list[str]:
        """get_generation_ids(self: openvino_genai.py_openvino_genai.GenerationResult) -> list[str]"""

    @property
    def m_request_id(self) -> int: ...


class GenerationStatus:
    __members__: ClassVar[dict] = ...  # read-only
    DROPPED_BY_HANDLE: ClassVar[GenerationStatus] = ...
    DROPPED_BY_PIPELINE: ClassVar[GenerationStatus] = ...
    FINISHED: ClassVar[GenerationStatus] = ...
    IGNORED: ClassVar[GenerationStatus] = ...
    RUNNING: ClassVar[GenerationStatus] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.GenerationStatus, value: int) -> None"""

    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""

    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""

    def __index__(self) -> int:
        """__index__(self: openvino_genai.py_openvino_genai.GenerationStatus) -> int"""

    def __int__(self) -> int:
        """__int__(self: openvino_genai.py_openvino_genai.GenerationStatus) -> int"""

    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class Generator:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.Generator) -> None"""


class ImageGenerationConfig:
    adapters: AdapterConfig | None
    generator: Generator
    guidance_scale: float
    height: int
    negative_prompt: str | None
    negative_prompt_2: str | None
    negative_prompt_3: str | None
    num_images_per_prompt: int
    num_inference_steps: int
    prompt_2: str | None
    prompt_3: str | None
    strength: float
    width: int

    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.ImageGenerationConfig) -> None"""

    def update_generation_config(self, **kwargs) -> None:
        """update_generation_config(self: openvino_genai.py_openvino_genai.ImageGenerationConfig, **kwargs) -> None"""

    def validate(self) -> None:
        """validate(self: openvino_genai.py_openvino_genai.ImageGenerationConfig) -> None"""


class LLMPipeline:
    @overload
    def __init__(self, models_path: os.PathLike, tokenizer: Tokenizer, device: str, config: dict[str, object] = ...,
                 **kwargs) -> None:
        '''__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.LLMPipeline, models_path: os.PathLike, tokenizer: openvino_genai.py_openvino_genai.Tokenizer, device: str, config: dict[str, object] = {}, **kwargs) -> None


                    LLMPipeline class constructor for manually created openvino_genai.Tokenizer.
                    models_path (str): Path to the model file.
                    tokenizer (openvino_genai.Tokenizer): tokenizer object.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is \'CPU\'.
                    Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
                    kwargs: Device properties.


        2. __init__(self: openvino_genai.py_openvino_genai.LLMPipeline, models_path: os.PathLike, device: str, config: dict[str, object] = {}, **kwargs) -> None


                    LLMPipeline class constructor.
                    models_path (str): Path to the model file.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is \'CPU\'.
                    Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
                    kwargs: Device properties.

        '''

    @overload
    def __init__(self, models_path: os.PathLike, device: str, config: dict[str, object] = ..., **kwargs) -> None:
        '''__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.LLMPipeline, models_path: os.PathLike, tokenizer: openvino_genai.py_openvino_genai.Tokenizer, device: str, config: dict[str, object] = {}, **kwargs) -> None


                    LLMPipeline class constructor for manually created openvino_genai.Tokenizer.
                    models_path (str): Path to the model file.
                    tokenizer (openvino_genai.Tokenizer): tokenizer object.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is \'CPU\'.
                    Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
                    kwargs: Device properties.


        2. __init__(self: openvino_genai.py_openvino_genai.LLMPipeline, models_path: os.PathLike, device: str, config: dict[str, object] = {}, **kwargs) -> None


                    LLMPipeline class constructor.
                    models_path (str): Path to the model file.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is \'CPU\'.
                    Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
                    kwargs: Device properties.

        '''

    def finish_chat(self) -> None:
        """finish_chat(self: openvino_genai.py_openvino_genai.LLMPipeline) -> None"""

    def generate(self, inputs: openvino._pyopenvino.Tensor | TokenizedInputs | str | list[str],
                 generation_config: GenerationConfig | None = ...,
                 streamer: Callable[[str], bool] | StreamerBase | None = ...,
                 **kwargs) -> EncodedResults | DecodedResults:
        '''generate(self: openvino_genai.py_openvino_genai.LLMPipeline, inputs: Union[openvino._pyopenvino.Tensor, openvino_genai.py_openvino_genai.TokenizedInputs, str, list[str]], generation_config: Optional[openvino_genai.py_openvino_genai.GenerationConfig] = None, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.EncodedResults, openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences or tokens for LLMs. If input is a string or list of strings then resulting sequences will be already detokenized.

            :param inputs: inputs in the form of string, list of strings or tokenized input_ids
            :type inputs: str, List[str], ov.genai.TokenizedInputs, or ov.Tensor

            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict

            :return: return results in encoded, or decoded form depending on inputs type
            :rtype: DecodedResults, EncodedResults, str


            Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
            and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
            be used while greedy and beam search parameters will not affect decoding at all.

            Parameters:
            max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                           max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
            eos_token_id:  token_id of <eos> (end of sentence)
            min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens. Ignored for non continuous batching.
            stop_strings: list of strings that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
            include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
            stop_token_ids: list of tokens that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
            echo:           if set to true, the model will echo the prompt in the output.
            logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                            Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).

            Beam search specific parameters:
            num_beams:         number of beams for beam search. 1 disables beam search.
            num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            diversity_penalty: value is subtracted from a beam\'s score if it generates the same token as any beam from other group at a particular time.
            length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
                the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
                likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
                length_penalty < 0.0 encourages shorter sequences.
            num_return_sequences: the number of sequences to return for grouped beam search decoding.
            no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
            stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
                "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
                "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
                "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).

            Random sampling parameters:
            temperature:        the value used to modulate token probabilities for random sampling.
            top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
            do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.

        '''

    def get_generation_config(self) -> GenerationConfig:
        """get_generation_config(self: openvino_genai.py_openvino_genai.LLMPipeline) -> openvino_genai.py_openvino_genai.GenerationConfig"""

    def get_tokenizer(self) -> Tokenizer:
        """get_tokenizer(self: openvino_genai.py_openvino_genai.LLMPipeline) -> openvino_genai.py_openvino_genai.Tokenizer"""

    def set_generation_config(self, config: GenerationConfig) -> None:
        """set_generation_config(self: openvino_genai.py_openvino_genai.LLMPipeline, config: openvino_genai.py_openvino_genai.GenerationConfig) -> None"""

    def start_chat(self, system_message: str = ...) -> None:
        """start_chat(self: openvino_genai.py_openvino_genai.LLMPipeline, system_message: str = '') -> None"""

    def __call__(self, inputs: openvino._pyopenvino.Tensor | TokenizedInputs | str | list[str],
                 generation_config: GenerationConfig | None = ...,
                 streamer: Callable[[str], bool] | StreamerBase | None = ...,
                 **kwargs) -> EncodedResults | DecodedResults:
        '''__call__(self: openvino_genai.py_openvino_genai.LLMPipeline, inputs: Union[openvino._pyopenvino.Tensor, openvino_genai.py_openvino_genai.TokenizedInputs, str, list[str]], generation_config: Optional[openvino_genai.py_openvino_genai.GenerationConfig] = None, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.EncodedResults, openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences or tokens for LLMs. If input is a string or list of strings then resulting sequences will be already detokenized.

            :param inputs: inputs in the form of string, list of strings or tokenized input_ids
            :type inputs: str, List[str], ov.genai.TokenizedInputs, or ov.Tensor

            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict

            :return: return results in encoded, or decoded form depending on inputs type
            :rtype: DecodedResults, EncodedResults, str


            Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
            and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
            be used while greedy and beam search parameters will not affect decoding at all.

            Parameters:
            max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                           max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
            eos_token_id:  token_id of <eos> (end of sentence)
            min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens. Ignored for non continuous batching.
            stop_strings: list of strings that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
            include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
            stop_token_ids: list of tokens that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
            echo:           if set to true, the model will echo the prompt in the output.
            logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                            Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).

            Beam search specific parameters:
            num_beams:         number of beams for beam search. 1 disables beam search.
            num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            diversity_penalty: value is subtracted from a beam\'s score if it generates the same token as any beam from other group at a particular time.
            length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
                the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
                likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
                length_penalty < 0.0 encourages shorter sequences.
            num_return_sequences: the number of sequences to return for grouped beam search decoding.
            no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
            stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
                "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
                "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
                "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).

            Random sampling parameters:
            temperature:        the value used to modulate token probabilities for random sampling.
            top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
            do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.

        '''


class MeanStdPair:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.MeanStdPair) -> None"""

    def __iter__(self) -> Iterator[float]:
        """__iter__(self: openvino_genai.py_openvino_genai.MeanStdPair) -> Iterator[float]"""

    @property
    def mean(self) -> float: ...

    @property
    def std(self) -> float: ...


class PerfMetrics:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.PerfMetrics) -> None"""

    def get_detokenization_duration(self) -> MeanStdPair:
        """get_detokenization_duration(self: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.MeanStdPair"""

    def get_generate_duration(self) -> MeanStdPair:
        """get_generate_duration(self: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.MeanStdPair"""

    def get_inference_duration(self) -> MeanStdPair:
        """get_inference_duration(self: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.MeanStdPair"""

    def get_ipot(self) -> MeanStdPair:
        """get_ipot(self: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.MeanStdPair"""

    def get_load_time(self) -> float:
        """get_load_time(self: openvino_genai.py_openvino_genai.PerfMetrics) -> float"""

    def get_num_generated_tokens(self) -> int:
        """get_num_generated_tokens(self: openvino_genai.py_openvino_genai.PerfMetrics) -> int"""

    def get_num_input_tokens(self) -> int:
        """get_num_input_tokens(self: openvino_genai.py_openvino_genai.PerfMetrics) -> int"""

    def get_throughput(self) -> MeanStdPair:
        """get_throughput(self: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.MeanStdPair"""

    def get_tokenization_duration(self) -> MeanStdPair:
        """get_tokenization_duration(self: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.MeanStdPair"""

    def get_tpot(self) -> MeanStdPair:
        """get_tpot(self: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.MeanStdPair"""

    def get_ttft(self) -> MeanStdPair:
        """get_ttft(self: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.MeanStdPair"""

    def __add__(self, metrics: PerfMetrics) -> PerfMetrics:
        """__add__(self: openvino_genai.py_openvino_genai.PerfMetrics, metrics: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.PerfMetrics"""

    def __iadd__(self, right: PerfMetrics) -> PerfMetrics:
        """__iadd__(self: openvino_genai.py_openvino_genai.PerfMetrics, right: openvino_genai.py_openvino_genai.PerfMetrics) -> openvino_genai.py_openvino_genai.PerfMetrics"""

    @property
    def raw_metrics(self) -> RawPerfMetrics: ...


class PipelineMetrics:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.PipelineMetrics) -> None"""

    @property
    def avg_cache_usage(self) -> float: ...

    @property
    def cache_usage(self) -> float: ...

    @property
    def max_cache_usage(self) -> float: ...

    @property
    def requests(self) -> int: ...

    @property
    def scheduled_requests(self) -> int: ...


class RawPerfMetrics:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.RawPerfMetrics) -> None"""

    @property
    def detokenization_durations(self) -> list[float]: ...

    @property
    def generate_durations(self) -> list[float]: ...

    @property
    def m_batch_sizes(self) -> list[int]: ...

    @property
    def m_durations(self) -> list[float]: ...

    @property
    def m_new_token_times(self) -> list[float]: ...

    @property
    def m_times_to_first_token(self) -> list[float]: ...

    @property
    def tokenization_durations(self) -> list[float]: ...


class Scheduler:
    class Type:
        __members__: ClassVar[dict] = ...  # read-only
        AUTO: ClassVar[Scheduler.Type] = ...
        DDIM: ClassVar[Scheduler.Type] = ...
        EULER_DISCRETE: ClassVar[Scheduler.Type] = ...
        FLOW_MATCH_EULER_DISCRETE: ClassVar[Scheduler.Type] = ...
        LCM: ClassVar[Scheduler.Type] = ...
        LMS_DISCRETE: ClassVar[Scheduler.Type] = ...
        __entries: ClassVar[dict] = ...

        def __init__(self, value: int) -> None:
            """__init__(self: openvino_genai.py_openvino_genai.Scheduler.Type, value: int) -> None"""

        def __eq__(self, other: object) -> bool:
            """__eq__(self: object, other: object) -> bool"""

        def __hash__(self) -> int:
            """__hash__(self: object) -> int"""

        def __index__(self) -> int:
            """__index__(self: openvino_genai.py_openvino_genai.Scheduler.Type) -> int"""

        def __int__(self) -> int:
            """__int__(self: openvino_genai.py_openvino_genai.Scheduler.Type) -> int"""

        def __ne__(self, other: object) -> bool:
            """__ne__(self: object, other: object) -> bool"""

        @property
        def name(self) -> str: ...

        @property
        def value(self) -> int: ...

    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

    @staticmethod
    def from_config(scheduler_config_path: os.PathLike, scheduler_type: Scheduler.Type = ...) -> Scheduler:
        """from_config(scheduler_config_path: os.PathLike, scheduler_type: openvino_genai.py_openvino_genai.Scheduler.Type = Scheduler.Type.AUTO) -> openvino_genai.py_openvino_genai.Scheduler"""


class SchedulerConfig:
    cache_eviction_config: CacheEvictionConfig
    cache_size: int
    dynamic_split_fuse: bool
    enable_prefix_caching: bool
    max_num_batched_tokens: int
    max_num_seqs: int
    num_kv_blocks: int
    use_cache_eviction: bool

    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.SchedulerConfig) -> None"""


class StopCriteria:
    __members__: ClassVar[dict] = ...  # read-only
    EARLY: ClassVar[StopCriteria] = ...
    HEURISTIC: ClassVar[StopCriteria] = ...
    NEVER: ClassVar[StopCriteria] = ...
    __entries: ClassVar[dict] = ...

    def __init__(self, value: int) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.StopCriteria, value: int) -> None"""

    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""

    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""

    def __index__(self) -> int:
        """__index__(self: openvino_genai.py_openvino_genai.StopCriteria) -> int"""

    def __int__(self) -> int:
        """__int__(self: openvino_genai.py_openvino_genai.StopCriteria) -> int"""

    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class StreamerBase:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.StreamerBase) -> None"""

    def end(self) -> None:
        """end(self: openvino_genai.py_openvino_genai.StreamerBase) -> None

        End is called at the end of generation. It can be used to flush cache if your own streamer has one
        """

    def put(self, token: int) -> bool:
        """put(self: openvino_genai.py_openvino_genai.StreamerBase, token: int) -> bool

        Put is called every time new token is decoded. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """


class Text2ImagePipeline:
    @overload
    def __init__(self, models_path: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, models_path: os.PathLike) -> None


                    Text2ImagePipeline class constructor.
                    models_path (str): Path to the folder with exported model files.


        2. __init__(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, models_path: os.PathLike, device: str, **kwargs) -> None


                    Text2ImagePipeline class constructor.
                    models_path (str): Path with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU).
                    kwargs: Text2ImagePipeline properties

        """

    @overload
    def __init__(self, models_path: os.PathLike, device: str, **kwargs) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, models_path: os.PathLike) -> None


                    Text2ImagePipeline class constructor.
                    models_path (str): Path to the folder with exported model files.


        2. __init__(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, models_path: os.PathLike, device: str, **kwargs) -> None


                    Text2ImagePipeline class constructor.
                    models_path (str): Path with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU).
                    kwargs: Text2ImagePipeline properties

        """

    def compile(self, device: str, **kwargs) -> None:
        """compile(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, device: str, **kwargs) -> None


                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.

        """

    def generate(self, prompt: str, **kwargs) -> openvino._pyopenvino.Tensor:
        """generate(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, prompt: str, **kwargs) -> Union[openvino._pyopenvino.Tensor]


            Generates images for text-to-image models.

            :param prompt: input prompt
            :type prompt: str

            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.

            Expected parameters list:
            prompt_2: str - second prompt,
            prompt_3: str - third prompt,
            negative_prompt: str - negative prompt,
            negative_prompt_2: str - second negative prompt,
            negative_prompt_3: str - third negative prompt,
            num_images_per_prompt: int - number of images, that should be generated per prompt,
            guidance_scale: float - guidance scale,
            generation_config: GenerationConfig,
            height: int - height of resulting images,
            width: int - width of resulting images,
            num_inference_steps: int - number of inference steps,
            generator: openvino_genai.CppStdGenerator or class inherited from openvino_genai.Generator - random generator
            adapters: LoRA adapters
            strength: strength for image to image generation. 1.0f means initial image is fully noised

            :return: ov.Tensor with resulting images
            :rtype: ov.Tensor


        """

    def get_generation_config(self) -> ImageGenerationConfig:
        """get_generation_config(self: openvino_genai.py_openvino_genai.Text2ImagePipeline) -> openvino_genai.py_openvino_genai.ImageGenerationConfig"""

    @staticmethod
    def latent_consistency_model(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel,
                                 vae: AutoencoderKL) -> Text2ImagePipeline:
        """latent_consistency_model(scheduler: openvino_genai.py_openvino_genai.Scheduler, clip_text_model: openvino_genai.py_openvino_genai.CLIPTextModel, unet: openvino_genai.py_openvino_genai.UNet2DConditionModel, vae: openvino_genai.py_openvino_genai.AutoencoderKL) -> openvino_genai.py_openvino_genai.Text2ImagePipeline"""

    def reshape(self, num_images_per_prompt: int, height: int, width: int, guidance_scale: float) -> None:
        """reshape(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, num_images_per_prompt: int, height: int, width: int, guidance_scale: float) -> None"""

    def set_generation_config(self, generation_config: ImageGenerationConfig) -> None:
        """set_generation_config(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, generation_config: openvino_genai.py_openvino_genai.ImageGenerationConfig) -> None"""

    def set_scheduler(self, scheduler: Scheduler) -> None:
        """set_scheduler(self: openvino_genai.py_openvino_genai.Text2ImagePipeline, scheduler: openvino_genai.py_openvino_genai.Scheduler) -> None"""

    @staticmethod
    def stable_diffusion(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel,
                         vae: AutoencoderKL) -> Text2ImagePipeline:
        """stable_diffusion(scheduler: openvino_genai.py_openvino_genai.Scheduler, clip_text_model: openvino_genai.py_openvino_genai.CLIPTextModel, unet: openvino_genai.py_openvino_genai.UNet2DConditionModel, vae: openvino_genai.py_openvino_genai.AutoencoderKL) -> openvino_genai.py_openvino_genai.Text2ImagePipeline"""

    @staticmethod
    def stable_diffusion_xl(scheduler: Scheduler, clip_text_model: CLIPTextModel,
                            clip_text_model_with_projection: CLIPTextModelWithProjection, unet: UNet2DConditionModel,
                            vae: AutoencoderKL) -> Text2ImagePipeline:
        """stable_diffusion_xl(scheduler: openvino_genai.py_openvino_genai.Scheduler, clip_text_model: openvino_genai.py_openvino_genai.CLIPTextModel, clip_text_model_with_projection: openvino_genai.py_openvino_genai.CLIPTextModelWithProjection, unet: openvino_genai.py_openvino_genai.UNet2DConditionModel, vae: openvino_genai.py_openvino_genai.AutoencoderKL) -> openvino_genai.py_openvino_genai.Text2ImagePipeline"""


class TokenizedInputs:
    attention_mask: openvino._pyopenvino.Tensor
    input_ids: openvino._pyopenvino.Tensor

    def __init__(self, input_ids: openvino._pyopenvino.Tensor, attention_mask: openvino._pyopenvino.Tensor) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.TokenizedInputs, input_ids: openvino._pyopenvino.Tensor, attention_mask: openvino._pyopenvino.Tensor) -> None"""


class Tokenizer:
    def __init__(self, tokenizer_path: os.PathLike, properties: dict[str, object] = ...) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.Tokenizer, tokenizer_path: os.PathLike, properties: dict[str, object] = {}) -> None"""

    def apply_chat_template(self, history: list[dict[str, str]], add_generation_prompt: bool,
                            chat_template: str = ...) -> str:
        """apply_chat_template(self: openvino_genai.py_openvino_genai.Tokenizer, history: list[dict[str, str]], add_generation_prompt: bool, chat_template: str = '') -> str

        Embeds input prompts with special tags for a chat scenario.
        """

    @overload
    def decode(self, tokens: list[int]) -> str:
        """decode(*args, **kwargs)
        Overloaded function.

        1. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: list[int]) -> str

        Decode a sequence into a string prompt.

        2. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: openvino._pyopenvino.Tensor) -> list[str]

        Decode tensor into a list of string prompts.

        3. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: list[list[int]]) -> list[str]

        Decode a batch of tokens into a list of string prompt.
        """

    @overload
    def decode(self, tokens: openvino._pyopenvino.Tensor) -> list[str]:
        """decode(*args, **kwargs)
        Overloaded function.

        1. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: list[int]) -> str

        Decode a sequence into a string prompt.

        2. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: openvino._pyopenvino.Tensor) -> list[str]

        Decode tensor into a list of string prompts.

        3. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: list[list[int]]) -> list[str]

        Decode a batch of tokens into a list of string prompt.
        """

    @overload
    def decode(self, tokens: list[list[int]]) -> list[str]:
        """decode(*args, **kwargs)
        Overloaded function.

        1. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: list[int]) -> str

        Decode a sequence into a string prompt.

        2. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: openvino._pyopenvino.Tensor) -> list[str]

        Decode tensor into a list of string prompts.

        3. decode(self: openvino_genai.py_openvino_genai.Tokenizer, tokens: list[list[int]]) -> list[str]

        Decode a batch of tokens into a list of string prompt.
        """

    @overload
    def encode(self, prompts: list[str], add_special_tokens: bool = ...) -> TokenizedInputs:
        """encode(*args, **kwargs)
        Overloaded function.

        1. encode(self: openvino_genai.py_openvino_genai.Tokenizer, prompts: list[str], add_special_tokens: bool = True) -> openvino_genai.py_openvino_genai.TokenizedInputs

        Encodes a list of prompts into tokenized inputs.

        2. encode(self: openvino_genai.py_openvino_genai.Tokenizer, prompt: str, add_special_tokens: bool = True) -> openvino_genai.py_openvino_genai.TokenizedInputs

        Encodes a single prompt into tokenized input.
        """

    @overload
    def encode(self, prompt: str, add_special_tokens: bool = ...) -> TokenizedInputs:
        """encode(*args, **kwargs)
        Overloaded function.

        1. encode(self: openvino_genai.py_openvino_genai.Tokenizer, prompts: list[str], add_special_tokens: bool = True) -> openvino_genai.py_openvino_genai.TokenizedInputs

        Encodes a list of prompts into tokenized inputs.

        2. encode(self: openvino_genai.py_openvino_genai.Tokenizer, prompt: str, add_special_tokens: bool = True) -> openvino_genai.py_openvino_genai.TokenizedInputs

        Encodes a single prompt into tokenized input.
        """

    def get_bos_token(self) -> str:
        """get_bos_token(self: openvino_genai.py_openvino_genai.Tokenizer) -> str"""

    def get_bos_token_id(self) -> int:
        """get_bos_token_id(self: openvino_genai.py_openvino_genai.Tokenizer) -> int"""

    def get_eos_token(self) -> str:
        """get_eos_token(self: openvino_genai.py_openvino_genai.Tokenizer) -> str"""

    def get_eos_token_id(self) -> int:
        """get_eos_token_id(self: openvino_genai.py_openvino_genai.Tokenizer) -> int"""

    def get_pad_token(self) -> str:
        """get_pad_token(self: openvino_genai.py_openvino_genai.Tokenizer) -> str"""

    def get_pad_token_id(self) -> int:
        """get_pad_token_id(self: openvino_genai.py_openvino_genai.Tokenizer) -> int"""

    def set_chat_template(self, chat_template: str) -> None:
        """set_chat_template(self: openvino_genai.py_openvino_genai.Tokenizer, chat_template: str) -> None

        Override a chat_template read from tokenizer_config.json.
        """


class UNet2DConditionModel:
    class Config:
        in_channels: int
        sample_size: int
        time_cond_proj_dim: int

        def __init__(self, config_path: os.PathLike) -> None:
            """__init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel.Config, config_path: os.PathLike) -> None"""

    @overload
    def __init__(self, root_dir: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, root_dir: os.PathLike) -> None


                    UNet2DConditionModel class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, root_dir: os.PathLike, device: str, **kwargs) -> None


                    UNet2DConditionModel class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, model: openvino_genai.py_openvino_genai.UNet2DConditionModel) -> None

        UNet2DConditionModel model
                    UNet2DConditionModel class
                    model (UNet2DConditionModel): UNet2DConditionModel model

        """

    @overload
    def __init__(self, root_dir: os.PathLike, device: str, **kwargs) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, root_dir: os.PathLike) -> None


                    UNet2DConditionModel class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, root_dir: os.PathLike, device: str, **kwargs) -> None


                    UNet2DConditionModel class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, model: openvino_genai.py_openvino_genai.UNet2DConditionModel) -> None

        UNet2DConditionModel model
                    UNet2DConditionModel class
                    model (UNet2DConditionModel): UNet2DConditionModel model

        """

    @overload
    def __init__(self, model: UNet2DConditionModel) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, root_dir: os.PathLike) -> None


                    UNet2DConditionModel class
                    root_dir (str): Model root directory.


        2. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, root_dir: os.PathLike, device: str, **kwargs) -> None


                    UNet2DConditionModel class
                    root_dir (str): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.


        3. __init__(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, model: openvino_genai.py_openvino_genai.UNet2DConditionModel) -> None

        UNet2DConditionModel model
                    UNet2DConditionModel class
                    model (UNet2DConditionModel): UNet2DConditionModel model

        """

    def compile(self, device: str, **kwargs) -> None:
        """compile(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, device: str, **kwargs) -> None


                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.

        """

    def get_config(self) -> UNet2DConditionModel.Config:
        """get_config(self: openvino_genai.py_openvino_genai.UNet2DConditionModel) -> openvino_genai.py_openvino_genai.UNet2DConditionModel.Config"""

    def infer(self, sample: openvino._pyopenvino.Tensor,
              timestep: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        """infer(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, sample: openvino._pyopenvino.Tensor, timestep: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor"""

    def reshape(self, batch_size: int, height: int, width: int,
                tokenizer_model_max_length: int) -> UNet2DConditionModel:
        """reshape(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, batch_size: int, height: int, width: int, tokenizer_model_max_length: int) -> openvino_genai.py_openvino_genai.UNet2DConditionModel"""

    def set_adapters(self, adapters: AdapterConfig | None) -> None:
        """set_adapters(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, adapters: Optional[openvino_genai.py_openvino_genai.AdapterConfig]) -> None"""

    def set_hidden_states(self, tensor_name: str, encoder_hidden_states: openvino._pyopenvino.Tensor) -> None:
        """set_hidden_states(self: openvino_genai.py_openvino_genai.UNet2DConditionModel, tensor_name: str, encoder_hidden_states: openvino._pyopenvino.Tensor) -> None"""


class VLMPipeline:
    def __init__(self, models_path: os.PathLike, device: str, **kwargs) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.VLMPipeline, models_path: os.PathLike, device: str, **kwargs) -> None

        device on which inference will be done
                    VLMPipeline class constructor.
                    models_path (str): Path to the folder with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
                    kwargs: Device properties

        """

    def finish_chat(self) -> None:
        """finish_chat(self: openvino_genai.py_openvino_genai.VLMPipeline) -> None"""

    @overload
    def generate(self, prompt: str, images: list[openvino._pyopenvino.Tensor], generation_config: GenerationConfig,
                 streamer: Callable[[str], bool] | StreamerBase | None = ..., **kwargs) -> DecodedResults:
        """generate(*args, **kwargs)
        Overloaded function.

        1. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, images: list[openvino._pyopenvino.Tensor], generation_config: openvino_genai.py_openvino_genai.GenerationConfig, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param images: image or list of images
            :type images: List[ov.Tensor] or ov.Tensor

            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict

            :return: return results in decoded form
            :rtype: DecodedResults



        2. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, images: openvino._pyopenvino.Tensor, generation_config: openvino_genai.py_openvino_genai.GenerationConfig, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param images: image or list of images
            :type images: List[ov.Tensor] or ov.Tensor

            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict

            :return: return results in decoded form
            :rtype: DecodedResults



        3. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.

            Expected parameters list:
            image: ov.Tensor - input image,
            images: List[ov.Tensor] - input images,
            generation_config: GenerationConfig,
            streamer: Callable[[str], bool], ov.genai.StreamerBase - streamer either as a lambda with a boolean returning flag whether generation should be stopped

            :return: return results in decoded form
            :rtype: DecodedResults


        """

    @overload
    def generate(self, prompt: str, images: openvino._pyopenvino.Tensor, generation_config: GenerationConfig,
                 streamer: Callable[[str], bool] | StreamerBase | None = ..., **kwargs) -> DecodedResults:
        """generate(*args, **kwargs)
        Overloaded function.

        1. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, images: list[openvino._pyopenvino.Tensor], generation_config: openvino_genai.py_openvino_genai.GenerationConfig, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param images: image or list of images
            :type images: List[ov.Tensor] or ov.Tensor

            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict

            :return: return results in decoded form
            :rtype: DecodedResults



        2. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, images: openvino._pyopenvino.Tensor, generation_config: openvino_genai.py_openvino_genai.GenerationConfig, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param images: image or list of images
            :type images: List[ov.Tensor] or ov.Tensor

            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict

            :return: return results in decoded form
            :rtype: DecodedResults



        3. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.

            Expected parameters list:
            image: ov.Tensor - input image,
            images: List[ov.Tensor] - input images,
            generation_config: GenerationConfig,
            streamer: Callable[[str], bool], ov.genai.StreamerBase - streamer either as a lambda with a boolean returning flag whether generation should be stopped

            :return: return results in decoded form
            :rtype: DecodedResults


        """

    @overload
    def generate(self, prompt: str, **kwargs) -> DecodedResults:
        """generate(*args, **kwargs)
        Overloaded function.

        1. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, images: list[openvino._pyopenvino.Tensor], generation_config: openvino_genai.py_openvino_genai.GenerationConfig, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param images: image or list of images
            :type images: List[ov.Tensor] or ov.Tensor

            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict

            :return: return results in decoded form
            :rtype: DecodedResults



        2. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, images: openvino._pyopenvino.Tensor, generation_config: openvino_genai.py_openvino_genai.GenerationConfig, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.StreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param images: image or list of images
            :type images: List[ov.Tensor] or ov.Tensor

            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict

            :return: return results in decoded form
            :rtype: DecodedResults



        3. generate(self: openvino_genai.py_openvino_genai.VLMPipeline, prompt: str, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            Generates sequences for VLMs.

            :param prompt: input prompt
            :type prompt: str

            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.

            Expected parameters list:
            image: ov.Tensor - input image,
            images: List[ov.Tensor] - input images,
            generation_config: GenerationConfig,
            streamer: Callable[[str], bool], ov.genai.StreamerBase - streamer either as a lambda with a boolean returning flag whether generation should be stopped

            :return: return results in decoded form
            :rtype: DecodedResults


        """

    def get_generation_config(self) -> GenerationConfig:
        """get_generation_config(self: openvino_genai.py_openvino_genai.VLMPipeline) -> openvino_genai.py_openvino_genai.GenerationConfig"""

    def get_tokenizer(self) -> Tokenizer:
        """get_tokenizer(self: openvino_genai.py_openvino_genai.VLMPipeline) -> openvino_genai.py_openvino_genai.Tokenizer"""

    def set_chat_template(self, new_template: str) -> None:
        """set_chat_template(self: openvino_genai.py_openvino_genai.VLMPipeline, new_template: str) -> None"""

    def set_generation_config(self, new_config: GenerationConfig) -> None:
        """set_generation_config(self: openvino_genai.py_openvino_genai.VLMPipeline, new_config: openvino_genai.py_openvino_genai.GenerationConfig) -> None"""

    def start_chat(self, system_message: str = ...) -> None:
        """start_chat(self: openvino_genai.py_openvino_genai.VLMPipeline, system_message: str = '') -> None"""


class WhisperDecodedResultChunk:
    def __init__(self) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.WhisperDecodedResultChunk) -> None"""

    @property
    def end_ts(self) -> float: ...

    @property
    def start_ts(self) -> float: ...

    @property
    def text(self) -> str: ...


class WhisperDecodedResults(DecodedResults):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

    @property
    def chunks(self) -> list[WhisperDecodedResultChunk] | None: ...


class WhisperGenerationConfig:
    begin_suppress_tokens: list[int]
    decoder_start_token_id: int
    eos_token_id: int
    is_multilingual: bool
    lang_to_id: dict[str, int]
    language: str | None
    max_initial_timestamp_index: int
    max_length: int
    max_new_tokens: int
    no_timestamps_token_id: int
    pad_token_id: int
    return_timestamps: bool
    suppress_tokens: list[int]
    task: str | None
    transcribe_token_id: int
    translate_token_id: int

    @overload
    def __init__(self, json_path: os.PathLike) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.WhisperGenerationConfig, json_path: os.PathLike) -> None

        path where generation_config.json is stored

        2. __init__(self: openvino_genai.py_openvino_genai.WhisperGenerationConfig, **kwargs) -> None
        """

    @overload
    def __init__(self, **kwargs) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: openvino_genai.py_openvino_genai.WhisperGenerationConfig, json_path: os.PathLike) -> None

        path where generation_config.json is stored

        2. __init__(self: openvino_genai.py_openvino_genai.WhisperGenerationConfig, **kwargs) -> None
        """

    def set_eos_token_id(self, tokenizer_eos_token_id: int) -> None:
        """set_eos_token_id(self: openvino_genai.py_openvino_genai.WhisperGenerationConfig, tokenizer_eos_token_id: int) -> None"""


class WhisperPipeline:
    def __init__(self, models_path: os.PathLike, device: str, **kwargs) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.WhisperPipeline, models_path: os.PathLike, device: str, **kwargs) -> None


                    WhisperPipeline class constructor.
                    models_path (str): Path to the model file.
                    device (str): Device to run the model on (e.g., CPU, GPU).

        """

    def generate(self, raw_speech_input: list[float], generation_config: WhisperGenerationConfig | None = ...,
                 streamer: Callable[[str], bool] | ChunkStreamerBase | None = ..., **kwargs) -> DecodedResults:
        '''generate(self: openvino_genai.py_openvino_genai.WhisperPipeline, raw_speech_input: list[float], generation_config: Optional[openvino_genai.py_openvino_genai.WhisperGenerationConfig] = None, streamer: Union[Callable[[str], bool], openvino_genai.py_openvino_genai.ChunkStreamerBase, None] = None, **kwargs) -> Union[openvino_genai.py_openvino_genai.DecodedResults]


            High level generate that receives raw speech as a vector of floats and returns decoded output.

            :param raw_speech_input: inputs in the form of list of floats. Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.
            :type raw_speech_input: List[float]

            :param generation_config: generation_config
            :type generation_config: WhisperGenerationConfig or a Dict

            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped.
                             Streamer supported for short-form audio (< 30 seconds) with `return_timestamps=False` only
            :type : Callable[[str], bool], ov.genai.StreamerBase

            :param kwargs: arbitrary keyword arguments with keys corresponding to WhisperGenerationConfig fields.
            :type : Dict

            :return: return results in encoded, or decoded form depending on inputs type
            :rtype: DecodedResults


            WhisperGenerationConfig
            :param max_length: the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                               `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
            :type max_length: int

            :param max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            :type max_new_tokens: int

            :param eos_token_id: End of stream token id.
            :type eos_token_id: int

            Whisper specific parameters:

            :param decoder_start_token_id: Corresponds to the ”<|startoftranscript|>” token.
            :type decoder_start_token_id: int

            :param pad_token_id: Padding token id.
            :type pad_token_id: int

            :param translate_token_id: Translate token id.
            :type translate_token_id: int

            :param transcribe_token_id: Transcribe token id.
            :type transcribe_token_id: int

            :param no_timestamps_token_id: No timestamps token id.
            :type no_timestamps_token_id: int

            :param is_multilingual:
            :type is_multilingual: bool

            :param begin_suppress_tokens: A list containing tokens that will be suppressed at the beginning of the sampling process.
            :type begin_suppress_tokens: list[int]

            :param suppress_tokens: A list containing the non-speech tokens that will be suppressed during generation.
            :type suppress_tokens: list[int]

            :param language: Language token to use for generation in the form of <|en|>.
                             You can find all the possible language tokens in the generation_config.json lang_to_id dictionary.
            :type language: Optional[str]

            :param lang_to_id: Language token to token_id map. Initialized from the generation_config.json lang_to_id dictionary.
            :type lang_to_id: Dict[str, int]

            :param task: Task to use for generation, either “translate” or “transcribe”
            :type task: int

            :param return_timestamps: If `true` the pipeline will return timestamps along the text for *segments* of words in the text.
                               For instance, if you get
                               WhisperDecodedResultChunk
                                   start_ts = 0.5
                                   end_ts = 1.5
                                   text = " Hi there!"
                               then it means the model predicts that the segment "Hi there!" was spoken after `0.5` and before `1.5` seconds.
                               Note that a segment of text refers to a sequence of one or more words, rather than individual words.
            :type return_timestamps: bool

        '''

    def get_generation_config(self) -> WhisperGenerationConfig:
        """get_generation_config(self: openvino_genai.py_openvino_genai.WhisperPipeline) -> openvino_genai.py_openvino_genai.WhisperGenerationConfig"""

    def get_tokenizer(self) -> Tokenizer:
        """get_tokenizer(self: openvino_genai.py_openvino_genai.WhisperPipeline) -> openvino_genai.py_openvino_genai.Tokenizer"""

    def set_generation_config(self, config: WhisperGenerationConfig) -> None:
        """set_generation_config(self: openvino_genai.py_openvino_genai.WhisperPipeline, config: openvino_genai.py_openvino_genai.WhisperGenerationConfig) -> None"""


class draft_model:
    def __init__(self, models_path: os.PathLike, device: str = ..., **kwargs) -> None:
        """__init__(self: openvino_genai.py_openvino_genai.draft_model, models_path: os.PathLike, device: str = '', **kwargs) -> None

        device on which inference will be performed
        """
