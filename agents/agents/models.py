"""
The following model specification classes are meant to define a comman interface for initialization parameters for ML models across supported model serving platforms.
"""

from typing import Optional, Dict

from attrs import define, field
from .ros import BaseAttrs, base_validators

__all__ = [
    "Encoder",
    "Llama3",
    "Llama3_1",
    "OllamaModel",
    "Idefics2",
    "Llava",
    "Whisper",
    "InstructBlip",
    "SpeechT5",
    "Bark",
    "VisionModel",
]

# ollama models map to model:latest tag
_ollama_mapping = [
    "llava",
    "llama3",
    "llama3_1",
    "phi3",
    "qwen2",
    "aya",
    "mistral",
    "mixtral",
    "gemma2",
]


@define(kw_only=True)
class Model(BaseAttrs):
    """Model configuration base class"""

    name: str
    checkpoint: str
    init_timeout: Optional[int] = field(default=None)

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {"checkpoint": self.checkpoint}


@define(kw_only=True)
class Encoder(Model):
    """A text encoder model that can be used with vector DBs.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "BAAI/bge-small-en".
    :type checkpoint: str
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional
    """

    checkpoint: str = field(default="BAAI/bge-small-en")


@define(kw_only=True)
class LLM(Model):
    """LLM/MLLM model configurations base class.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional
    """

    system_prompt: Optional[str] = field(default=None)
    quantization: Optional[str] = field(
        default="4bit", validator=base_validators.in_(["4bit", "8bit", None])
    )

    def _set_ollama_checkpoint(self):
        """Get ollama compatible checkpoint name."""
        # TODO: Extend model name with quantization
        if self.__class__.__name__ == "OllamaModel":
            return
        if self.__class__.__name__.lower() not in _ollama_mapping:
            raise ValueError(
                f"Could not load {self.__class__.__name__} to be used with OllamaClient. Try passing an OllamaModel to the client, by setting model.checkpoint to an Ollama tag. Check available tags on 'ollama.com/library'"
            )
        self.checkpoint = f"{self.__class__.__name__.lower().replace('_', '.')}"

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {
            "checkpoint": self.checkpoint,
            "quantization": self.quantization,
            "system_prompt": self.system_prompt,
        }


@define(kw_only=True)
class OllamaModel(LLM):
    """An Ollama model that needs to be initialized with an ollama tag as checkpoint.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. For available checkpoints consult [Ollama Models](https://ollama.com/library)
    :type checkpoint: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    llm = OllamaModel(name='ollama1', checkpoint="gemma2:latest")
    ```
    """

    port: Optional[int] = field(default=11434)


@define(kw_only=True)
class TransformersLLM(LLM):
    """An LLM model that needs to be initialized with any LLM checkpoint available on HuggingFace transformers. This model can be used with a roboml client.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "microsoft/Phi-3-mini-4k-instruct". For available checkpoints consult [HuggingFace LLM Models](https://huggingface.co/models?other=LLM)
    :type checkpoint: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    llm = TransformersLLM(name='llm', checkpoint="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ```
    """

    checkpoint: str = field(default="microsoft/Phi-3-mini-4k-instruct")


@define(kw_only=True)
class TransformersMLLM(LLM):
    """An MLLM model that needs to be initialized with any MLLM checkpoint available on HuggingFace transformers. This model can be used with a roboml client.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "HuggingFaceM4/idefics2-8b". For available checkpoints consult [HuggingFace Image-Text to Text Models](https://huggingface.co/models?pipeline_tag=image-text-to-text)
    :type checkpoint: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    mllm = TransformersMLLM(name='mllm', checkpoint="gemma2:latest")
    ```
    """

    checkpoint: str = field(default="HuggingFaceM4/idefics2-8b")


@define(kw_only=True)
class Llama3(TransformersLLM):
    """A pre-trained language model from MetaAI for tasks such as text generation, question answering, and more. [Details](https://llama.meta.com)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "meta-llama/Meta-Llama-3-8B-Instruct". For available checkpoints, consult [LLama3 checkpoints on HuggingFace](https://huggingface.co/models?search="llama3").
    :type checkpoint: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    llama = Llama3(name='llama', checkpoint="other_checkpoint_name")  # Initialize with a custom checkpoint
    ```
    """

    checkpoint: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct")


@define(kw_only=True)
class Llama3_1(TransformersLLM):
    """A pre-trained language model from MetaAI for tasks such as text generation, question answering, and more. [Details](https://llama.meta.com)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "meta-llama/Meta-Llama-3.1-8B-Instruct". For available checkpoints, consult [LLama3 checkpoints on HuggingFace](https://huggingface.co/models?search="llama3").
    :type checkpoint: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    llama = Llama3_1(name='llama', checkpoint="other_checkpoint_name")  # Initialize with a custom checkpoint
    ```
    """

    checkpoint: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")


@define(kw_only=True)
class Idefics2(TransformersMLLM):
    """A pre-trained visual language model from HuggingFace for tasks such as visual question answering. [Details](https://huggingface.co/HuggingFaceM4/idefics2-8b)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "HuggingFaceM4/idefics2-8b". For available checkpoints, consult [Idefics2 checkpoints on HuggingFace](https://huggingface.co/HuggingFaceM4/idefics2-8b).
    :type checkpoint: str
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    idefics = Idefics2(name='mllm1', quantization="8bit")  # Initialize with a custom checkpoint
    ```
    """

    checkpoint: str = field(default="HuggingFaceM4/idefics2-8b")


@define(kw_only=True)
class Llava(TransformersMLLM):
    """LLaVA is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture. [Details](https://llava-vl.github.io/)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "liuhaotian/llava-v1.6-mistral-7b". For available checkpoints, consult [Llava checkpoints on HuggingFace](https://huggingface.co/liuhaotian).
    :type checkpoint: str
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    llava = Llava(name='mllm2', quantization="4bit")
    ```
    """

    checkpoint: str = field(default="liuhaotian/llava-v1.6-mistral-7b")


@define(kw_only=True)
class InstructBlip(TransformersMLLM):
    """An open-source general purpose vision language model by SalesForce built using instruction-tuning. [Details](https://arxiv.org/abs/2305.06500)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "Salesforce/instructblip-vicuna-7b". For available checkpoints, consult [InstructBlip checkpoints on HuggingFace](https://huggingface.co/models?search=instructblip).
    :type checkpoint: str
    :param history_reset_phrase: A phrase used to reset the conversation history. Defaults to "chat reset".
    :type history_reset_phrase: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    blip = InstructBlip(name='mllm3', quantization="4bit")
    ```
    """

    checkpoint: str = field(default="Salesforce/instructblip-vicuna-7b")


@define(kw_only=True)
class Whisper(Model):
    """Whisper is an automatic speech recognition (ASR) system by OpenAI trained on 680,000 hours of multilingual and multitask supervised data collected from the web. [Details](https://openai.com/index/whisper/)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "openai/whisper-small.en". For available checkpoints, consult [Whisper checkpoints on HuggingFace](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013).
    :type checkpoint: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    whisper = Whisper(name='s2t', checkpoint="openai/whisper-medium") # Initialize with a different checkpoint
    ```
    """

    checkpoint: str = field(default="openai/whisper-small.en")
    quantization: Optional[str] = field(
        default="4bit", validator=base_validators.in_(["4bit", "8bit", None])
    )

    def get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {"checkpoint": self.checkpoint, "quantization": self.quantization}


@define(kw_only=True)
class SpeechT5(Model):
    """A model for text-to-speech synthesis developed by Microsoft. [Details](https://github.com/microsoft/SpeechT5)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "microsoft/speecht5_tts".
    :type checkpoint: str
    :param voice: The voice to use for synthesis. Can be one of "awb", "bdl", "clb", "jmk", "ksp", "rms", or "slt". Default is "clb".
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    speecht5 = SpeechT5(name='t2s1', voice="bdl")  # Initialize with a different voice
    ```
    """

    checkpoint: str = field(default="microsoft/speecht5_tts")
    voice: str = field(
        default="clb",
        validator=base_validators.in_([
            "awb",
            "bdl",
            "clb",
            "jmk",
            "ksp",
            "rms",
            "slt",
        ]),
    )

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {"checkpoint": self.checkpoint, "voice": self.voice}


@define(kw_only=True)
class Bark(Model):
    """A model for text-to-speech synthesis developed by SunoAI. [Details](https://github.com/suno-ai/bark)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. [Bark checkpoints on HuggingFace](https://huggingface.co/collections/suno/bark-6502bdd89a612aa33a111bae). Default is "suno/bark-small".
    :type checkpoint: str
    :param attn_implementation: The attention implementation to use for the model. Default is "flash_attention_2".
    :param voice: The voice to use for synthesis. More choices are available [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). Default is "v2/en_speaker_6".
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    bark = Bark(name='t2s2', voice="v2/en_speaker_1")  # Initialize with a different voice
    ```
    """

    checkpoint: str = field(default="suno/bark-small")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    voice: str = field(default="v2/en_speaker_6")

    def get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {
            "checkpoint": self.checkpoint,
            "attn_implementation": self.attn_implementation,
            "voice": self.voice,
        }


@define(kw_only=True)
class VisionModel(Model):
    """Object Detection Model with Optional Tracking.

    This vision model provides a flexible framework for object detection and tracking using the [mmdet framework](https://github.com/open-mmlab/mmdetection). It can be used as a standalone detector or as a tracker to follow detected objects over time. It can be initizaled with any checkpoint available in the mmdet framework.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. [All available checkpoints in the mmdet framework](https://github.com/open-mmlab/mmdetection?tab=readme-ov-file#overview-of-benchmark-and-model-zoo). Default is "dino-4scale_r50_8xb2-12e_coco".
    :type checkpoint: str
    :param cache_dir: The directory where downloaded models are cached. Default is 'mmdet'.
    :type cache_dir: str
    :param setup_trackers: Whether to set up trackers using norfair or not. Default is False.
    :type setup_trackers: bool
    :param tracking_distance_function: The function used to calculate the distance between detected objects. This can be any distance metric string available in [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) Default is "euclidean".
    :type tracking_distance_function: str
    :param tracking_distance_threshold: The threshold for determining whether two object in consecutive frames are considered close enough to be considered the same object. Default is 30, with a minimum value of 1.
    :type tracking_distance_threshold: int
    :param deploy_tensorrt: Deploy the vision model using NVIDIA TensorRT. To utilize this feature with roboml, checkout the instructions [here](https://github.com/automatika-robotics/roboml). Default is False.
    :type deploy_tensorrt: bool
    :param _num_trackers: The number of trackers to use. This number depends on the number of inputs image streams being given to the component. It is set automatically if **setup_trackers** is True.
    :type _num_trackers: int
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    model = DetectionModel(name='detection1', setup_trackers=True, num_trackers=1, tracking_distance_threshold=20)  # Initialize the model for tracking one object
    ```
    """

    checkpoint: str = field(default="dino-4scale_r50_8xb2-12e_coco")
    cache_dir: str = field(default="mmdet")
    setup_trackers: bool = field(default=False)
    tracking_distance_function: str = field(default="euclidean")
    tracking_distance_threshold: int = field(
        default=30, validator=base_validators.gt(0)
    )
    deploy_tensorrt: bool = field(default=False)
    _num_trackers: int = field(default=1, validator=base_validators.gt(0))

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {
            "checkpoint": self.checkpoint,
            "cache_dir": self.cache_dir,
            "setup_trackers": self.setup_trackers,
            "num_trackers": self._num_trackers,
            "tracking_distance_function": self.tracking_distance_function,
            "tracking_distance_threshold": self.tracking_distance_threshold,
            "deploy_tensorrt": self.deploy_tensorrt,
        }
