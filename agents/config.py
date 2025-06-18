from typing import Optional, Union, Dict, List
from pathlib import Path

from attrs import define, field, Factory

from .ros import base_validators, BaseComponentConfig, Topic, Route
from .utils import validate_kwargs_from_default, _LANGUAGE_CODES
from .utils.vision import _MS_COCO_LABELS

__all__ = [
    "LLMConfig",
    "MLLMConfig",
    "SpeechToTextConfig",
    "TextToSpeechConfig",
    "SemanticRouterConfig",
    "MapConfig",
    "VideoMessageMakerConfig",
    "VisionConfig",
]


@define(kw_only=True)
class ModelComponentConfig(BaseComponentConfig):
    warmup: Optional[bool] = field(default=False)

    def get_inference_params(self) -> Dict:
        """Get inference params from model components"""
        return self._get_inference_params()

    def _get_inference_params(self) -> Dict:
        raise NotImplementedError(
            "This method needs to be implemented by model config classes"
        )


@define(kw_only=True)
class LLMConfig(ModelComponentConfig):
    """
    Configuration for the Large Language Model (LLM) component.

    It defines various settings that control how the LLM component operates, including
    whether to enable chat history, retreival augmented generation (RAG) and more.

    :param enable_rag: Enables or disables Retreival Augmented Generation.
    :type enable_rag: bool
    :param collection_name: The name of the vectordb collection to use for RAG.
    :type collection_name: Optional[str]
    :param distance_func: The distance metric used for nearest neighbor search for RAG.
        Supported values are "l2", "ip", and "cosine".
    :type distance_func: str
    :param n_results: The maximum number of results to return for RAG. Defaults to 1.
        For numbers greater than 1, results will be concatenated together in a single string.
    :type n_results: int
    :param chat_history: Whether to include chat history in the LLM's prompt.
    :type chat_history: bool
    :param history_reset_phrase: Phrase to reset chat history. Defaults to 'chat reset'
    :type history_reset_phrase: str
    :param history_size: Number of user messages to keep in chat history. Defaults to 10
    :type history_size: int
    :param temperature: Temperature used for sampling tokens during generation.
        Default is 0.8 and must be greater than 0.0.
    :type temperature: float
    :param max_new_tokens: The maximum number of new tokens to generate.
        Default is 100 and must be greater than 0.
    :type max_new_tokens: int
    :param stream: Publish the llm output as a stream of tokens, useful when sending llm output to a user facing client or to a TTS component. Cannot be used in conjunction with tool calling.
        Default is false
    :type stream: bool
    :param break_character: A string character marking that the output thus far received in a stream should be published. This parameter only takes effect when stream is set to True. As stream output is received token by token, it is useful to publish full sentences instead of individual tokens as the components output (for example, for downstream text to speech conversion). This value can be set to an empty string to publish output token by token.
        Default is '.' (period)
    :type break_character: str
    :param response_terminator: A string token marking that the end of a single response from the model. This token is only used in case of a persistent clients, such as a websocket client and when stream is set to True. It is not published. This value cannot be an empty string.
        Default is '<<Response Ended>>'
    :type response_terminator: str

    Example of usage:
    ```python
    config = LLMConfig(enable_rag=True, collection_name="my_collection", distance_func="l2")
    ```
    """

    enable_rag: bool = field(default=False)
    collection_name: Optional[str] = field(default=None)
    distance_func: str = field(
        default="l2", validator=base_validators.in_(["l2", "ip", "cosine"])
    )
    n_results: int = field(default=1)
    add_metadata: bool = field(default=False)
    chat_history: bool = field(default=False)
    history_reset_phrase: str = field(default="chat reset")
    history_size: int = field(
        default=10, validator=base_validators.gt(4)
    )  # number of user messages
    temperature: float = field(default=0.8, validator=base_validators.gt(0.0))
    max_new_tokens: int = field(default=100, validator=base_validators.gt(0))
    stream: bool = field(default=False)
    break_character: str = field(default=".")
    response_terminator: str = field(default="<<Response Ended>>")
    _system_prompt: Optional[str] = field(default=None, alias="_system_prompt")
    _component_prompt: Optional[Union[str, Path]] = field(
        default=None, alias="_component_prompt"
    )
    _topic_prompts: Dict[str, Union[str, Path]] = field(
        default=Factory(dict), alias="_topic_prompts"
    )
    _tool_descriptions: List[Dict] = field(
        default=Factory(list), alias="_tool_descriptions"
    )
    _tool_response_flags: Dict[str, bool] = field(
        default=Factory(dict), alias="_tool_response_flags"
    )

    @response_terminator.validator
    def _not_empty(self, _, value):
        if not value:
            raise ValueError("response_terminator must not be an empty string")

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "stream": self.stream,
        }


@define(kw_only=True)
class MLLMConfig(LLMConfig):
    """
    Configuration for the Multi-Modal Large Language Model (MLLM) component.

    It defines various settings that control how the LLM component operates, including
    whether to enable chat history, retreival augmented generation (RAG) and more.

    :param enable_rag: Enables or disables Retreival Augmented Generation.
    :type enable_rag: bool
    :param collection_name: The name of the vectordb collection to use for RAG.
    :type collection_name: Optional[str]
    :param distance_func: The distance metric used for nearest neighbor search for RAG.
        Supported values are "l2", "ip", and "cosine".
    :type distance_func: str
    :param n_results: The maximum number of results to return for RAG. Defaults to 1.
        For numbers greater than 1, results will be concatenated together in a single string.
    :type n_results: int
    :param chat_history: Whether to include chat history in the LLM's prompt.
    :type chat_history: bool
    :param history_reset_phrase: Phrase to reset chat history. Defaults to 'chat reset'
    :type history_reset_phrase: str
    :param history_size: Number of user messages to keep in chat history. Defaults to 10
    :type history_size: int
    :param temperature: Temperature used for sampling tokens during generation.
        Default is 0.8 and must be greater than 0.0.
    :type temperature: float
    :param max_new_tokens: The maximum number of new tokens to generate.
        Default is 100 and must be greater than 0.
    :type max_new_tokens: int
    :param stream: Publish the llm output as a stream of tokens, useful when sending llm output to a user facing client or to a TTS component. Cannot be used in conjunction with tool calling.
        Default is false
    :type stream: bool
    :param break_character: A string character marking that the output thus far received in a stream should be published. This parameter only takes effect when stream is set to True. As stream output is received token by token, it is useful to publish full sentences instead of individual tokens as the components output (for example, for downstream text to speech conversion). This value can be set to an empty string to publish output token by token.
        Default is '.' (period)
    :type break_character: str
    :param response_terminator: A string token marking that the end of a single response from the model. This token is only used in case of a persistent clients, such as a websocket client and when stream is set to True. It is not published. This value cannot be an empty string.
        Default is '<<Response Ended>>'
    :type response_terminator: str

    Example of usage:
    ```python
    config = MLLMConfig(enable_rag=True, collection_name="my_collection", distance_func="l2")
    ```
    """

    pass


@define(kw_only=True)
class VisionConfig(ModelComponentConfig):
    """Configuration for a detection component.

       The config allows you to customize the detection and/or tracking process.

       :param threshold: The confidence threshold for object detection, ranging from 0.1 to 1.0 (default: 0.5).
       :type threshold: float
       :param get_dataset_labels: Whether to return data labels along with detections (default: True).
       :type get_dataset_labels: bool
       :param labels_to_track: A list of specific labels to track, when the model is used as a tracker (default: None).
       :type labels_to_track: Optional[list]
    :param enable_visualization: Whether to enable visualization of detections (default: False). Useful for testing vision component output.
       :type enable_visualization: Optional[bool]
       :param enable_local_classifier: Whether to enable a local classifier model for detections (default: False). If a model client is given to the component, than this has no effect.
       :type enable_local_classifier: bool
       :param input_height: Height of the input to local classifier model in pixels (default: 640). This parameter is only effective when enable_local_classifier is set to True.
       :type input_height: int
       :param input_width: Width of the input to local classifier in pixels (default: 640). This parameter is only effective when enable_local_classifier is set to True.
       :type input_width: int
       :param dataset_labels: A dictionary mapping label indices to names, used to interpret model outputs (default: COCO labels). This parameter is only effective when enable_local_classifier is set to True.
       :type dataset_labels: Dict
       :param device_local_classifier: Device to run the local classifier on, either "cpu" or "gpu" (default: "gpu"). This parameter is only effective when enable_local_classifier is set to True.
       :type device_local_classifier: str
       :param ncpu_local_classifier: Number of CPU cores to allocate to the local classifier when using CPU (default: 1). This parameter is only effective when enable_local_classifier is set to True.
       :type ncpu_local_classifier: int
       :param local_classifier_model_path: Path or URL to the ONNX model used by the local classifier (default: DEIM, Huang et al. CVPR 2025). Other models based on [DEIM](https://github.com/ShihuaHuang95/DEIM?tab=readme-ov-file#deim-d-fine) can be checked [here](https://github.com/automatika-robotics/embodied-agents/releases/tag/0.3.3). This parameter is only effective when enable_local_classifier is set to True.
       :type local_classifier_model_path: str

       Example of usage:
       ```python
       config = DetectionConfig(threshold=0.3)
       ```
    """

    threshold: float = field(
        default=0.5, validator=base_validators.in_range(min_value=0.1, max_value=1.0)
    )
    get_dataset_labels: bool = field(default=True)
    labels_to_track: Optional[List[str]] = field(default=None)
    enable_visualization: Optional[bool] = field(default=False)
    enable_local_classifier: bool = field(default=False)
    input_height: int = field(default=640)
    input_width: int = field(default=640)
    dataset_labels: Dict = field(default=_MS_COCO_LABELS)
    device_local_classifier: str = field(
        default="cuda", validator=base_validators.in_(["cpu", "cuda", "tensorrt"])
    )
    ncpu_local_classifier: int = field(default=1)
    local_classifier_model_path: str = field(
        default="https://github.com/automatika-robotics/embodied-agents/releases/download/0.3.3/deim_dfine_hgnetv2_n_coco_160e.onnx"
    )

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "threshold": self.threshold,
            "get_dataset_labels": self.get_dataset_labels,
            "labels_to_track": self.labels_to_track,
        }


@define(kw_only=True)
class TextToSpeechConfig(ModelComponentConfig):
    """Configuration for a Text-To-Speech component.

    This class defines the configuration options for a Text-To-Speech component.

    :param play_on_device: Whether to play the audio on available audio device (default: False).
    :type play_on_device: bool
    :param device: Optional device id (int) for playing the audio. Only effective if play_on_device is True (default: None).
    :type play_on_device: bool
    :param buffer_size: Size of the buffer for playing audio on device. Only effective if play_on_device is True (default: 20).
    :type buffer_size: int
    :param block_size: Size of the audio block to be read for playing audio on device. Only effective if play_on_device is True (default: 1024).
    :type block_size: int
    :param thread_shutdown_timeout: Timeout to shutdown a playback thread, if data is not received for more than a certain number of seconds. Only effective if play_on_device is True (default: 5 seconds).
    :type thread_shutdown_timeout: int
    :param stream: Stram output when used with WebSocketClient. Useful when model output is large and broken into chunks by the server. (default: True).
    :type thread_shutdown_timeout: int


    Example of usage:
    ```python
    config = TextToSpeechConfig(play_on_device=True, get_bytes=False)
    ```
    """

    play_on_device: bool = field(default=False)
    device: Optional[int] = field(default=None)
    buffer_size: int = field(default=20)
    block_size: int = field(default=1024)
    thread_shutdown_timeout: int = field(default=5)
    stream: bool = field(default=True)
    _get_bytes: bool = field(default=False, alias="_get_bytes")

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {"get_bytes": self._get_bytes}


@define(kw_only=True)
class SpeechToTextConfig(ModelComponentConfig):
    """
    Configuration for a Speech-To-Text component.

    This class defines the configuration options for speech transcription, voice activity detection,
    wakeword detection, and audio streaming.

    --------------------
    Transcription
    --------------------
    :param initial_prompt: Optional initial prompt to guide transcription (e.g. speaker name or topic).
                           Defaults to None.
    :type initial_prompt: str or None

    :param language: Language code for transcription (e.g. "en", "zh"). Must be one of the supported language codes.
                     Defaults to "en".
    :type language: str

    :param max_new_tokens: Maximum number of tokens to generate. If None, no limit is applied.
                           Defaults to None.
    :type max_new_tokens: int or None

    --------------------
    Voice Activity Detection (VAD)
    --------------------
    :param enable_vad: Enable VAD to detect when speech is present in audio input.
                       Requires onnxruntime and silero-vad model.
                       Defaults to False.
    :type enable_vad: bool

    :param device_audio: Audio input device ID. Only used if `enable_vad` is True.
                         Defaults to None.
    :type device_audio: Optional[int]

    :param vad_threshold: Threshold above which speech is considered present.
                          Only used if `enable_vad` is True. Range: 0.0–1.0.
                          Defaults to 0.5.
    :type vad_threshold: float

    :param min_silence_duration_ms: Minimum silence duration (ms) before it's treated as a pause.
                                    Only used if `enable_vad` is True.
                                    Defaults to 300.
    :type min_silence_duration_ms: int

    :param speech_pad_ms: Silence padding (ms) added to start and end of detected speech regions.
                          Only used if `enable_vad` is True.
                          Defaults to 30.
    :type speech_pad_ms: int

    :param speech_buffer_max_len: Max length of speech buffer in ms.
                                  Only used if `enable_vad` is True.
                                  Defaults to 30000.
    :type speech_buffer_max_len: int

    :param device_vad: Device for VAD ('cpu' or 'gpu').
                       Only used if `enable_vad` is True.
                       Defaults to 'cpu'.
    :type device_vad: str

    :param ncpu_vad: Number of CPU cores to use for VAD (if `device_vad` is 'cpu').
                     Defaults to 1.
    :type ncpu_vad: int

    --------------------
    Wakeword Detection
    --------------------
    :param enable_wakeword: Enable detection of a wakeword phrase (e.g. 'Hey Jarvis').
                            Requires `enable_vad` to be True.
                            Defaults to False.
    :type enable_wakeword: bool

    :param wakeword_threshold: Minimum confidence score to trigger wakeword detection.
                               Only used if `enable_wakeword` is True.
                               Defaults to 0.6.
    :type wakeword_threshold: float

    :param device_wakeword: Device for Wakeword Detection ('cpu' or 'gpu').
                             Only used if `enable_wakeword` is True.
                             Defaults to 'cpu'.
    :type device_wakeword: str

    :param ncpu_wakeword: Number of CPU cores for Wakeword Detection (if `device_wakeword` is 'cpu').
                          Defaults to 1.
    :type ncpu_wakeword: int

    --------------------
    Streaming
    --------------------
    :param stream: Send audio as a stream to a persistent client (e.g., websockets).
                   Requires `enable_vad` to be True.
                   Useful for real-time transcription.
                   Defaults to False.
    :type stream: bool

    :param min_chunk_size: Audio chunk size in ms to send when streaming.
                       Requires `stream` to be True. Must be > 100 ms.
                       Defaults to 2000.
    :type min_chunk_size: int

    --------------------
    Model Paths
    --------------------
    :param vad_model_path: Path or URL to VAD ONNX model.
                           Defaults to the Silero VAD model URL.
    :type vad_model_path: str

    :param melspectrogram_model_path: Path or URL to melspectrogram model used in wakeword detection.
                                      Defaults to openWakeWord model URL.
    :type melspectrogram_model_path: str

    :param embedding_model_path: Path or URL to audio embedding model for wakeword detection.
                                 Defaults to openWakeWord model URL.
    :type embedding_model_path: str

    :param wakeword_model_path: Path or URL to wakeword ONNX model (e.g. 'Hey Jarvis').
                                Defaults to a pretrained openWakeWord model.
                                For custom models, see:
                                https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
    :type wakeword_model_path: str

    --------------------
    Example
    --------------------
    Example usage:
    ```python
    config = SpeechToTextConfig(
        enable_vad=True,
        enable_wakeword=True,
        vad_threshold=0.5,
        wakeword_threshold=0.6,
        min_silence_duration_ms=1000,
        speech_pad_ms=30,
        speech_buffer_max_len=8000,
    )
    ```
    """

    initial_prompt: Optional[str] = field(default=None)
    language: str = field(
        default="en",
        validator=base_validators.in_(_LANGUAGE_CODES),
    )
    max_new_tokens: Optional[int] = field(default=None)
    enable_vad: bool = field(default=False)
    enable_wakeword: bool = field(default=False)
    device_audio: Optional[int] = field(default=None)
    vad_threshold: float = field(
        default=0.5, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    wakeword_threshold: float = field(
        default=0.6, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    min_silence_duration_ms: int = field(default=500)
    speech_pad_ms: int = field(default=30)
    speech_buffer_max_len: int = field(default=30000)
    stream: bool = field(default=False)
    min_chunk_size: int = field(default=2000, validator=base_validators.gt(500))
    device_vad: str = field(
        default="cpu", validator=base_validators.in_(["cpu", "cuda", "tensorrt"])
    )
    device_wakeword: str = field(
        default="cpu", validator=base_validators.in_(["cpu", "cuda", "tensorrt"])
    )
    ncpu_vad: int = field(default=1)
    ncpu_wakeword: int = field(default=1)
    vad_model_path: str = field(
        default="https://raw.githubusercontent.com/snakers4/silero-vad/refs/heads/master/src/silero_vad/data/silero_vad.onnx"
    )
    melspectrogram_model_path: str = field(
        default="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
    )
    embedding_model_path: str = field(
        default="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
    )
    wakeword_model_path: str = field(
        default="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx"
    )
    _sample_rate: int = field(default=16000, alias="_sample_rate")
    _block_size: int = field(default=1280, alias="_block_size")
    _vad_filter: bool = field(init=False, alias="_vad_filter")
    _word_timestamps: bool = field(init=False, alias="_word_timestamps")

    @enable_wakeword.validator
    def _check_wakeword(self, _, value):
        """Wakeword validator"""
        if value and not self.enable_vad:
            raise ValueError(
                "enable_vad (voice activity detection) must be set to True when enable_wakeword is set to True"
            )

    @stream.validator
    def _check_stream(self, _, value):
        """Stream validator"""
        if value and not self.enable_vad:
            raise ValueError(
                "enable_vad (voice activity detection) must be set to True when stream is set to True"
            )

    def __attrs_post_init__(self):
        """Set values of undefined privates"""
        self._word_timestamps = self.stream
        self._vad_filter = not self.enable_vad

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "language": self.language,
            "initial_prompt": self.initial_prompt,
            "max_new_tokens": self.max_new_tokens,
            "word_timestamps": self._word_timestamps,
            "vad_filter": self._vad_filter,
        }


def _get_optional_topic(topic: Union[Topic, Dict]) -> Optional[Topic]:
    if not topic:
        return
    if isinstance(topic, Topic):
        return topic
    return Topic(**topic)


@define(kw_only=True)
class MapConfig(BaseComponentConfig):
    """Configuration for a MapEncoding component.

    :param map_name: The name of the map.
    :type map_name: str
    :param distance_func: The function used to calculate distance when retreiving information from the map collection. Can be one of "l2" (L2 distance), "ip" (Inner Product), or "cosine" (Cosine similarity). Default is "l2".
    :type distance_func: str

    Example of usage:
    ```python
    config = MapConfig(map_name="my_map", distance_func="ip")
    ```
    """

    map_name: str = field()
    distance_func: str = field(
        default="l2", validator=base_validators.in_(["l2", "ip", "cosine"])
    )
    _position: Optional[Union[Topic, Dict]] = field(
        default=None, converter=_get_optional_topic, alias="_position"
    )
    _map_topic: Optional[Union[Topic, Dict]] = field(
        default=None, converter=_get_optional_topic, alias="_map_topic"
    )


def _get_optional_route(route: Union[Route, Dict]) -> Optional[Route]:
    if not route:
        return
    if isinstance(route, Route):
        return route
    return Route(**route)


@define(kw_only=True)
class SemanticRouterConfig(BaseComponentConfig):
    """Configuration parameters for a semantic router component.

    :param router_name: The name of the router.
    :type router_name: str
    :param distance_func: The function used to calculate distance from route samples in vectordb. Can be one of "l2" (L2 distance), "ip" (Inner Product), or "cosine" (Cosine similarity). Default is "l2".
    :type distance_func: str
    :param maximum_distance: The maximum distance threshold for routing. A value between 0.1 and 1.0. Defaults to 0.4
    :type maximum_distance: float

    Example of usage:
    ```python
    config = SemanticRouterConfig(router_name="my_router")
    # or
    config = SemanticRouterConfig(router_name="my_router", distance_func="ip", maximum_distance=0.7)
    ```
    """

    router_name: str = field()
    distance_func: str = field(
        default="l2", validator=base_validators.in_(["l2", "ip", "cosine"])
    )
    maximum_distance: float = field(
        default=0.4, validator=base_validators.in_range(min_value=0.1, max_value=1.0)
    )
    _default_route: Optional[Union[Route, Dict]] = field(
        default=None, converter=_get_optional_route, alias="_default_route"
    )


@define(kw_only=True)
class VideoMessageMakerConfig(BaseComponentConfig):
    """Configuration parameters for a video message maker component.

    :param min_video_frames: The minimum number of frames in a video segment. Default is 15, assuming a 0.5 second video at 30 fps.
    :type min_video_frames: int
    :param max_video_frames: The maximum number of frames in a video segment. Default is 600, assuming a 20 second video at 30 fps.
    :type max_video_frames: int
    :param motion_estimation_func: The function used for motion estimation. Can be one of "frame_difference" or "optical_flow". Default is None.
    :type motion_estimation_func: Optional[str]
    :param threshold: The threshold value for motion detection. A float between 0.1 and 5.0. Default is 0.3.
    :type threshold: float
    :param flow_kwargs: Additional keyword arguments for the optical flow algorithm. Default is a dictionary with reasonable values.

    Example of usage:
    ```python
    config = VideoMessageMakerConfig()
    # or
    config = VideoMessageMakerConfig(min_video_frames=30, motion_estimation_func="optical_flow", threshold=0.5)
    ```
    """

    min_video_frames: int = field(default=15)  # assuming 0.5 second video at 30 fps
    max_video_frames: int = field(default=600)  # assuming 20 second video at 30 fps
    motion_estimation_func: Optional[str] = field(
        default=None,
        validator=base_validators.in_(["frame_difference", "optical_flow"]),
    )
    threshold: float = field(
        default=0.3, validator=base_validators.in_range(min_value=0.1, max_value=5.0)
    )
    flow_kwargs: Dict = field(
        default={
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.1,
            "flags": 0,
        },
        validator=validate_kwargs_from_default,
    )
