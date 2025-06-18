from typing import Any, Union, Optional, List, Dict
import queue
import threading
import numpy as np
from collections import deque

from ..clients.model_base import ModelClient
from ..config import SpeechToTextConfig
from ..ros import Audio, String, Topic
from ..utils import validate_func_args, VADStatus, WakeWordStatus, load_model
from .model_component import ModelComponent
from .component_base import ComponentRunType


class SpeechToText(ModelComponent):
    """
    This component takes in audio input and outputs a text representation of the audio using Speech-to-Text models (e.g. Whisper).

    :param inputs: The input topics for the STT.
        This should be a list of Topic objects, limited to Audio type.
    :type inputs: list[Topic]
    :param outputs: The output topics for the STT.
        This should be a list of Topic objects, String type is handled automatically.
    :type outputs: list[Topic]
    :param model_client: The model client for the STT.
        This should be an instance of ModelClient.
    :type model_client: ModelClient
    :param config: The configuration for the STT.
        This should be an instance of SpeechToTextConfig. If not provided, defaults to SpeechToTextConfig().
    :type config: Optional[SpeechToTextConfig]
    :param trigger: The trigger value or topic for the STT.
        This can be a single Topic object, a list of Topic objects.
    :type trigger: Union[Topic, list[Topic], float]
    :param callback_group: An optional callback group for the STT.
        If provided, this should be a string. Otherwise, it defaults to None.
    :type callback_group: str
    :param component_name: The name of the STT component.
        This should be a string and defaults to "speechtotext_component".
    :type component_name: str

    Example usage:
    ```python
    audio_topic = Topic(name="audio", msg_type="Audio")
    text_topic = Topic(name="text", msg_type="String")
    config = SpeechToTextConfig(enable_vad=True)
    model = Whisper(name="whisper")
    model_client = ModelClient(model=model)
    stt_component = SpeechToText(
        inputs=[audio_topic],
        outputs=[text_topic],
        model_client=model_client,
        config=config,
        component_name='stt_component'
    )
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Topic],
        outputs: List[Topic],
        model_client: ModelClient,
        config: Optional[SpeechToTextConfig] = None,
        trigger: Union[Topic, List[Topic]],
        component_name: str,
        callback_group=None,
        **kwargs,
    ):
        self.config: SpeechToTextConfig = config or SpeechToTextConfig()
        self.allowed_inputs = {"Required": [Audio]}
        self.handled_outputs = [String]

        if isinstance(trigger, float):
            raise TypeError(
                "SpeechToText component cannot be started as a timed component"
            )

        super().__init__(
            inputs,
            outputs,
            model_client,
            self.config,
            trigger,
            callback_group,
            component_name,
            **kwargs,
        )

    def custom_on_activate(self):
        """Custom activation"""
        # NOTE: Custom activate to ensure creation of separate thread if VAD is enabled
        # happens after activation as VAD starts sending received voice to execution
        # step right away

        # Activate component
        super().custom_on_activate()

        # If VAD is enabled, start a listening stream on a separate thread
        if self.config.enable_vad:
            from ..utils.voice import VADIterator

            self.event = threading.Event()
            self.queue = queue.Queue()
            self.vad_iterator = VADIterator(
                model_path=load_model("silero_vad", self.config.vad_model_path),
                threshold=self.config.vad_threshold,
                sample_rate=self.config._sample_rate,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms,
                ncpu=self.config.ncpu_vad,
                device=self.config.device_vad,
            )
            self.speech_buffer = deque(
                maxlen=int(
                    self.config._sample_rate
                    * self.config.speech_buffer_max_len
                    / (1000 * self.config._block_size)
                )
            )
            if self.config.enable_wakeword:
                from ..utils.voice import WakeWord, AudioFeatures

                self.audio_features = AudioFeatures(
                    melspectogram_model_path=load_model(
                        "melspec", self.config.melspectrogram_model_path
                    ),
                    embedding_model_path=load_model(
                        "voice_embeddings", self.config.embedding_model_path
                    ),
                    ncpu=self.config.ncpu_wakeword,
                    device=self.config.device_wakeword,
                )
                self.wake_word = WakeWord(
                    model_path=load_model(
                        "hey_jarvis", self.config.wakeword_model_path
                    ),
                    threshold=self.config.wakeword_threshold,
                    ncpu=self.config.ncpu_wakeword,
                    device=self.config.device_wakeword,
                )
                self.wake_word_triggered = False
            self.listening_thread = threading.Thread(target=self._process_audio).start()

    def custom_on_deactivate(self):
        # If VAD is enabled, stop the listening stream thread
        if self.config.enable_vad:
            self.event.set()
            if self.listening_thread:
                self.listening_thread.join()

        # Deactivate component
        super().custom_on_deactivate()

    def _create_input(self, *_, **kwargs) -> Optional[Dict[str, Any]]:
        """Create inference input for SpeechToText models
        :param args:
        :param kwargs:
        :rtype: dict[str, Any]
        """

        if self.config.enable_vad and kwargs.get("speech") is not None:
            query = kwargs["speech"]
        else:
            # set query as trigger
            trigger = kwargs.get("topic")
            if not trigger:
                return None
            query = self.trig_callbacks[trigger.name].get_output()
            if query is None or len(query) == 0:
                return None

        return {"query": query, **self.config._get_inference_params()}

    def _stream_callback(
        self, indata: bytes, frames: int, _, status
    ) -> tuple[bytes, int]:
        """Stream callback function for processing audio

        :param indata:
        :type indata: np.ndarray
        :param frames:
        :type frames: int
        :param _:
        :param status:
        :type status: sd.CallbackFlags
        :rtype: None
        """
        assert frames == self.config._block_size
        if status:
            self.get_logger().warn(f"Status: {status}")
        try:
            import pyaudio
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "enable_vad configuration for SpeechToText component requires pyaudio module to be installed. Please install it with `pip install pyaudio`"
            ) from e
        np_frames = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
        vad_output = self.vad_iterator(np_frames)

        # if wake word is enabled then store speech when its triggered
        if self.config.enable_wakeword:
            # create audio embeddings for wakeword classifier
            self.audio_features(np_frames)
            if self.wake_word_triggered:
                self.speech_buffer.append(indata)
        # otherwise store speech when vad is triggered
        elif self.vad_iterator.triggered:
            self.speech_buffer.append(indata)

        # add vad status outputs to queue
        if vad_output:
            self.queue.put_nowait(vad_output)

        return indata, pyaudio.paContinue

    def _process_audio(self) -> None:
        """Creates a stream to process audio from device"""

        # clear event and queue
        self.queue.queue.clear()
        self.event.clear()

        try:
            import pyaudio
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "enable_vad configuration for SpeechToText component requires pyaudio module to be installed. Please install it with `pip install pyaudio`"
            ) from e
        # Create an interface to PortAudio
        audio_interface = pyaudio.PyAudio()

        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config._sample_rate,
            frames_per_buffer=self.config._block_size,
            input=True,
            start=True,
            stream_callback=self._stream_callback,  # type: ignore
        )

        while True:
            vad_output = self.queue.get()
            if vad_output is VADStatus.START:
                # When someone starts speaking, check for wakeword if enabled
                self.get_logger().debug("Speech started")
                if self.config.enable_wakeword:
                    self.wake_word(
                        self.audio_features.get_embeddings(self.wake_word.model_input)
                    )
            elif vad_output is VADStatus.ONGOING:
                self.get_logger().debug("Speech ongoing")
                if self.config.enable_wakeword:
                    wake_status = self.wake_word(
                        self.audio_features.get_embeddings(self.wake_word.model_input)
                    )
                    if wake_status is WakeWordStatus.END:
                        self.get_logger().debug("Wakeword ended")
                        self.wake_word_triggered = True
            elif vad_output is VADStatus.END:
                # Send audio when speech finishes
                self.get_logger().debug("Speech ended")
                if self.config.enable_wakeword:
                    self.wake_word_triggered = False
                self._execution_step(speech=self.speech_buffer)
                self.speech_buffer.clear()
            if self.event.is_set():
                stream.stop_stream()
                stream.close()
                audio_interface.terminate()
                break
        self.event.wait()

    def _execution_step(self, *args, **kwargs):
        """_execution_step.

        :param args:
        :param kwargs:
        """
        # Check for direct speech before checking for triggers
        if self.config.enable_vad and kwargs.get("speech"):
            self.get_logger().debug(
                f"Received speech from speech thread: {len(kwargs['speech'])}"
            )
            kwargs["speech"] = b"".join(kwargs["speech"])
        elif self.run_type is ComponentRunType.EVENT:
            trigger = kwargs.get("topic")
            if not trigger:
                return
            self.get_logger().debug(f"Received trigger on topic {trigger.name}")
        else:
            time_stamp = self.get_ros_time().sec
            self.get_logger().debug(f"Sending at {time_stamp}")

        # create inference input
        inference_input = self._create_input(*args, **kwargs)
        # call model inference
        if not inference_input:
            self.get_logger().warning("Input not received, not calling model inference")
            return

        # conduct inference
        if self.model_client:
            result = self.model_client.inference(inference_input)
            if result:
                # publish inference result
                if self.publishers_dict:
                    for publisher in self.publishers_dict.values():
                        publisher.publish(**result)
            else:
                # raise a fallback trigger via health status
                self.health_status.set_failure()

    def _warmup(self):
        """Warm up and stat check"""
        import time
        from pathlib import Path

        with open(
            str(Path(__file__).parents[1] / Path("resources/test.wav")), "rb"
        ) as file:
            file_bytes = file.read()

        inference_input = {"query": file_bytes, **self.config._get_inference_params()}

        # Run inference once to warm up and once to measure time
        self.model_client.inference(inference_input)

        start_time = time.time()
        result = self.model_client.inference(inference_input)
        elapsed_time = time.time() - start_time

        self.get_logger().warning(f"Model Output: {result}")
        self.get_logger().warning(f"Approximate Inference time: {elapsed_time} seconds")
        self.get_logger().warning(
            f"RTF: {elapsed_time / 2}"  # audio length, 2 seconds
        )
