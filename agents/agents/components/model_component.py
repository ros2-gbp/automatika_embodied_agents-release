from abc import abstractmethod
import inspect
import json
from typing import Any, Optional, Sequence, Union, List, Dict

from ..clients.model_base import ModelClient
from ..config import ModelComponentConfig
from ..ros import FixedInput, Topic, SupportedType
from .component_base import Component


class ModelComponent(Component):
    """ModelComponent."""

    def __init__(
        self,
        inputs: Optional[Sequence[Union[Topic, FixedInput]]] = None,
        outputs: Optional[Sequence[Topic]] = None,
        model_client: Optional[ModelClient] = None,
        config: Optional[ModelComponentConfig] = None,
        trigger: Union[Topic, List[Topic], float] = 1.0,
        callback_group=None,
        component_name: str = "model_component",
        **kwargs,
    ):
        # setup model client
        self.model_client = model_client if model_client else None

        self.handled_outputs: List[type[SupportedType]]

        if not config:
            self.config = ModelComponentConfig()

        # Initialize Component
        super().__init__(
            inputs,
            outputs,
            config,
            trigger,
            callback_group,
            component_name,
            **kwargs,
        )

    def custom_on_configure(self):
        """
        Create model client if provided and initialize model.
        """
        self.get_logger().debug(f"Current Status: {self.health_status.value}")

        # validate output topics if handled_outputs exist
        self.get_logger().info("Validating Model Component Output Topics")
        self._validate_output_topics()

        # Initialize model
        if self.model_client:
            self.model_client.check_connection()
            self.model_client.initialize()
            if self.config.warmup:
                try:
                    self._warmup()
                except Exception as e:
                    self.get_logger().error(f"Error encountered in warmup: {e}")

    def custom_on_deactivate(self):
        """
        Destroy model client if it exists
        """
        # Deinitialize model
        if self.model_client:
            self.model_client.check_connection()
            self.model_client.deinitialize()

    def _validate_output_topics(self) -> None:
        """
        Verify that output topics that are not handled, have pre-processing functions provided. We just check that there is a pre-processing function and do not check whether the functions have output of the corresponding type.
        """

        if hasattr(self, "publishers_dict") and hasattr(self, "handled_outputs"):
            for name, pub in self.publishers_dict.items():
                if pub.output_topic.msg_type not in self.handled_outputs and (
                    not self._external_processors
                ):
                    func_body = inspect.getsource(pub.output_topic.msg_type.convert)
                    raise TypeError(f"""{type(self).__name__} components can only handle output topics of type(s) {self.handled_outputs} automatically. Topic {name} is of type {pub.output_topic.msg_type}. EITHER provide a pre-processing function for this topic and attach it to the topic by calling the `add_publisher_preprocessor` on the component {self.node_name} OR provide a tool call that can provide structured inference output and attach it by calling `register_tool` on {self.node_name}. Make sure the output can be passed as parameter `output` to the following function:
{func_body}""")

    @property
    def warmup(self) -> bool:
        """Enable warmup of the model."""
        return self.config.warmup

    @warmup.setter
    def warmup(self, value: bool) -> None:
        """Enable warmup of the model."""
        self.config.warmup = value

    @abstractmethod
    def _create_input(self, *args, **kwargs) -> Union[Dict[str, Any], None]:
        """_create_input.

        :param args:
        :param kwargs:
        :rtype: dict[str, Any] | None
        """
        raise NotImplementedError(
            "_create_input method needs to be implemented by child components."
        )

    @abstractmethod
    def _execution_step(self, *args, **kwargs):
        """_execution_step.

        :param args:
        :param kwargs:
        """
        raise NotImplementedError(
            "_execution_step method needs to be implemented by child components."
        )

    @abstractmethod
    def _warmup(self, *args, **kwargs):
        """_warmup.

        :param args:
        :param kwargs:
        """
        raise NotImplementedError(
            "_warmup method needs to be implemented by child components."
        )

    def _update_cmd_args_list(self):
        """
        Update launch command arguments
        """
        super()._update_cmd_args_list()

        self.launch_cmd_args = [
            "--model_client",
            self._get_model_client_json(),
        ]

    def _get_model_client_json(self) -> Union[str, bytes, bytearray]:
        """
        Serialize component routes to json

        :return: Serialized inputs
        :rtype:  str | bytes | bytearray
        """
        if not self.model_client:
            return ""
        return json.dumps(self.model_client.serialize())
