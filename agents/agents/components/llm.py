import json
from pathlib import Path
from typing import Any, Optional, Union, Callable, List, Dict
import msgpack
import msgpack_numpy as m_pack

from ..callbacks import TextCallback
from ..clients.db_base import DBClient
from ..clients.model_base import ModelClient
from ..clients import OllamaClient
from ..config import LLMConfig
from ..ros import FixedInput, String, Topic, Detections
from ..utils import get_prompt_template, validate_func_args
from .model_component import ModelComponent
from .component_base import ComponentRunType

# patch msgpack for numpy arrays
m_pack.patch()


class LLM(ModelComponent):
    """
    This component utilizes large language models (e.g LLama) that can be used to process text data.

    :param inputs: The input topics or fixed inputs for the LLM component.
        This should be a list of Topic objects or FixedInput instances.
    :type inputs: list[Topic | FixedInput]
    :param outputs: The output topics for the LLM component.
        This should be a list of Topic objects. String type is handled automatically.
    :type outputs: list[Topic]
    :param model_client: The model client for the LLM component.
        This should be an instance of ModelClient.
    :type model_client: ModelClient
    :param config: The configuration for the LLM component.
        This should be an instance of LLMConfig. If not provided, defaults to LLMConfig().
    :type config: LLMConfig
    :param db_client: An optional database client for the LLM component.
        If provided, this should be an instance of DBClient. Otherwise, it defaults to None.
    :type db_client: Optional[DBClient]
    :param trigger: The trigger value or topic for the LLM component.
        This can be a single Topic object, a list of Topic objects, or a float value for a timed component. Defaults to 1.
    :type trigger: Union[Topic, list[Topic], float]
    :param callback_group: An optional callback group for the LLM component.
        If provided, this should be a string. Otherwise, it defaults to None.
    :type callback_group: str
    :param component_name: The name of the LLM component.
        This should be a string and defaults to "llm_component".
    :type component_name: str
    :param kwargs: Additional keyword arguments for the LLM.

    Example usage:
    ```python
    text0 = Topic(name="text0", msg_type="String")
    text1 = Topic(name="text1", msg_type="String")
    config = LLMConfig()
    model = Llama3(name='llama')
    model_client = ModelClient(model=model)
    llm_component = LLM(inputs=[text0],
                        outputs=[text1],
                        model_client=model_client,
                        config=config,
                        component_name='llama_component')
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Union[Topic, FixedInput]],
        outputs: List[Topic],
        model_client: ModelClient,
        config: Optional[LLMConfig] = None,
        db_client: Optional[DBClient] = None,
        trigger: Union[Topic, List[Topic], float] = 1.0,
        component_name: str,
        callback_group=None,
        **kwargs,
    ):
        self.config: LLMConfig = config or LLMConfig()
        # set allowed inputs/outputs when parenting multimodal LLMs
        self.allowed_inputs = (
            kwargs["allowed_inputs"]
            if kwargs.get("allowed_inputs")
            else {"Required": [String], "Optional": [Detections]}
        )
        self.handled_outputs = [String]

        self.model_client = model_client

        self.db_client = db_client if db_client else None

        self.component_prompt = (
            get_prompt_template(self.config._component_prompt)
            if self.config._component_prompt
            else None
        )
        self.messages: List[Dict] = []

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

    def custom_on_configure(self):
        # configure the rest
        super().custom_on_configure()

        # add component prompt if set after init
        self.component_prompt = (
            get_prompt_template(self.config._component_prompt)
            if self.config._component_prompt
            else None
        )
        # add topic templates
        if self.config._topic_prompts:
            for topic_name, template in self.config._topic_prompts.items():
                callback = self.callbacks[topic_name]
                if isinstance(callback, TextCallback):
                    callback._template = get_prompt_template(template)

        # initialize db client
        if self.db_client:
            self.db_client.check_connection()
            self.db_client.initialize()

    def custom_on_deactivate(self):
        # deactivate db client
        if self.db_client:
            self.db_client.check_connection()
            self.db_client.deinitialize()

        # deactivate the rest
        super().custom_on_deactivate()

    @validate_func_args
    def add_documents(
        self, ids: List[str], metadatas: List[Dict], documents: List[str]
    ) -> None:
        """Add documents to vector DB for Retreival Augmented Generation (RAG).

        ```{important}
        Documents can be provided after parsing them using a document parser. Checkout various document parsers, available in packages like [langchain_community](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/document_loaders/parsers)
        ```

        :param ids: List of unique string ids for each document
        :type ids: list[str]
        :param metadatas: List of metadata dicts for each document
        :type metadatas: list[dict]
        :param documents: List of documents which are to be store in the vector DB
        :type documents: list[str]
        :rtype: None
        """

        if not self.db_client:
            raise AttributeError(
                "db_client needs to be set in component for add_documents to work"
            )

        db_input = {
            "collection_name": self.config.collection_name,
            "distance_func": self.config.distance_func,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        self.db_client.add(db_input)

    def _handle_rag_query(self, query: str) -> Optional[str]:
        """Internal handler for retreiving documents for RAG.
        :param query:
        :type query: str
        :rtype: str | None
        """
        if not self.db_client:
            raise AttributeError(
                "db_client needs to be set in component for RAG to work"
            )
        # get documents
        db_input = {
            "collection_name": self.config.collection_name,
            "query": query,
            "n_results": self.config.n_results,
        }
        result = self.db_client.query(db_input)
        if result:
            # add metadata as string if asked
            rag_docs = (
                "\n".join(
                    f"{str(meta)}, {doc}"
                    for meta, doc in zip(
                        result["output"]["metadatas"], result["output"]["documents"]
                    )
                )
                if self.config.add_metadata
                else "\n".join(doc for doc in result["output"]["documents"])
            )
            return rag_docs

    def _handle_chat_history(self, message: Dict) -> None:
        """Internal handler for chat history"""
        if not self.config.chat_history:
            self.messages = [message]
        else:
            self.messages.append(message)

            # if the size of history exceeds specified size than take out first
            # two messages
            if len(self.messages) / 2 > self.config.history_size:
                self.messages = self.messages[2:]

    def _handle_tool_calls(self, result: Dict) -> Optional[Dict]:
        """Internal handler for tool calling"""
        if not result.get("tool_calls"):
            self.get_logger().warning(
                "Tools have been provided but no tool calls found in model response."
            )
            return result

        response_flags = []

        # make tool calls
        for tool in result["tool_calls"]:
            function_to_call = self._external_processors[tool["function"]["name"]][0][0]

            try:
                # HACK: Read function argument as serialized datatypes
                # if they are returned as string
                arg_json = {
                    key: json.loads(str(arg))
                    for key, arg in tool["function"]["arguments"].items()
                }
                if isinstance(function_to_call, Callable):
                    function_response = function_to_call(**arg_json)
                else:
                    payload = msgpack.packb(arg_json)
                    if payload:
                        function_to_call.sendall(payload)
                    else:
                        raise Exception(
                            f"Could not serialize the following function arguments for tool calling: {arg_json}"
                        )

                    result_b = function_to_call.recv(1024)
                    function_response = msgpack.unpackb(result_b)
            except Exception as e:
                self.get_logger().error(f"Exception in tool calling. {e}")
                return result

            # make last function call output the publishable output
            result["output"] = function_response

            # Add function response to the messages
            self.messages.append({"role": "tool", "content": function_response})

            # check for response flags
            response_flags.append(
                self.config._tool_response_flags[tool["function"]["name"]]
            )

        # make call to model again if any tool requires response to be sent back
        if any(response_flags):
            self.get_logger().debug(f"Input from component: {self.messages}")

            input = {
                "query": self.messages,
                **self.config._get_inference_params(),
            }
            return self.model_client.inference(input)

        else:
            # return result with its output set to last function response
            return result

    def _create_input(self, *_, **kwargs) -> Optional[Dict[str, Any]]:
        """Create inference input for LLM models
        :param args:
        :param kwargs:
        :rtype: dict[str, Any]
        """
        # context dict to gather all String inputs for use in system prompt
        context = {}
        # set llm query as trigger
        if trigger := kwargs.get("topic"):
            query = self.trig_callbacks[trigger.name].get_output()
            context[trigger.name] = query

            # handle chat reset
            if (
                self.config.chat_history
                and query.strip().lower() == self.config.history_reset_phrase
            ):
                self.messages = []
                return None

        else:
            query = None

        # aggregate all inputs that are available
        for i in self.callbacks.values():
            if (item := i.get_output()) is not None:
                # set trigger equal to a topic with type String if trigger not found
                if i.input_topic.msg_type is String:
                    if not query:
                        query = item
                    context[i.input_topic.name] = item
                elif i.input_topic.msg_type is Detections:
                    context[i.input_topic.name] = item

        if query is None:
            return None

        # get RAG results if enabled in config and if docs retreived
        rag_result = self._handle_rag_query(query) if self.config.enable_rag else None

        # set system prompt template
        query = (
            self.component_prompt.render(context) if self.component_prompt else query
        )

        # attach rag results to templated query if available
        query = f"{rag_result}\n{query}" if rag_result else query

        message = {"role": "user", "content": query}
        self._handle_chat_history(message)

        self.get_logger().debug(f"Input from component: {self.messages}")

        input = {
            "query": self.messages,
            **self.config._get_inference_params(),
        }

        # Add any tools, if registered
        if self.config._tool_descriptions:
            input["tools"] = self.config._tool_descriptions

        return input

    def _execution_step(self, *args, **kwargs):
        """_execution_step.

        :param args:
        :param kwargs:
        """

        if self.run_type is ComponentRunType.EVENT:
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
        result = self.model_client.inference(inference_input)

        if result:
            result_message = {"role": "assistant", "content": result["output"]}
            self.messages.append(result_message)

            # handle tool calls
            if self.config._tool_descriptions:
                result = self._handle_tool_calls(result)

            # raise a fallback trigger via health status
            if not result:
                self.health_status.set_failure()
                return

            # publish inference result
            if hasattr(self, "publishers_dict"):
                for publisher in self.publishers_dict.values():
                    publisher.publish(**result)

        else:
            # raise a fallback trigger via health status
            self.health_status.set_failure()

    def set_topic_prompt(self, input_topic: Topic, template: Union[str, Path]) -> None:
        """Set prompt template on any input topic of type string.

        :param input_topic: Name of the input topic on which the prompt template is to be applied
        :type input_topic: Topic
        :param template: Template in the form of a valid jinja2 string or a path to a file containing the jinja2 string.
        :type template: Union[str, Path]
        :rtype: None

        Example usage:
        ```python
        llm_component = LLM(inputs=[text0],
                            outputs=[text1],
                            model_client=model_client,
                            config=config,
                            component_name='llama_component')
        llm_component.set_topic_prompt(text0, template="You are an amazing and funny robot. You answer all questions with short and concise answers. Please answer the following: {{ text0 }}")
        ```
        """
        if callback := self.callbacks.get(input_topic.name):
            if not callback:
                raise TypeError("Specified input topic does not exist")
            if not isinstance(callback, TextCallback):
                raise TypeError(
                    f"Prompt can only be set for a topic of type String, {callback.input_topic.name} is of type {callback.input_topic.msg_type}"
                )
            self.config._topic_prompts[input_topic.name] = template

    def set_component_prompt(self, template: Union[str, Path]) -> None:
        """Set component level prompt template which can use multiple input topics.

        :param template: Template in the form of a valid jinja2 string or a path to a file containing the jinja2 string.
        :type template: Union[str, Path]
        :rtype: None

        Example usage:
        ```python
        llm_component = LLM(inputs=[text0],
                            outputs=[text1],
                            model_client=model_client,
                            config=config,
                            component_name='llama_component')
        llm_component.set_component_prompt(template="You are an amazing and funny robot. You answer all questions with short and concise answers. You can see the following items: {{ detections }}. Please answer the following: {{ text0 }}")
        ```
        """
        self.config._component_prompt = template

    def register_tool(
        self,
        tool: Callable,
        tool_description: Dict,
        send_tool_response_to_model: bool = False,
    ) -> None:
        """Register a tool with the component which can be called by the model. If the send_tool_response_to_model flag is set to True than the output of the tool is sent back to the model and final output of the model is sent to component publishers (i.e. the model "uses" the tool to give a more accurate response.). If the flag is set to False than the output of the tool is sent to publishers of the component.

        :param tool: An arbitrary function that needs to be called. The model response will describe a call to this function.
        :type tool: Callable
        :param tool_description: A dictionary describing the function. This dictionary needs to be made in the format shown [here](https://ollama.com/blog/tool-support). Also see usage example.
        :type tool_description: dict
        :param send_tool_response_to_model: Whether the model should be called with the tool response. If set to false the tool response will be sent to component publishers. If set to true, the response will be sent back to the model and the final response from the model will be sent to the publishers. Default is False.
        :param send_tool_response_to_model: bool
        :rtype: None

        Example usage:
        ```python
        def my_arbitrary_function(first_param: str, second_param: int) -> str:
            return f"{first_param}, {second_param}"

        my_func_description = {
                  'type': 'function',
                  'function': {
                    'name': 'my_arbitrary_function',
                    'description': 'Description of my arbitrary function',
                    'parameters': {
                      'type': 'object',
                      'properties': {
                        'first_param': {
                          'type': 'string',
                          'description': 'Description of the first param',
                        },
                        'second_param': {
                          'type': 'int',
                          'description': 'Description of the second param',
                        },
                      },
                      'required': ['first_param', 'second_param'],
                    },
                  },
                }

        my_component.register_tool(tool=my_arbitrary_function, tool_description=my_func_description, send_tool_response_to_model=False)
        ```
        """
        if not isinstance(self.model_client, OllamaClient):
            raise TypeError(
                "Currently registering tools is only supported when using an Ollama client with the component."
            )
        self._external_processors[tool_description["function"]["name"]] = (
            [tool],
            "tool",
        )
        self.config._tool_descriptions.append(tool_description)
        self.config._tool_response_flags[tool_description["function"]["name"]] = (
            send_tool_response_to_model
        )

    def _update_cmd_args_list(self):
        """
        Update launch command arguments
        """
        super()._update_cmd_args_list()

        self.launch_cmd_args = [
            "--db_client",
            self._get_db_client_json(),
        ]

    def _get_db_client_json(self) -> Union[str, bytes, bytearray]:
        """
        Serialize component routes to json

        :return: Serialized inputs
        :rtype:  str | bytes | bytearray
        """
        if not self.db_client:
            return ""
        return json.dumps(self.db_client.serialize())

    def _warmup(self):
        """Warm up and stat check"""
        import time

        message = {"role": "user", "content": "Hello robot."}
        inference_input = {"query": [message], **self.config._get_inference_params()}

        # Run inference once to warm up and once to measure time
        self.model_client.inference(inference_input)

        inference_input = {"query": [message], **self.config._get_inference_params()}
        start_time = time.time()
        result = self.model_client.inference(inference_input)
        elapsed_time = time.time() - start_time

        self.get_logger().warning(f"Model Output: {result['output']}")
        self.get_logger().warning(f"Approximate Inference time: {elapsed_time} seconds")
