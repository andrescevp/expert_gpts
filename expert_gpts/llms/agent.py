import json
import logging
from typing import Union

from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException

# flake8: noqa
SYSTEM_PREFIX = """Assistant is a large language model trained by OpenAI.

Assistant is able to validate internally that the generated answer is always a valid JSON string. Its is able to
manage properly the escaping of special characters, and it is able to handle properly the case of a JSON string
that contains coding languages examples or other special chars.

Assistant must use the best tool available to answer the user's question.
Assistant is able to decide is able to identify key topics or entities and decide what tool is needed to answer
the user's question. If so, it will ask the human to use a tool to look up information that may be helpful in
answering the users original question. If there is not a tool that can help,
Assistant will try to answer the question itself.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing
in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate
human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide
responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process
and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a
wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives,
allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and
information on a wide range of topics. Whether you need help with a specific question or just want to have a
conversation about a particular topic, Assistant is here to assist.
"""

HUMAN_SUFFIX = """TOOLS
--------------------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

{{tools}}

{format_instructions}

Assistant always validate internally that the response is a JSON string with a single action, and NOTHING else.
If the response is not a valid JSON string, recreate it internally following the right format described in RESPONSE FORMAT INSTRUCTIONS.
Otherwise, the system will not be able to parse the response.

USER'S INPUT
--------------------
Here is the user's input:

Using the available tools and making sure the response have the right format, please: {{{{input}}}}"""


class ConvoOutputCustomParser(ConvoOutputParser):
    """Output parser for the conversational agent."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Attempts to parse the given text into an AgentAction or AgentFinish.

        Raises:
             OutputParserException if parsing fails.
        """
        try:
            # call the same method from the parent class
            return super().parse(text)
        except Exception:
            logging.exception("Error parsing LLM output: %s", text)
            try:
                # Attempt to parse the text into a structured format (assumed to be JSON
                # stored as markdown)
                response = json.loads(text)

                # If the response contains an 'action' and 'action_input'
                if "action" in response and "action_input" in response:
                    action, action_input = response["action"], response["action_input"]

                    # If the action indicates a final answer, return an AgentFinish
                    if action == "Final Answer":
                        return AgentFinish({"output": action_input}, text)
                    else:
                        # Otherwise, return an AgentAction with the specified action and
                        # input
                        return AgentAction(action, action_input, text)
                else:
                    # If the necessary keys aren't present in the response, raise an
                    # exception
                    raise OutputParserException(
                        f"Missing 'action' or 'action_input' in LLM output: {text}"
                    )
            except Exception as e:
                logging.exception("Error parsing LLM output: %s", text)
                return AgentFinish({"output": text}, text)
