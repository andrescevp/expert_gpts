import json
import logging
from typing import Any, Union

from langchain.agents import AgentOutputParser, ConversationalChatAgent
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


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
                # If any other exception is raised during parsing, also raise an
                # OutputParserException
                raise OutputParserException(
                    f"Could not parse LLM output: {text}"
                ) from e


class ConversationalChatAgentCustomParser(ConversationalChatAgent):
    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ConvoOutputCustomParser()
