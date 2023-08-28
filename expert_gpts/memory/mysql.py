import json
import logging
from typing import List

from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict
from sqlalchemy.orm import Session

from expert_gpts.database import get_db_session
from expert_gpts.database.chat_message import ChatMessage as ExpertGPTsChatMessage

logger = logging.getLogger(__name__)


class MysqlChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Postgres database."""

    def __init__(
        self,
        session_id: str,
        session: Session,
    ):
        self.session = session
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        with get_db_session() as session:
            messages = ExpertGPTsChatMessage.get_messages_by_session_id(
                session, self.session_id
            )
            items = [record.to_json() for record in messages]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        with get_db_session() as session:
            ExpertGPTsChatMessage.add_message(
                self.session_id, json.dumps(_message_to_dict(message)), session
            )

    def clear(self) -> None:
        with get_db_session() as session:
            ExpertGPTsChatMessage.clear(session, self.session_id)

    def fuzzy_search(self, search: str, distance: int = 5, limit: int = 5):
        logger.info(f"Searching for {search} in {self.session_id}")
        with get_db_session() as session:
            messages = ExpertGPTsChatMessage.search_by_message(
                self.session_id, search, session, distance, limit
            )
            items = []
            for record in messages:
                content = json.loads(json.loads(record.message))
                print(content, type(content))
                items.append(
                    f"At {record.created_at}, by {content['type']}: {content['data']['content']}"
                )
            return items