from datetime import datetime

from langchain.schema.messages import BaseMessage
from sqlalchemy import JSON, DateTime, String, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from expert_gpts.database import Base


class ChatMessage(Base):
    __tablename__ = "message_store"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[str] = mapped_column(DateTime(), default=datetime.utcnow)
    session_id: Mapped[str] = mapped_column(String(190))
    message: Mapped[str] = mapped_column(JSON())

    @classmethod
    def create_levenstein(cls, engine: Engine):
        """
        let's see how it works
        https://lucidar.me/en/web-dev/levenshtein-distance-in-mysql/

        :param engine:
        :return:
        """
        levenstein = """
CREATE FUNCTION IF NOT EXISTS levenshtein( s1 VARCHAR(255), s2 VARCHAR(255) )
RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE s1_len, s2_len, i, j, c, c_temp, cost INT;
    DECLARE s1_char CHAR;
    DECLARE cv0, cv1 VARBINARY(256);

    SET s1_len = CHAR_LENGTH(s1), s2_len = CHAR_LENGTH(s2), cv1 = 0x00, j = 1, i = 1, c = 0;

    IF s1 = s2 THEN
        RETURN 0;
    ELSEIF s1_len = 0 THEN
        RETURN s2_len;
    ELSEIF s2_len = 0 THEN
        RETURN s1_len;
    ELSE
        WHILE j <= s2_len DO
            SET cv1 = CONCAT(cv1, UNHEX(HEX(j))), j = j + 1;
        END WHILE;

        WHILE i <= s1_len DO
            SET s1_char = SUBSTRING(s1, i, 1), c = i, cv0 = UNHEX(HEX(i)), j = 1;
            WHILE j <= s2_len DO
                SET c = c + 1;
                IF s1_char = SUBSTRING(s2, j, 1) THEN
                    SET cost = 0;
                ELSE
                    SET cost = 1;
                END IF;

                SET c_temp = CONV(HEX(SUBSTRING(cv1, j, 1)), 16, 10) + cost;
                IF c > c_temp THEN SET c = c_temp; END IF;

                SET c_temp = CONV(HEX(SUBSTRING(cv1, j+1, 1)), 16, 10) + 1;
                IF c > c_temp THEN
                    SET c = c_temp;
                END IF;

                SET cv0 = CONCAT(cv0, UNHEX(HEX(c))), j = j + 1;
            END WHILE;

            SET cv1 = cv0, i = i + 1;
        END WHILE;
    END IF;

    RETURN c;
END;

"""
        with engine.connect() as connection:
            connection.execute(text(levenstein))

    @classmethod
    def search_by_message(
        cls,
        session_id: str,
        message: str,
        session: Session,
        distance: int = 4,
        limit: int = 3,
    ) -> list["ChatMessage"]:
        # create query with query builder
        # SELECT *, JSON_VALUE(message, '$.message.content') as message_2
        # FROM `message_store` WHERE
        # levenshtein({message}, JSON_VALUE(message, '$.message.content'))
        # BETWEEN 0 AND {distance} LIMIT {limit};

        query = """
SELECT *, JSON_VALUE(message, '$.message.content') as message_2
FROM `message_store` WHERE
session_id = "{session_id}"
AND
levenshtein("{message}", JSON_VALUE(message, '$.message.content'))
BETWEEN 0 AND {distance} LIMIT {limit};
        """

        query = session.execute(
            text(
                query.format(
                    message=message,
                    distance=distance,
                    limit=limit,
                    session_id=session_id,
                )
            )
        )
        return query.all()

    @classmethod
    def clear(cls, session: Session, session_id: str) -> None:
        query = f"DELETE FROM {cls.__tablename__} WHERE session_id = :session_id;"
        session.execute(text(query), {"session_id": session_id})

    @classmethod
    def truncate(cls, session: Session) -> None:
        query = f"TRUNCATE TABLE {cls.__tablename__}"
        session.execute(text(query))

    @classmethod
    def add_message(cls, session_id: str, message: str, session: Session) -> None:
        """Append the message to the record in PostgreSQL"""
        new_message = ChatMessage(
            session_id=session_id, message=message, created_at=datetime.utcnow()
        )
        session.add(new_message)
        session.commit()

    @classmethod
    def get_messages_by_session_id(
        cls, session: Session, session_id: str
    ) -> list[BaseMessage]:
        """Get all messages by session_id"""
        query = session.query(ChatMessage).filter(ChatMessage.session_id == session_id)
        return [message.message for message in query.all()]
