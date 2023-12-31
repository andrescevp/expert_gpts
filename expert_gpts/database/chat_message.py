from typing import Optional

from sqlalchemy import JSON, DateTime, Integer, String, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from expert_gpts.database import Base


class ChatMessage(Base):
    __tablename__ = "message_store"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[str] = mapped_column(DateTime())
    session_id: Mapped[str] = mapped_column(String(190))
    ai_key: Mapped[str] = mapped_column(String(190))
    message: Mapped[str] = mapped_column(JSON())
    quality: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)

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
    def search_by_message_levenstein(
        cls,
        ai_key: str,
        session_id: str,
        message: str,
        session: Session,
        distance: int = 4,
        limit: int = 3,
    ) -> list["ChatMessage"]:
        # create query with query builder - order of where very important
        # SELECT *, JSON_VALUE(message, '$.message.content') as message_2
        # FROM `message_store` WHERE session_id = "{session_id}"
        # AND
        # levenshtein({message}, JSON_VALUE(message, '$.message.content'))
        # BETWEEN 0 AND {distance} LIMIT {limit};

        query = """
SELECT *, JSON_VALUE(message, '$.message.content') as message_2
FROM `message_store` WHERE
session_id = "{session_id}"
AND
ai_key = "{ai_key}"
AND
levenshtein("{message}", JSON_VALUE(message, '$.message.content'))
BETWEEN 0 AND {distance} ORDER BY created_at ASC LIMIT {limit};
        """

        query = session.execute(
            text(
                query.format(
                    message=message,
                    distance=distance,
                    limit=limit,
                    session_id=session_id,
                    ai_key=ai_key,
                )
            )
        )
        return query.all()
