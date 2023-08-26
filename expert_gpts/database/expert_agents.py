from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from expert_gpts.database import Base


class ExpertAgentToolPrompt(Base):
    __tablename__ = "expert_agent_tool_prompt"

    id: Mapped[int] = mapped_column(primary_key=True)
    expert_agent_key: Mapped[str] = mapped_column(String(190))
    expert_agent_tool_prompt: Mapped[str] = mapped_column(Text())
