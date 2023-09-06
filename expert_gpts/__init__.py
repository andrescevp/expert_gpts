from expert_gpts.database import engine
from expert_gpts.database.chat_message import ChatMessage
from expert_gpts.database.expert_agents import ExpertAgentToolPrompt

ExpertAgentToolPrompt.metadata.create_all(engine)
ChatMessage.metadata.create_all(engine)
