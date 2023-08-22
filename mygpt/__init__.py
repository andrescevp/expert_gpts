from mygpt.database import engine
from mygpt.database.expert_agents import ExpertAgentToolPrompt

ExpertAgentToolPrompt.metadata.create_all(engine)
