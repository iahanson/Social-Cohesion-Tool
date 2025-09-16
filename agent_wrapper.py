from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint="https://shared-ai-hub-foundry.services.ai.azure.com/api/projects/Team-15-Communities")

agent = project.agents.get_agent("asst_hpw4j8I1zrSDVIFidlfegvSi")

thread = project.agents.threads.create()
print(f"Created thread, ID: {thread.id}")

message=project.agents.messages.create(
    thread_id=thread.id,
    role="user",
    content=(
        """Research UK news sources and give me a summary of the most recent trending topics"""
    )
)

run = project.agents.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id)

if run.status == "failed":
    print(f"Run failed: {run.last_error}")
else:
    messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)

    for message in messages:
        if message.text_messages:
            print(f"{message.role}: {message.text_messages[-1].text.value}")