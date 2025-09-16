from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint="https://shared-ai-hub-foundry.services.ai.azure.com/api/projects/Team-15-Communities")

agent = project.agents.get_agent("asst_MKXJWz6odu2T5o8yVuRPonxB")

thread = project.agents.threads.create()
print(f"Created thread, ID: {thread.id}")

message=project.agents.messages.create(
    thread_id=thread.id,
    role="user",
    content=(
        """Dont ask questions and only output the finished csv. Scan local news articles in the  UK across all local authorities and provide links based on the defined categories.Output a csv
        Include the source area of the article and also the area referenced in the article."""
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