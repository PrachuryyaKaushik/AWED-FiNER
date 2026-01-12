from smolagents import CodeAgent, HfApiModel
from tool import AWEDFiNERTool

# 1. Initialize the Expert Tool and the Agent
ner_tool = AWEDFiNERTool()
agent = CodeAgent(tools=[ner_tool], model=HfApiModel())

# 2. Demo: A Vulnerable Language Case (Bishnupriya)
print("--- AWED-FiNER: Agentic Workflow Demo ---")
task = "Identify the entities in this Bishnupriya sentence: 'মাজুলী অসমর এক সাংস্কৃতিক কেন্দ্র।'"

# The agent will detect it's Bishnupriya and call AWED-FiNER automatically
response = agent.run(task)

print(f"\nTask: {task}")
print(f"Agent Final Answer: {response}")
