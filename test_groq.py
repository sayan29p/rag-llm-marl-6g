import sys
import os
sys.path.append(os.path.dirname(__file__))

from llm.coordinator import LLMCoordinator

coordinator = LLMCoordinator()

result = coordinator.get_hint(
    state_text="Edge node 1 queue=3 tasks",
    context_string="No past data",
)

print(result)
