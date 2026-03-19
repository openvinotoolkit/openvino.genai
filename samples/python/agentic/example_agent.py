"""
OpenVINO GenAI ReAct Agent Execution

This executable demonstrates a modern, transparent ReAct (Reasoning and Acting) agent 
loop using pure `langchain_core`. By bypassing deprecated legacy components like 
`AgentExecutor`, this architecture guarantees forward compatibility. It binds the 
OpenVINO hardware-accelerated local inference directly to cognitive tool utilization.
"""
import sys
import os
import re
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm_wrapper import OpenVINOChatModel

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVINO GenAI ReAct Agent")
    parser.add_argument("model_dir", type=str, nargs="?", default="./models/llama-3.2-3b", help="Path to the OpenVINO model directory")
    args = parser.parse_args()

    llm = OpenVINOChatModel(
        model_path=args.model_dir,
        device="CPU",
        generation_kwargs={"max_new_tokens": 256, "temperature": 0.1}
    )
    
    system_prompt = """Answer questions as best you can. You have access to the following tools:
    calculator: Evaluates a mathematical expression.

    Use the following format:
    Question: the input question
    Thought: think about what to do
    Action: the action to take, should be one of [calculator]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation repeats until Final Answer)
    Thought: I know the final answer
    Final Answer: the final answer"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Question: What is 25 * 4 + 10?")
    ]
    
    print("Initializing OpenVINO Agentic Loop...\n")
    
    for step in range(5):
        response = llm.invoke(messages, stop=["\nObservation:"])
        content = response.content
        print(content)
        
        messages.append(AIMessage(content=content))
        
        if "Final Answer:" in content:
            break
            
        action_match = re.search(r"Action:\s*(.*?)\n", content)
        input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", content)
        
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            
            if action == "calculator":
                obs = calculator(action_input)
                print(f"\nObservation: {obs}\n")
                messages.append(HumanMessage(content=f"Observation: {obs}"))
            else:
                messages.append(HumanMessage(content=f"Observation: Unknown tool {action}"))
        else:
            messages.append(HumanMessage(content="Observation: Invalid format. Please provide Action and Action Input."))

if __name__ == "__main__":
    main()