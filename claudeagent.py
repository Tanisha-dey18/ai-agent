import os
import json
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Initialize Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

# 1. Tool Definitions
def web_search(query: str) -> str:
    """Performs a web search using Tavily API."""
    response = tavily_client.search(query=query, max_results=3)
    return json.dumps(response.get("results", []))

def calculate(expression: str) -> str:
    """Evaluates basic mathematical expressions."""
    try:
        # Simple eval for basic arithmetic operations
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Tool mapping dict
available_tools = {
    "web_search": web_search,
    "calculate": calculate,
}

# Tools schema for Groq API
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time or current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate basic mathematical expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression like '12 * 45'"}
                },
                "required": ["expression"],
            },
        },
    },
]

# 2. Agent Execution Function
def run_agent(user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]

    # First call to Groq model
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # If LLM decides to call a tool
    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_to_call = available_tools[function_name]

            # Execute tool
            function_response = function_to_call(**function_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_response),
            })

        # Second call to LLM with tool response context
        second_response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        return second_response.choices[0].message.content

    return response_message.content

if __name__ == "__main__":
    user_input = input("Ask me anything: ")
    answer = run_agent(user_input)
    print("\nResponse:\n", answer)