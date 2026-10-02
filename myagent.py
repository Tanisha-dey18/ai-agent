import os
import ast
import json
import operator

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL = "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """Search the web via Tavily and return a short text answer."""
    result = tavily_client.search(query=query, include_answer=True, max_results=5)

    if result.get("answer"):
        return result["answer"]

    # Fall back to stitching together the top results if no direct answer.
    snippets = [f"- {r['title']}: {r['content']}" for r in result.get("results", [])]
    return "\n".join(snippets) if snippets else "No results found."


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression like '2*2' or '(3+4)/2'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ---------------------------------------------------------------------------
# Tool definitions (standard OpenAI/Groq function-calling format)
# ---------------------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information and return an answer to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression (addition, subtraction, multiplication, division, exponents) and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate, e.g. '2*2' or '(10-4)/3'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

available_functions = {
    "web_search": web_search,
    "calculate": calculate,
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the web_search tool for questions needing "
            "current or factual information, and the calculate tool for arithmetic. "
            "Otherwise answer directly.",
        },
        {"role": "user", "content": user_query},
    ]

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        return response_message.content

    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)

        function_response = function_to_call(**function_args)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_response),
            }
        )

    second_response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return second_response.choices[0].message.content


if __name__ == "__main__":
    user_input = input("Ask me anything: ")
    answer = run_agent(user_input)
    print(answer)
