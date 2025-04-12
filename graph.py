    

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
import time 
from langchain_core.tools import tool

def simple_interest(P: float, r: float, t: int) -> float:
    """
    Calculate simple interest.

    Args:
        P (int): The principal amount.
        r (int): The annual interest rate (as a percentage).
        t (int): The time the money is invested for (in years).
    """
    return P * r * t

def compound_interest(P: float, r: float, n: int, t: int) -> float:
    """
    Calculate compound interest.

    Args:
        P (float): The principal amount.
        r (float): The annual interest rate (in decimal form).
        n (int): The number of times the interest is compounded per year.
        t (int): The time the money is invested for (in years).
    """
    return P * (1 + r / n) ** (n * t)

# Define conversion rates
rates = {
    "USD_to_EUR": 0.85,
    "EUR_to_USD": 1.18,
    "USD_to_GBP": 0.75,
    "GBP_to_USD": 1.33,
    "EUR_to_GBP": 0.88,
    "GBP_to_EUR": 1.14
}

# Currency conversion function
def currency_convert(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert currency based on predefined rates.

    Args:
        amount (float): The amount to convert.
        from_currency (str): The currency to convert from (e.g., "USD").
        to_currency (str): The currency to convert to (e.g., "EUR").
    """
    key = f"{from_currency}_to_{to_currency}"
    if key in rates:
        return round(amount * rates[key], 2)
    else:
        raise ValueError(f"Conversion rate for {key} not available.")



tools = [simple_interest, compound_interest, currency_convert]

llm = ChatOllama(
    model = "qwen2.5:7b",
    temperature = 0,
)

llm_with_tools = llm.bind_tools(tools)

# Create the memory persistence layer
memory = MemorySaver()

# Create the ReAct agent with the tools and prompt 
graph = create_react_agent(
    llm,
    tools=tools,
    prompt="You are a helpful assistant tasked with performing financial calculations on a set of inputs",
    checkpointer=memory
)


