import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

@tool
def scientific_calculator(expression: str) -> str:
    """
    Use this tool for ANY math calculation, including:
    - Arithmetic (+, -, *, /)
    - Powers and roots
    - Trigonometric functions (sin, cos, tan)
    - Logarithms and exponentials
    - Derivatives (diff)
    - Integrals (integrate)
    Accept input as a string, e.g. 'integrate(cos(x))' or 'diff(sin(x))'.
    Return the simplified result as a string.
    """
    import sympy
    from sympy import sympify
    from sympy.abc import x, y, z

    try:
        expr = sympify(expression, evaluate=True)
        result = expr.doit() if hasattr(expr, 'doit') else expr.evalf()
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0,
    )

    tools = [scientific_calculator]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

    print("🧮 Scientific AI Calculator Ready. Type 'quit' to exit.")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() == "quit":
            break

        result = agent.invoke(query)
        print("\nAssistant:", result["output"])




if __name__ == "__main__":
    main()
