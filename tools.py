# from langchain.tools import Tool, tool
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.utilities import WolframAlphaAPIWrapper
# from langchain.agents import AgentExecutor, create_react_agent
# from app.rag import RAGPipeline
# from langchain.agents import Tool
# from langchain.tools import BaseTool
# from typing import Optional, Type
# from pydantic import BaseModel, Field
# import sympy
# from sympy import solve, symbols, parse_expr
# import numpy as np
# import matplotlib.pyplot as plt
# import io
# import base64

# class MathInput(BaseModel):
#     """Input for math operations."""
#     expression: str = Field(..., description="The mathematical expression or equation to evaluate")
#     operation: str = Field(..., description="The type of operation to perform (solve, evaluate, plot, etc.)")

# class MathTool(BaseTool):
#     name = "math_tool"
#     description = """Use this tool for mathematical operations including:
#     - Solving equations
#     - Evaluating expressions
#     - Plotting functions
#     - Performing calculus operations (derivatives, integrals)
#     Input should be a clear mathematical expression or equation."""
    
#     args_schema: Type[BaseModel] = MathInput

#     def _solve_equation(self, equation: str) -> str:
#         """Solve mathematical equations."""
#         try:
#             x = symbols('x')
#             # Convert equation to expression (move everything to left side)
#             if '=' in equation:
#                 left, right = equation.split('=')
#                 equation = f"({left})-({right})"
            
#             expr = parse_expr(equation)
#             solution = solve(expr, x)
#             return f"Solution: x = {solution}"
#         except Exception as e:
#             return f"Error solving equation: {str(e)}"

#     def _evaluate_expression(self, expression: str) -> str:
#         """Evaluate mathematical expressions."""
#         try:
#             expr = parse_expr(expression)
#             result = expr.evalf()
#             return f"Result: {result}"
#         except Exception as e:
#             return f"Error evaluating expression: {str(e)}"

#     def _plot_function(self, expression: str) -> str:
#         """Plot mathematical functions and return base64 encoded image."""
#         try:
#             x = symbols('x')
#             expr = parse_expr(expression)
            
#             # Create x values
#             x_vals = np.linspace(-10, 10, 200)
#             # Convert sympy expression to numpy function
#             f = sympy.lambdify(x, expr, "numpy")
#             y_vals = f(x_vals)
            
#             # Create plot
#             plt.figure(figsize=(8, 6))
#             plt.plot(x_vals, y_vals)
#             plt.grid(True)
#             plt.title(f"Plot of {expression}")
#             plt.xlabel("x")
#             plt.ylabel("y")
            
#             # Save plot to bytes buffer
#             buf = io.BytesIO()
#             plt.savefig(buf, format='png')
#             plt.close()
#             buf.seek(0)
            
#             # Convert to base64
#             image = base64.b64encode(buf.getvalue()).decode()
#             return f"data:image/png;base64,{image}"
#         except Exception as e:
#             return f"Error plotting function: {str(e)}"

#     def _run(self, expression: str, operation: str) -> str:
#         """Execute the math tool based on the operation type."""
#         operation = operation.lower()
#         if operation == "solve":
#             return self._solve_equation(expression)
#         elif operation == "evaluate":
#             return self._evaluate_expression(expression)
#         elif operation == "plot":
#             return self._plot_function(expression)
#         else:
#             return f"Unsupported operation: {operation}"

#     def _arun(self, query: str):
#         """Async implementation of the math tool."""
#         raise NotImplementedError("MathTool does not support async")

# def get_math_tool() -> Tool:
#     """Create and return the math tool."""
#     return Tool(
#         name="Math Tool",
#         description="Useful for solving mathematical problems, equations, and plotting functions",
#         func=lambda x: MathTool()._run(x.get("expression"), x.get("operation"))
#     )

# @tool
# def wolfram_alpha(query: str) -> str:
#     """Use WolframAlpha to get the answer to a question."""
#     wolfram = WolframAlphaAPIWrapper()
#     math_tool = Tool(
#         name="Wolfram Alpha",
#         func=wolfram.run,
#         description="Use WolframAlpha to get the answer to a question."
#     )

# agent = create_react_agent(
#     llm=ChatOpenAI(temperature=0),
#     tools=[math_tool, get_math_tool()],
#     verbose=True
# )
print(len("""Hi [First Name], I’m [Your Name], [Role] at enode—Europe’s first student-led initiative bridging AI/data science academia-industry gaps. We empower students via real-world projects (e.g., sustainability, fintech) guided by experts.Given your expertise in [industry/role at Company], would you consider mentorship or collaborating on [Event]? I’d love to discuss this further at your convenience.
Best,[Your Name]esadenode.com"""))