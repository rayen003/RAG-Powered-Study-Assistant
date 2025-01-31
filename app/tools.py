from langchain.agents import Tool
from langchain.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
import sympy
from sympy import solve, symbols, parse_expr
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
import traceback
import sys

class MathInput(BaseModel):
    """Input for math operations."""
    expression: str = Field(..., description="The mathematical expression or equation to evaluate")
    operation: str = Field(..., description="The type of operation to perform (solve, evaluate, plot, etc.)")

class MathTool(BaseTool):
    name: str = "math_tool"
    description: str = """Use this tool for mathematical operations including:
    - Solving equations
    - Evaluating expressions
    - Plotting functions
    - Performing calculus operations (derivatives, integrals)
    Input should be a clear mathematical expression or equation."""
    
    args_schema: Type[BaseModel] = MathInput

    def _clean_expression(self, expression: str) -> str:
        """Clean and standardize mathematical expressions."""
        print(f"Original expression: {expression}")
        
        # Detailed logging of characters
        print("Character breakdown:")
        for i, char in enumerate(expression):
            print(f"Index {i}: '{char}' (Unicode: U+{ord(char):04X})")
        
        try:
            # Remove LaTeX and other special notations
            expression = re.sub(r'\\[a-zA-Z]+', '', expression)  # Remove LaTeX commands
            expression = re.sub(r'[\{\}]', '', expression)  # Remove braces
            expression = expression.replace('±', '')  # Remove plus-minus symbol
            expression = expression.replace('\\pm', '')  # Remove LaTeX plus-minus
            expression = expression.replace('\\sqrt', 'sqrt')  # Convert sqrt notation
            
            # Replace power notations
            expression = expression.replace('^', '**')
            expression = re.sub(r'([0-9])([a-zA-Z])', r'\1*\2', expression)  # Add implicit multiplication
            
            # Remove spaces and clean up
            expression = expression.replace(' ', '').strip()
            
            print(f"Cleaned expression: {expression}")
            return expression
        except Exception as e:
            print(f"Error in _clean_expression: {e}")
            print(traceback.format_exc())
            raise

    def _solve_equation(self, equation: str) -> str:
        """Solve mathematical equations."""
        try:
            # Clean the equation
            equation = self._clean_expression(equation)
            
            x = symbols('x')
            
            # Convert equation to expression (move everything to left side)
            if '=' in equation:
                left, right = equation.split('=')
                equation = f"({left})-({right})"
            
            print(f"Solving equation: {equation}")
            
            expr = parse_expr(equation)
            solution = solve(expr, x)
            
            # Format the solution nicely
            if len(solution) == 0:
                return "This equation has no real solutions."
            elif len(solution) == 1:
                return f"The solution is: x = {solution[0]}"
            else:
                return f"The solutions are: x = {', x = '.join(str(sol) for sol in solution)}"
        except Exception as e:
            print(f"Error solving equation: {e}")
            print(traceback.format_exc())
            return f"Error solving equation: {str(e)}"

    def _plot_function(self, expression: str) -> str:
        """Plot mathematical functions and return base64 encoded image."""
        try:
            # Clean the expression
            expression = self._clean_expression(expression)
            
            # If the expression is an equation, extract the left side
            if "=" in expression:
                left, _ = expression.split("=")
                expression = left.strip()
            
            print(f"Plotting function: {expression}")
            
            x = symbols('x')
            expr = parse_expr(expression)
            
            # Create x values
            x_vals = np.linspace(-5, 5, 200)
            # Convert sympy expression to numpy function
            f = sympy.lambdify(x, expr, "numpy")
            y_vals = f(x_vals)
            
            # Create plot
            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label=f'y = {expression}')
            plt.grid(True)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.title(f"Plot of y = {expression}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            
            # Save plot to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{image}"
        except Exception as e:
            print(f"Error plotting function: {e}")
            print(traceback.format_exc())
            return f"Error plotting function: {str(e)}"

    def _run(self, expression: str, operation: str) -> str:
        """Execute the math tool based on the operation type."""
        try:
            print(f"Running math tool with operation: {operation}, expression: {expression}")
            
            operation = operation.lower()
            if operation == "solve":
                return self._solve_equation(expression)
            elif operation == "evaluate":
                # Implement evaluation logic if needed
                return f"Evaluation not implemented for: {expression}"
            elif operation == "plot":
                return self._plot_function(expression)
            else:
                return f"Unsupported operation: {operation}"
        except Exception as e:
            print(f"Error in _run method: {e}")
            print(traceback.format_exc())
            return f"Unexpected error: {str(e)}"

    def _arun(self, query: str):
        """Async implementation of the math tool."""
        raise NotImplementedError("MathTool does not support async")

def get_math_tool() -> Tool:
    """Create and return the math tool."""
    return Tool(
        name="Math Tool",
        description="Useful for solving mathematical problems, equations, and plotting functions",
        func=lambda x: MathTool()._run(x.get("expression"), x.get("operation"))
    )