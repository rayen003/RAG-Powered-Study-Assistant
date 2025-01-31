from tools import MathTool

def test_math_tool():
    math_tool = MathTool()
    
    # Test equation solving
    print("Testing equation solving:")
    result = math_tool._run(expression="x^2 + 2*x - 3 = 0", operation="solve")
    print(result)
    
    # Test expression evaluation
    print("\nTesting expression evaluation:")
    result = math_tool._run(expression="2*5 + 3^2", operation="evaluate")
    print(result)
    
    # Test function plotting
    print("\nTesting function plotting:")
    result = math_tool._run(expression="sin(x)", operation="plot")
    print("Plot generated successfully" if "data:image/png;base64" in result else "Plot failed")

if __name__ == "__main__":
    test_math_tool()