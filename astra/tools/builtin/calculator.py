"""Calculator tool"""

import ast
import operator
import math
from typing import Dict, Any, List

from ..base import Tool, ToolParameter


class CalculatorTool(Tool):
    """Python calculator tool"""
    
    # Supported operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }
    
    # Supported functions
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Execute math calculations. Supports basic operations and math functions. Example: 2+3*4, sqrt(16), sin(pi/2), etc."
        )
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        Execute calculation

        Args:
            parameters: Dict containing input parameter

        Returns:
            Calculation result
        """
        expression = parameters.get("input", "") or parameters.get("expression", "")
        if not expression:
            return "Error: Calculation expression cannot be empty"

        print(f"🧮 Calculating: {expression}")

        try:
            node = ast.parse(expression, mode='eval')
            result = self._eval_node(node.body)
            result_str = str(result)
            print(f"✅ Result: {result_str}")
            return result_str
        except Exception as e:
            error_msg = f"Calculation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _eval_node(self, node):
        """Recursively evaluate AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.OPERATORS[type(node.op)](
                self._eval_node(node.left), 
                self._eval_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            else:
                raise ValueError(f"Undefined variable: {node.id}")
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")
    
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Math expression to calculate, supports basic operations and math functions",
                required=True
            )
        ]


# Convenience function
def calculate(expression: str) -> str:
    """
    Execute math calculation

    Args:
        expression: Math expression

    Returns:
        Result string
    """
    tool = CalculatorTool()
    return tool.run({"input": expression})

