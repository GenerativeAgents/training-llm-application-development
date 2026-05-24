import random

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="random-number-mcp")


@mcp.tool()
def generate_random_number(min_value: int = 0, max_value: int = 100) -> int:
    """ランダムな数字を生成します。min_valueからmax_valueまでの範囲で整数を返します。"""

    return random.randint(min_value, max_value)


if __name__ == "__main__":
    mcp.run(transport="stdio")
