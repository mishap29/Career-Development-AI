from typing import Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("career-ai-mcp")

# --- Resources Example ---
@mcp.resource()
def get_resource(name: str) -> str:
    """Return the contents of a resource file from the assets directory."""
    try:
        with open(f"assets/{name}", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Resource not found."

# --- Tools Example ---
@mcp.tool()
def say_hello(name: str) -> str:
    """Say hello to a user."""
    return f"Hello, {name}!"

@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# --- Prompts Example ---
PROMPTS = {
    "resume": "Please provide your work experience and education to generate a resume.",
    "job_search": "What kind of job are you looking for?"
}

@mcp.prompt()
def get_prompt(name: str) -> str:
    """Return a pre-written prompt template."""
    return PROMPTS.get(name, "Prompt not found.")

if __name__ == "__main__":
    mcp.run(transport='stdio')