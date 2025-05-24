"""
src/core.py
A minimal extensible multi-agent orchestration framework.

Minimal extensible multi-agent orchestration framework whom anyone can use for any taks just by 
adding agents, tools and orchestrating them.

Core Principles and Ideas behind building:
Minimal
Non-Overengineered
Ready to use
Dynamic
Intelligent
Production Ready
"""

from __future__ import annotations

import inspect, json, logging, typing as t, re
from dataclasses import dataclass, field
from collections import defaultdict, deque
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

__all__ = [
    "tool",
    "ToolRegistry",
    "LLM",
    "AgentConfig",
    "BaseAgent",
    "Orchestrator",
]

# ════════════════════════════════════════════════════════════════════════
#  Registry helpers
# ════════════════════════════════════════════════════════════════════════
class ToolRegistry:
    """Global registry of callable tools & their JSON schemas."""

    _tools: dict[str, t.Callable] = {}
    _schemas: dict[str, dict] = {}

    # ── public ──────────────────────────────────────────────────────────
    @classmethod
    def register(cls, fn: t.Callable) -> t.Callable:
        name = fn.__name__
        cls._tools[name] = fn
        cls._schemas[name] = cls._build_schema(fn)
        return fn

    @classmethod
    def get(cls, name: str) -> t.Callable:  # raises KeyError if missing
        return cls._tools[name]

    @classmethod
    def schema(cls, name: str) -> dict:
        return cls._schemas[name]

    # ── internal ────────────────────────────────────────────────────────
    @staticmethod
    def _py_to_json(anno):
        origin = getattr(anno, "__origin__", anno)
        return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(origin, "string")

    @classmethod
    def _build_schema(cls, fn):
        sig = inspect.signature(fn)
        props, req = {}, []
        for n, p in sig.parameters.items():
            anno = p.annotation if p.annotation is not inspect._empty else str
            props[n] = {"type": cls._py_to_json(anno)}
            if p.default is inspect._empty:
                req.append(n)
        return {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": fn.__doc__ or "",
                "parameters": {"type": "object", "properties": props, "required": req},
            },
        }


def tool(name: str | None = None):
    """Decorator → registers *name* as a callable tool."""

    def deco(fn):
        fn.__name__ = name or fn.__name__
        return ToolRegistry.register(fn)

    return deco

# ════════════════════════════════════════════════════════════════════════
#  LLM adapter interface
# ════════════════════════════════════════════════════════════════════════
class LLM:
    """Minimal chat‑model wrapper so you can swap GPT, Claude, Mistral, etc."""

    def __init__(self, model_name: str, client):
        self.model = model_name
        self.client = client

    def chat(self, messages: list[dict], **kwargs):
        """Pass‑through to the SDK; returns the first choice's message."""
        # Suppress HTTP request logs by redirecting stderr temporarily if needed
        import sys, io, logging
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        logging.getLogger('http.client').setLevel(logging.ERROR)
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
            return resp.choices[0].message
        finally:
            sys.stderr = old_stderr

# ════════════════════════════════════════════════════════════════════════
#  Agent base class
# ════════════════════════════════════════════════════════════════════════
@dataclass
class AgentConfig:
    name: str
    llm: LLM
    tools: list[str] = field(default_factory=list)
    allowed_agents: list[str] = field(default_factory=list)
    task: str = "generic agent"
    temperature: float = 0.7


class BaseAgent:
    """Superclass; subclass *decide()* or override *_system_prompt()* if desired."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    # ── main decision wrapper ───────────────────────────────────────────
    def decide(self, state: dict) -> dict:
        messages = [
            {"role": "system", "content": self._system_prompt(state)},
            {"role": "user", "content": "Decide next action now"},
        ]
        
        # Log the current agent and state for debugging
        logging.debug(f"Agent {self.cfg.name} is making a decision with state: {state}")
        logging.info(f"\n{"="*20} {Fore.MAGENTA}{self.cfg.name}{Style.RESET_ALL} {"="*20}\n\nInput: {pretty_print_json(state)}")
        
        msg = self.cfg.llm.chat(
            messages,
            tools=[ToolRegistry.schema(n) for n in self.cfg.tools],
            tool_choice="auto",
            response_format={"type": "json_object"},  # OPENAI JSON‑mode
            temperature=self.cfg.temperature,
        )

        # ── Parse model output ───────────────────────────────────────────
        if getattr(msg, "tool_calls", None):
            # Only process the first tool call, ignoring any parallel calls
            tc = msg.tool_calls[0]
            decision = {
                "action": "tool",
                "tool": tc.function.name,
                "args": json.loads(tc.function.arguments or "{}"),
            }
            logging.debug(f"Agent {self.cfg.name} decided to use tool: {tc.function.name}")
            logging.info(f"\n{"-"*20} {Fore.YELLOW}{tc.function.name} Tool {Style.RESET_ALL} {"-"*20}\n\nInput: {pretty_print_json(json.loads(tc.function.arguments or '{}'))}")
            return decision

        # Parse the content as JSON or handle plain text
        decision = _safe_json(msg.content)
        logging.debug(f"Agent {self.cfg.name} decision: {decision}")
        
        # Normalize the decision format - handle common field name variations
        normalized = {"action": decision.get("action", "")}
        
        # Handle handoff with different field names
        if normalized["action"] == "handoff" or "recipient_name" in decision or "agent" in decision:
            normalized["action"] = "handoff"
            # Get agent name from any of the possible fields
            agent_name = decision.get("agent") or decision.get("recipient_name") or decision.get("target")
            normalized["agent"] = agent_name
            
            # If no valid agent is specified, convert to a respond action
            if not agent_name:
                logging.warning(f"Agent {self.cfg.name} attempted handoff without specifying target agent")
                return {
                    "action": "respond",
                    "output": "I'm not sure which agent should handle this next. Could you clarify?",
                }
        
        # Handle respond with different field names
        elif normalized["action"] == "respond" or "message" in decision or "output" in decision:
            normalized["action"] = "respond"
            # Get output from any of the possible fields
            output = decision.get("output") or decision.get("message") or decision.get("content") or ""
            normalized["output"] = output
            
            # Ensure there's always an output
            if not output:
                logging.warning(f"Agent {self.cfg.name} responded with empty output")
                normalized["output"] = f"The {self.cfg.name} has processed your request but didn't provide a specific response."
        
        # Handle tool with different field names
        elif normalized["action"] == "tool" or "tool" in decision or "tool_name" in decision:
            normalized["action"] = "tool"
            normalized["tool"] = decision.get("tool") or decision.get("tool_name")
            normalized["args"] = decision.get("args") or decision.get("parameters") or decision.get("arguments") or {}
        
        # If we couldn't determine the action, default to respond
        else:
            logging.warning(f"Agent {self.cfg.name} provided an unknown action: {normalized['action']}")
            normalized = {
                "action": "respond",
                "output": f"The {self.cfg.name} encountered an issue processing your request."
            }
        
        logging.debug(f"Agent {self.cfg.name} normalized decision: {normalized}")
        return normalized

    # ── prompt builder ──────────────────────────────────────────────────
    def _system_prompt(self, state: dict) -> str:
        short_state = json.dumps(state, default=str, indent=2)[:1800]
        
        # Build examples for each action type
        tool_example = ""
        if self.cfg.tools:
            tool_example = f'{{"action": "tool", "tool": "{self.cfg.tools[0]}", "args": {{"param1": "value1"}}}}'
        
        handoff_example = ""
        if self.cfg.allowed_agents:
            handoff_example = f'{{"action": "handoff", "agent": "{self.cfg.allowed_agents[0]}"}}'  
            
        respond_example = '{"action": "respond", "output": "This is my response to the user."}'
        
        examples = []
        if tool_example: examples.append(f"Tool usage: {tool_example}")
        if handoff_example: examples.append(f"Agent handoff: {handoff_example}")
        examples.append(f"User response: {respond_example}")
        
        return (
        f"You are **{self.cfg.name}**."
        f"Mission: {self.cfg.task}\n\n"
        "You have some data which might or might not be relevant to the task. If you need additional data, either a tool or agent can provide that."
        f"Allowed tools for you to use: {', '.join(self.cfg.tools) or 'none'}\n"
        f"Agents you can hand-off to: {', '.join(self.cfg.allowed_agents) or 'none'}\n\n"
        "IMPORTANT: You must respond with a valid JSON object containing an 'action' field with one of these values:\n"
        "1. 'tool' - to use a tool (include 'tool' and 'args' fields)\n"
        "2. 'handoff' - to hand off to another agent (include 'agent' field)\n"
        "3. 'respond' - to respond to the user (include 'output' field)\n\n"
        "CRITICAL: Only use ONE tool at a time. DO NOT attempt to use multiple tools in parallel.\n\n"
        "IMPORTANT TOOL HANDLING GUIDELINES:\n"
        "1. If a tool returns empty results (empty list, empty dict, etc.), DO NOT retry with the same parameters.\n"
        "2. For empty tool results, either try different BUT VALID parameters or hand off to another agent that can handle the situation.\n"
        "3. Avoid loops by not repeating the same action when it doesn't produce useful results.\n\n"
        f"Examples:\n{chr(10).join(examples)}\n\n"
        "If you're very unclear about which tool to call or which agent to hand-off to, respond with:\n"
        "{\"action\": \"respond\", \"output\": \"I need more information to proceed. Please clarify...\"}\n\n"
        "Current shared state ↓\n" + short_state
        )

# ════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ════════════════════════════════════════════════════════════════════════
class Orchestrator:
    """Runs agents until one decides to respond to the user."""

    def __init__(self, agents: dict[str, BaseAgent], max_turns: int = 30):
        if not agents:
            raise ValueError("At least one agent required")
        self.agents = agents
        self.current = next(iter(agents))  # default entry agent
        self.state: dict = {}
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=4))
        self.max_turns = max_turns

    # ── public API ──────────────────────────────────────────────────────
    def start(self, entry_agent: str, user_message: str) -> str:
        if entry_agent not in self.agents:
            raise KeyError(entry_agent)
        self.current = entry_agent
        self.state["user_message"] = user_message
        return self.run()

    def run(self) -> str:
        logging.debug(f"Starting orchestration with initial agent: {self.current}")
        logging.debug(f"Initial state: {self.state}")
        
        for turn in range(self.max_turns):
            logging.debug(f"Turn {turn+1}: Current agent is {self.current}")
            agent = self.agents[self.current]
            
            try:
                decision = agent.decide(self.state)
                logging.debug(f"Decision from {self.current}: {decision}")
                
                if self._loop_guard(agent.cfg.name, decision):
                    logging.warning(f"Loop detected with agent {agent.cfg.name}")
                    return f"{agent.cfg.name} repeated the same action 3×. Aborting to avoid an infinite loop."  # failsafe

                match decision.get("action"):
                    case "tool":
                        name = decision.get("tool")
                        if not name:
                            logging.error(f"Agent {self.current} tried to use a tool without specifying which one")
                            self.state["error"] = "Tool action missing tool name"
                            return f"Agent {self.current} tried to use a tool but didn't specify which one."
                            
                        try:
                            tool_name = name
                            # Remove any module prefix if present (e.g., 'functions.search_flights' -> 'search_flights')
                            if '.' in tool_name:
                                tool_name = tool_name.split('.')[-1]
                                
                            logging.debug(f"Executing tool {tool_name} with args: {decision.get('args', {})}")
                            logging.info(f"\n{"-"*20} {Fore.YELLOW}{tool_name} Tool {Style.RESET_ALL} {"-"*20}\n\nInput: {pretty_print_json(decision.get('args', {}))}")
                            
                            try:
                                result = ToolRegistry.get(tool_name)(**decision.get("args", {}))
                                logging.debug(f"Tool {tool_name} result: {result}")
                                # Pretty print JSON results if possible
                                if isinstance(result, (dict, list)):
                                    logging.info(f"Output: {pretty_print_json(result)}")
                                else:
                                    logging.info(f"Output: {result}")
                            except KeyError:
                                # Try with the original name as fallback
                                result = ToolRegistry.get(name)(**decision.get("args", {}))
                                logging.debug(f"Tool {name} result: {result}")
                                # Pretty print JSON results if possible
                                if isinstance(result, (dict, list)):
                                    logging.info(f"Output: {pretty_print_json(result)}")
                                else:
                                    logging.info(f"Output: {result}")
                            self.state[name] = result
                        except Exception as e:
                            logging.error(f"Error executing tool {name}: {str(e)}")
                            self.state["error"] = f"Tool execution error: {str(e)}"
                            return f"Error executing tool {name}: {str(e)}"
                            
                    case "handoff":
                        target = decision.get("agent")
                        if not target or target not in self.agents:
                            logging.warning(f"Invalid handoff target: {target}")
                            self.state["error"] = f"Invalid handoff target: {target}"
                            return f"Agent {self.current} tried to hand‑off to an unknown agent ({target})."
                        logging.debug(f"Handing off from {self.current} to {target}")
                        logging.info(f"Output: Handing off to {target}")
                        self.current = target
                        
                    case "respond":
                        response = decision.get("output")
                        if not response or response == "(empty response)":
                            logging.warning(f"Agent {self.current} returned empty response")
                            response = f"The {self.current} has processed your request but didn't provide a specific response."
                        logging.debug(f"Final response from {self.current}: {response}")
                        logging.info(f"Output: {response}\n\n*** FINAL RESPONSE ***\n{response}\n")
                        return response
                        
                    case _:
                        logging.error(f"Unknown action type: {decision}")
                        return f"Unknown action type from {self.current}: {decision.get('action', 'None')}"
                        
            except Exception as e:
                logging.error(f"Error during orchestration: {str(e)}")
                return f"Error during processing: {str(e)}"
                
        logging.error("Maximum turns exceeded")
        raise RuntimeError("Maximum turns exceeded (possible loop)")

    # ── internal helpers ────────────────────────────────────────────────
    def _loop_guard(self, agent: str, decision: dict) -> bool:
        key = json.dumps(decision, sort_keys=True)
        self._history[agent].append(key)
        return len(set(self._history[agent])) == 1 and len(self._history[agent]) >= 3

# ════════════════════════════════════════════════════════════════════════
#  Misc helpers
# ════════════════════════════════════════════════════════════════════════

def _safe_json(raw: str) -> dict:
    """Parse JSON safely, with fallbacks for common errors."""
    # First try direct JSON parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse JSON: {raw[:100]}...")
        
        # Try to extract JSON if it's embedded in text (e.g., markdown code blocks)
        json_pattern = r'```(?:json)?\s*({.*?})\s*```'
        match = re.search(json_pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logging.warning("Failed to parse JSON from code block")
        
        # Check if it looks like JSON but has formatting issues
        if '{' in raw and '}' in raw:
            try:
                # Try to extract just the JSON part
                start = raw.find('{')
                end = raw.rfind('}')
                if start >= 0 and end > start:
                    potential_json = raw[start:end+1]
                    return json.loads(potential_json)
            except json.JSONDecodeError:
                logging.warning("Failed to extract JSON from malformed string")
        
        # Default fallback - treat as plain text response
        return {"action": "respond", "output": raw.strip()}

# Helper function for pretty printing JSON
def pretty_print_json(data):
    """Format JSON data with proper indentation and colorification for display purposes.
    Keys are displayed in blue, strings in green, numbers in yellow, and booleans in magenta.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data
    
    # Use a custom function to colorify the JSON output
    return _colorify_json(data)

def _colorify_json(data, indent=0):
    """Helper function to recursively colorify JSON data."""
    if isinstance(data, dict):
        result = "{"
        items = []
        for key, value in data.items():
            key_str = f"{Fore.BLUE}\"{key}\":{Style.RESET_ALL}"
            value_str = _colorify_json(value, indent + 2)
            items.append(f"{'  ' * (indent + 1)}{key_str} {value_str}")
        if items:
            result += "\n" + ",\n".join(items) + f"\n{'  ' * indent}"
        result += "}"
        return result
    elif isinstance(data, list):
        result = "["
        items = [f"{'  ' * (indent + 1)}{_colorify_json(item, indent + 1)}" for item in data]
        if items:
            result += "\n" + ",\n".join(items) + f"\n{'  ' * indent}"
        result += "]"
        return result
    elif isinstance(data, str):
        return f"{Fore.GREEN}\"{data}\":{Style.RESET_ALL}"
    elif isinstance(data, (int, float)):
        return f"{Fore.YELLOW}{data}{Style.RESET_ALL}"
    elif isinstance(data, bool) or data is None:
        return f"{Fore.MAGENTA}{json.dumps(data)}{Style.RESET_ALL}"
    else:
        return str(data)

