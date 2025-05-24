"""
Tools here mimic the API of different things
"""

from __future__ import annotations
import random, datetime as dt, logging, json
from src.core import tool, LLM, AgentConfig, BaseAgent, Orchestrator
from openai import OpenAI  # pip install openai>=1.0
from dotenv import load_dotenv
load_dotenv()

# Custom log filter to remove HTTP request information
class HTTPRequestFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str) and "HTTP Request:" in record.msg:
            return False
        return True

# Apply the filter to the root logger
logging.getLogger().addFilter(HTTPRequestFilter())

# ── 1  Mock tools ───────────────────────────────────────────────────────

@tool()
def get_datetime(tz: str | None = None) -> str:
    """
    Return the current UTC time in ISO format or converted to *tz*.

    args
    """
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    if tz:
        try:
            import zoneinfo
            now = now.astimezone(zoneinfo.ZoneInfo(tz))
        except Exception:
            pass
    return now.isoformat(timespec="seconds")


@tool()
def search_flights(origin: str, dest: str, date: str) -> list[dict]:
    """Return up to 3 mock flight options.

    Parameters
    ----------
    origin, dest : str
        Airport / city codes.
    date : str
        Departure date "YYYY‑MM‑DD".

    Returns
    -------
    list[dict]
        Example ::
            [{"id": "F123", "depart": "2025-06-15T07:00", "arrive": "2025-06-15T08:15", "price": 129.0}, ...]
    """
    if random.random() < 0.2:
        return []  # simulate no availability
    opts = []
    for i in range(random.randint(1, 3)):
        price = round(random.uniform(80, 200), 2)
        opts.append({
            "id": f"F{random.randint(100, 999)}",
            "depart": f"{date}T{6+i}:00",
            "arrive": f"{date}T{7+i}:15",
            "price": price,
        })
    return opts


@tool()
def search_hotels(city: str, check_in: str, nights: int = 1) -> list[dict]:
    """Mock hotel search returning at most 2 options.

    Parameters
    ----------
    city : str
    check_in : str  (YYYY‑MM‑DD)
    nights : int

    Returns
    -------
    list[dict]
        [{"name":"Grand", "nightly":150, "rating":4.3}, ...]
    """
    if random.random() < 0.3:
        return []
    return [{
        "name": random.choice(["Grand", "Ibis", "Hilton"]),
        "nightly": random.randint(90, 180),
        "rating": round(random.uniform(3.5, 5.0), 1),
    } for _ in range(random.randint(1, 2))]


@tool()
def get_weather(city: str, date: str) -> dict:
    """Return weather with a  chance of rain.

    Returns :: {"forecast":"sunny|rain", "high":, "low":15}
    """
    forecast = random.choice(["sunny", "sunny", "rain"])  # 1⁄3 rain chance
    high = random.randint(18, 30)
    low = high - random.randint(4, 8)
    return {"forecast": forecast, "high": high, "low": low}


@tool()
def calc_budget(*costs: float) -> float:
    """Return the sum of *costs* rounded to 2 decimals."""
    return round(sum(costs), 2)


# ── 2  Agents ───────────────────────────────────────────────────────────

client = OpenAI()
llm_default = LLM("gpt-4o", client)

class IntentAgent(BaseAgent):
    def __init__(self):
        cfg = AgentConfig(
            name="IntentAgent",
            llm=llm_default,
            tools=[],  # relies on LLM only
            allowed_agents=["FlightAgent"],
            allowed_agent_descriptions={
                "FlightAgent": "Searches for available flights that match user requirements",
                "ErrorAgent": "Handles error cases and requests additional information from the user",
            },
            description="Extracts travel intent, origin, destination, date and budget from user input",
            task=("Extract origin, destination, date and budget from the user paragraph."
                  " If anything is missing ask the user via respond; else handoff to FlightAgent."),
            temperature=0.2,
        )
        super().__init__(cfg)

    # very lightweight — rely on LLM parsing


class FlightAgent(BaseAgent):
    def __init__(self):
        cfg = AgentConfig(
            name="FlightAgent",
            llm=llm_default,
            tools=["search_flights"],
            allowed_agents=["WeatherAgent", "ErrorAgent"],
            allowed_agent_descriptions={
                "WeatherAgent": "Checks weather conditions at the destination",
                "ErrorAgent": "Handles error cases and requests additional information from the user"
            },
            description="Searches for available flights that match user requirements",
            task="Find the cheapest flight that meets date & budget.  If none → ErrorAgent.",
            temperature=0.4,
        )
        super().__init__(cfg)


class WeatherAgent(BaseAgent):
    def __init__(self):
        cfg = AgentConfig(
            name="WeatherAgent",
            llm=llm_default,
            tools=["get_weather"],
            allowed_agents=["BudgetAgent", "ErrorAgent"],
            allowed_agent_descriptions={
                "BudgetAgent": "Calculates and validates total trip cost against user budget",
                "ErrorAgent": "Handles error cases and requests additional information from the user"
            },
            description="Checks weather conditions at the destination",
            task=("Check forecast at destination.  If rain is predicted → ErrorAgent asking for new date;"
                  " else hand‑off to BudgetAgent."),
        )
        super().__init__(cfg)


class BudgetAgent(BaseAgent):
    def __init__(self):
        cfg = AgentConfig(
            name="BudgetAgent",
            llm=llm_default,
            tools=["calc_budget"],
            allowed_agents=["PlannerAgent", "ErrorAgent"],
            allowed_agent_descriptions={
                "PlannerAgent": "Creates a detailed travel itinerary based on all collected information",
                "ErrorAgent": "Handles error cases and requests additional information from the user"
            },
            description="Calculates and validates total trip cost against user budget",
            task=("Add up flight and optional hotel cost; if > user budget+10% → ErrorAgent, else PlannerAgent."),
        )
        super().__init__(cfg)


class PlannerAgent(BaseAgent):
    def __init__(self):
        cfg = AgentConfig(
            name="PlannerAgent",
            llm=llm_default,
            tools=[],
            allowed_agents=[],
            description="Creates a detailed travel itinerary based on all collected information",
            task="Produce a friendly itinerary summarising choices in markdown and respond to user.",
        )
        super().__init__(cfg)

class ErrorAgent(BaseAgent):
    def __init__(self):
        cfg = AgentConfig(
            name="ErrorAgent",
            llm=llm_default,
            tools=[],
            allowed_agents=[],
            description="Handles error cases and requests additional information from the user",
            task="Give the current context, solution found so far BUT definitely ask for revised inputs or any missing info. Don't ask unnecessary info",
        )
        super().__init__(cfg)


# ── 3  Wiring & demo ────────────────────────────────────────────────────

def build_orchestrator() -> Orchestrator:
    agents = {
        "IntentAgent": IntentAgent(),
        "FlightAgent": FlightAgent(),
        "WeatherAgent": WeatherAgent(),
        "BudgetAgent": BudgetAgent(),
        "PlannerAgent": PlannerAgent(),
        "ErrorAgent": ErrorAgent(),
    }
    return Orchestrator(agents)


if __name__ == "__main__":
    # Set logging level to INFO to show the formatted output
    logging.getLogger().setLevel(logging.INFO)
    
    # Create a custom formatter that will filter out HTTP request logs
    class NoHTTPRequestFormatter(logging.Formatter):
        def format(self, record):
            if isinstance(record.msg, str) and "HTTP Request:" in record.msg:
                return ""  # Return empty string for HTTP request logs
            return super().format(record)
    
    # Apply the custom formatter to all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(NoHTTPRequestFormatter('%(message)s'))
    
    print("Starting agent demo with improved logging format...\n")
    
    # Demo 1: Complete information
    print("\n▶ DEMO #1 — Complete information\n")
    orch = build_orchestrator()
    query1 = ("I'm going on a trip to Paris from Delhi on 2025-08-08 between 06:00-12:00. "
             "I must return on 2025-08-15 between 10:00-18:00. "
             "My max budget is $500. "
             "How can I reach there, where can I stay, what can I do, "
             "and what's the overall cost?")
    print(f"User query: {query1}")
    response1 = orch.start("IntentAgent", query1)
    # Final response is already printed by the logging system
    
    # Demo 2: Incomplete information requiring clarification
    print("\n\n▶ DEMO #2 — Incomplete info → clarification\n")
    orch = build_orchestrator()  # fresh state for clean demo
    query2 = ("I'm going on a trip to Paris on 2025-08-08 between 06:00-12:00. "
             "I must return on 2025-08-15 between 10:00-18:00. "
             "My max budget is $500. "
             "How can I reach there, where can I stay, what can I do, "
             "and what's the overall cost?")

    print(f"User query: {query2}")
    response2 = orch.start("IntentAgent", query2)
    # Final response is already printed by the logging system

