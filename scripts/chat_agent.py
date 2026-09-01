"""Chat with the shopping agent using your own product requests."""

from __future__ import annotations

import uuid
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.agent import Agent  # noqa: E402


TOP_K = 10
MAX_TURNS = 10


def display_response(agent: Agent, response: dict) -> None:
    message = str(response.get("message") or "").strip()
    if message:
        print(f"\nAgent: {message}")

    recommendations = response.get("recommendations") or []
    if not recommendations:
        print("\nNo products recommended yet.")
        return

    print("\nRecommendations:")
    for position, recommendation in enumerate(recommendations, start=1):
        asin = str(recommendation.get("parent_asin") or "")
        product = agent.catalog.get(asin, {})
        title = product.get("title") or "Untitled product"
        price = product.get("price")
        rating = product.get("average_rating")
        details = [f"ASIN: {asin}"]
        if price not in (None, ""):
            details.append(f"${price}")
        if rating not in (None, ""):
            details.append(f"rating: {rating}/5")
        print(f"  {position}. {title}")
        print(f"     {' | '.join(details)}")


def main() -> None:
    print("Loading the product catalog...")
    agent = Agent(PROJECT_ROOT / "data/catalog.jsonl")
    session_id = str(uuid.uuid4())
    agent.reset(session_id, {})
    turn = 1

    print("Ready. Ask for a product, answer follow-up questions normally, or type /quit.")
    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_message:
            continue
        if user_message.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            return
        if user_message.lower() == "/reset":
            session_id = str(uuid.uuid4())
            agent.reset(session_id, {})
            turn = 1
            print("Started a new shopping session.")
            continue

        response = agent.respond(session_id, user_message, turn=turn, top_k=TOP_K)
        display_response(agent, response)
        turn += 1

        if turn == MAX_TURNS + 1:
            print("\nReached 10 turns. Type /reset to start another session or /quit to exit.")


if __name__ == "__main__":
    main()
