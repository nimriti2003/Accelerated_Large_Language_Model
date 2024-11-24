from src.re_router import ReRouter
from src.fuzzy_logic import FuzzyLogicController
from src.sub_llm_handler import SubLLMHandler

if __name__ == "__main__":
    # Train Re-Router
    re_router = ReRouter()
    re_router.train("data/query_data.csv")

    # Train Fuzzy Logic
    fuzzy_logic = FuzzyLogicController()
    fuzzy_logic.train()

    # Handle queries
    handler = SubLLMHandler()
    user_query = "What are the top AI trends?"
    response = handler.handle_query(user_query)
    print(f"Response: {response}")
