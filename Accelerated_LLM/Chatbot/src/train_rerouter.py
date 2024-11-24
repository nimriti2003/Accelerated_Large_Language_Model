from src.re_router import ReRouter

# File path to your dataset
data_path = r"C:\Users\nirmiti.deshmukh\Accelerated_LLM\Chatbot\Data\query_data.csv"

# Train the Re-Router
re_router = ReRouter(model_dir="C:/Users/nirmiti.deshmukh/Accelerated_LLM/Chatbot/models/re_router/")
re_router.train(data_path=data_path)
