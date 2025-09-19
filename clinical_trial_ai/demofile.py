from backend.services.ai_layer.llm_service import LLMService

# Test LLM service
llm_service = LLMService()
print("LLM Service Status:", llm_service.get_provider_info())

# if llm_service.is_configured():
#     # Test a simple query
#     response = await llm_service.generate_with_context(
#         query="What is a clinical trial?",
#         context="A clinical trial is a research study that tests new treatments."
#     )
#     print("Test Response:", response)
# else:
#     print("LLM service not configured. Please set up your API key.")