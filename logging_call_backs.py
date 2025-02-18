from langchain.callbacks.base import BaseCallbackHandler
from write_to_excel import write_to_excel
import streamlit as st

class LoggingCallbacks(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when LLM starts running."""
        print(f"🔄 Starting new query with {len(prompts)} prompts")
        
    def on_llm_end(self, response, **kwargs):
        """Log when LLM ends running."""
        # You can access token usage if available in the response
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            print("📊 Token Usage for Query:")
            print(f"- Input tokens: {usage.get('prompt_tokens', 'N/A')}. Price: ${usage.get('prompt_tokens', 'N/A') * 0.00000015:0.8f}")
            print(f"- Output tokens: {usage.get('completion_tokens', 'N/A')}. Price: ${usage.get('completion_tokens', 'N/A') * 0.0000006:0.8f}")

            total_cost = usage.get('prompt_tokens', 'N/A') * 0.00000015 + usage.get('completion_tokens', 'N/A') * 0.0000006
            
            # write to excel
            write_to_excel(usage.get('prompt_tokens', 'N/A'), usage.get('completion_tokens', 'N/A'), usage.get('total_tokens', 'N/A'), total_cost, "Query")

