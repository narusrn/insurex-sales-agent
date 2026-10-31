"""get_llm() — provider switch (openai | azure). See implementation-plan.md §3.7."""
import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI


def get_llm(temperature: float = 0, **kwargs):
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            temperature=temperature,
            **kwargs,
        )
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        **kwargs,
    )


if __name__ == "__main__":
    # smoke test — Phase 0.3: one raw call returns a response
    from dotenv import load_dotenv
    load_dotenv()
    reply = get_llm().invoke("Say 'ok' and nothing else.")
    assert reply.content.strip(), "empty response from get_llm()"
    print("llm.py smoke test OK:", reply.content)
