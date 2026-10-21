
from langchain_openai import ChatOpenAI
import os
import re

EXTRA_CHUNKS = 2   # extra chunks allowed for multi-sentence questions (0 = old behaviour)

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    # 1) Search with the whole question
    primary = vectorstore.similarity_search(query, k=k)

    # 2) If the question has several sentences, also search each sentence on its own
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", query) if len(s.strip()) > 15]
    extras = []
    if len(sentences) > 1:
        per_sentence = [vectorstore.similarity_search(s, k=2) for s in sentences]
        for rank in range(2):
            for results in per_sentence:
                if rank < len(results):
                    extras.append(results[rank])

    # 3) Merge without duplicates: whole-question results first
    merged, seen = [], set()
    for r in primary + extras:
        key = r.page_content.strip()
        if key not in seen:
            seen.add(key)
            merged.append(r)
    if extras:
        merged = merged[: k + EXTRA_CHUNKS]

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in merged
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an AI assistant helping Aurora Stays guest service staff answer questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely, including amounts, deadlines, time zones, taxes, limits, conditions, and exceptions.
    4. Many policies are property-specific or depend on the rate plan. When the query names a property or rate plan, use the information for that property or plan and name it in your answer. If the query needs a property or rate plan and does not name one, say which detail is needed.
    5. Retrieved table rows are written as "Column: value" pairs. Use those labels to interpret the values.
    6. Focus only on information directly relevant to the user's query, and do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand. Do not repeat the same information in a closing summary, and do not add advice, referrals, or next steps that are not stated in the retrieved policy information.
    9. Do not invent fees, limits, deadlines, refund conditions, or exceptions. Keep qualifiers such as "generally", "normally", or "may" exactly as stated, and do not turn them into absolute statements. Include limits and exceptions that accompany a fee or rule.
    10. If the retrieved information answers only part of the question, answer that part and state clearly what is missing. If nothing relevant was retrieved, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "qwen/qwen3.8-27b",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
