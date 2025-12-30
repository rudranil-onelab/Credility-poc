import boto3
import json
from Nodes.tools.bedrock_client import get_bedrock_client

def call_llm(system_prompt: str, user_message: str):
    client = get_bedrock_client()

    response = client.chat_completion(
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no markdown or explanations.",
        temperature=0
    )

    # YOUR SDK returns string, not structured dict
    if isinstance(response, str):
        raw = response
    else:
        # fallback if SDK changes
        raw = response.get("content", [{}])[0].get("text", "")

    try:
        return json.loads(raw)
    except Exception:
        raise ValueError(f"LLM returned invalid JSON:\n{raw}")