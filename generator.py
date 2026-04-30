import anthropic
import json

def generate_cfo_commentary(kpis: dict, prior_period: dict) -> str:
    client = anthropic.Anthropic()
    prompt = f"""You are a CFO writing management commentary.
CURRENT PERIOD KPIs: {json.dumps(kpis)}
PRIOR PERIOD KPIs: {json.dumps(prior_period)}
"""
    message = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=400,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return message.content[0].text
