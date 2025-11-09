import os
import requests
from fastapi import HTTPException
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT



def call_ollama(prompt, max_tokens=512):
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'phi3:instruct')
    try:
        resp = requests.post('http://localhost:11434/api/generate', json={
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'max_tokens': max_tokens
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get('choices', [{}])[0].get('message', {}).get('content', '') or data.get('text', '')
        return str(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Ollama error: {e}')


def call_openai(prompt, max_tokens=512, model='gpt-4o-mini'):
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail='OpenAI API key is not configured')
    try:
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                      {'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return resp['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'OpenAI error: {e}')


def call_anthropic(prompt, max_tokens=512, model='claude-2.1'):
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail='Anthropic API key is not configured')
    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        full_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        resp = client.completions.create(model=model, prompt=full_prompt, max_tokens_to_sample=max_tokens)
        return resp['completion']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Anthropic error: {e}')


def call_huggingface(prompt, max_tokens=512):
    HF_API_TOKEN = os.getenv('HF_API_TOKEN', '')
    # Switch to a smaller widely-available instruction model; allow override via env
    HF_MODEL = os.getenv('HF_MODEL', 'google/flan-t5-small')
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail='Hugging Face API token not configured')
    headers = {'Authorization': f'Bearer {HF_API_TOKEN}'}
    payload = {
        'inputs': prompt,
        'parameters': {'max_new_tokens': max_tokens, 'temperature': 0.0},
        'options': {'wait_for_model': True}
    }
    attempts = [
        f'https://api-inference.huggingface.co/models/{HF_MODEL}',
        f'https://router.huggingface.co/hf-inference/models/{HF_MODEL}',
    ]
    last_err = None
    for url in attempts:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    # OpenAI-like or single dict
                    if 'choices' in data and data['choices']:
                        choice = data['choices'][0]
                        return choice.get('message', {}).get('content') or choice.get('text', '') or ''
                    if 'generated_text' in data:
                        return data['generated_text']
                    # Some models return {'error': ...}
                    if 'error' in data:
                        last_err = data['error']
                        continue
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    for key in ('generated_text', 'text'):
                        if key in data[0]:
                            return data[0][key]
                    return str(data[0])
                return str(data)
            # For 404/410 try next endpoint
            if resp.status_code in (404, 410):
                last_err = resp.text
                continue
            # Raise for other failures
            resp.raise_for_status()
        except Exception as e:
            last_err = str(e)
            continue
    # Final fallback: return empty string instead of raising to let caller degrade gracefully
    return ''
