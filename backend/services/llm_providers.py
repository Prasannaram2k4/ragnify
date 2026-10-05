import os
import re
from typing import List, Tuple

import requests

SYSTEM = 'Answer the question using only the provided context. If the answer is not in the context, say "I don\'t know."'
_STOP = {'the', 'a', 'an', 'is', 'of', 'to', 'in', 'and', 'what', 'how', 'does', 'do', 'are', 'it'}


class LLMError(Exception):
    pass


def _chat(url: str, key: str, model: str, prompt: str, max_tokens: int) -> str:
    resp = requests.post(
        url,
        headers={'Authorization': f'Bearer {key}'},
        json={
            'model': model,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': 0.0,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()


def call_openai(prompt: str, max_tokens: int = 512) -> str:
    key = os.getenv('OPENAI_API_KEY', '')
    if not key:
        raise LLMError('OPENAI_API_KEY not configured')
    return _chat('https://api.openai.com/v1/chat/completions', key,
                 os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), prompt, max_tokens)


def call_openrouter(prompt: str, max_tokens: int = 512) -> str:
    key = os.getenv('OPENROUTER_API_KEY', '')
    if not key:
        raise LLMError('OPENROUTER_API_KEY not configured')
    return _chat('https://openrouter.ai/api/v1/chat/completions', key,
                 os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.3-70b-instruct:free'), prompt, max_tokens)


def call_huggingface(prompt: str, max_tokens: int = 512) -> str:
    key = os.getenv('HF_API_TOKEN', '')
    if not key:
        raise LLMError('HF_API_TOKEN not configured')
    return _chat('https://router.huggingface.co/v1/chat/completions', key,
                 os.getenv('HF_MODEL', 'Qwen/Qwen2.5-7B-Instruct'), prompt, max_tokens)


def call_anthropic(prompt: str, max_tokens: int = 512) -> str:
    key = os.getenv('ANTHROPIC_API_KEY', '')
    if not key:
        raise LLMError('ANTHROPIC_API_KEY not configured')
    resp = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={'x-api-key': key, 'anthropic-version': '2023-06-01'},
        json={
            'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-5-haiku-latest'),
            'system': SYSTEM,
            'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
        },
        timeout=45,
    )
    resp.raise_for_status()
    return ''.join(b.get('text', '') for b in resp.json()['content']).strip()


def resolve_provider() -> str:
    p = os.getenv('LLM_PROVIDER', 'auto').lower()
    if p != 'auto':
        return p
    if os.getenv('OPENROUTER_API_KEY'):
        return 'openrouter'
    if os.getenv('OPENAI_API_KEY'):
        return 'openai'
    if os.getenv('ANTHROPIC_API_KEY'):
        return 'anthropic'
    if os.getenv('HF_API_TOKEN'):
        return 'huggingface'
    return 'extractive'


def extractive_answer(question: str, chunks: List[str]) -> str:
    """Key-free fallback: return the sentences from retrieved chunks that best overlap the question."""
    q_terms = set(re.findall(r'[a-z0-9]+', question.lower())) - _STOP
    sentences = [s.strip() for c in chunks for s in re.split(r'(?<=[.!?])\s+|\n+', c) if len(s.strip()) > 20]
    scored = sorted(
        ((len(q_terms & set(re.findall(r'[a-z0-9]+', s.lower()))), s) for s in sentences),
        key=lambda x: x[0], reverse=True,
    )
    best = [s for score, s in scored[:3] if score > 0]
    return ' '.join(best) if best else "I don't know."


def generate(question: str, chunks: List[str]) -> Tuple[str, str]:
    """Return (answer, provider_used); falls back to an extractive answer if the LLM fails."""
    context = '\n\n'.join(f'[{i + 1}] {c}' for i, c in enumerate(chunks))
    prompt = f'Context:\n{context}\n\nQuestion: {question}\nAnswer:'
    fn = {'openrouter': call_openrouter, 'openai': call_openai, 'anthropic': call_anthropic,
          'huggingface': call_huggingface}.get(resolve_provider())
    if fn:
        try:
            ans = fn(prompt)
            if ans:
                return ans, resolve_provider()
        except Exception:
            pass
    return extractive_answer(question, chunks), 'extractive'
