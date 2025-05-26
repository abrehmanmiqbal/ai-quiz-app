import requests

def generate_mcqs(topic, num_questions=10):
    api_key = "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"  # Replace with your actual key or keep it private via env
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Generate {num_questions} multiple-choice questions on the topic: "{topic}".
Each question must have:
- A clear question statement
- 4 options
- The correct answer labeled

Return format (as JSON list):
[{{"question": "...", "options": ["A", "B", "C", "D"], "answer": "..."}}]
"""

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt.strip()}
        ]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        result = response.json()

        # extract generated text
        content = result["choices"][0]["message"]["content"]

        # try to parse as JSON
        import json
        questions = json.loads(content)
        return questions if isinstance(questions, list) else []
    except Exception as e:
        print("Error:", e)
        return []
