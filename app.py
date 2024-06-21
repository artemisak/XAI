from flask import Flask, render_template, request, jsonify
import anthropic
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Your Anthropic API key
ANTHROPIC_API_KEY = "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"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def split_words_in_dict(word_dict):
    for key, value in word_dict.items():
        word_dict[key] = list(set(' '.join(word_dict[key]).split()))
    return word_dict

def replace_with_mask(text, word_dict, key_word):
    text = text.replace(',', ' ,').replace('.', ' .')
    words = text.split()
    masked_words = []

    for word in words:
        if word.lower() in [v.lower() for v in word_dict.get(key_word, [])]:
            masked_words.append('[MASK]')
        else:
            masked_words.append(word)

    masked_text = ' '.join(masked_words).replace(' ,', ',').replace(' .', '.')
    return masked_text

def get_jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def get_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def get_levenshtein_distance(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return 1 - dp[m][n] / max(m, n)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    prompt = request.form['prompt']
    scheme = request.form['scheme']

    # Extract concepts
    message_2 = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="Make your answers clear and concise.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Extract up to 5 key concepts from the text below, give an answer in the form of a comma-separated list, each concept should consist of one or two words or be a named entity: {prompt}"
                    }
                ]
            }
        ]
    )

    concepts = message_2.content[0].text.split(', ')

    # Relate concepts to words
    message_3 = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="Make your answers clear and concise.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Relate these key concepts to the words in the text. Provide the answer in the form of a python dictionary, with the key being the concept and the value being a list of words that are directly or conventionally related to the concept. Key concepts: {concepts}. Text: {prompt}."
                    }
                ]
            }
        ]
    )

    concepts_with_words = split_words_in_dict(ast.literal_eval(message_3.content[0].text))

    return jsonify({
        'concepts': concepts,
        'concepts_with_words': concepts_with_words
    })

@app.route('/mask', methods=['POST'])
def mask():
    prompt = request.form['prompt']
    scheme = request.form['scheme']
    concepts = request.form.getlist('concepts[]')
    concepts_with_words = ast.literal_eval(request.form['concepts_with_words'])

    # Original answer
    message_1 = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="Make your answers clear and concise.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}{scheme}"
                    }
                ]
            }
        ]
    )

    original_answer = message_1.content[0].text

    # Masked prompt
    mask_prompt = prompt
    for concept in concepts:
        mask_prompt = replace_with_mask(mask_prompt, concepts_with_words, concept)

    # Masked answer
    message_4 = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="Make your answers clear and concise.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{mask_prompt}{scheme}"
                    }
                ]
            }
        ]
    )

    masked_answer = message_4.content[0].text

    # Calculate similarities
    jaccard = get_jaccard_similarity(original_answer, masked_answer)
    cosine = get_cosine_similarity(original_answer, masked_answer)
    levenshtein = get_levenshtein_distance(original_answer, masked_answer)

    return jsonify({
        'original_prompt': prompt,
        'masked_prompt': mask_prompt,
        'original_answer': original_answer,
        'masked_answer': masked_answer,
        'jaccard': jaccard,
        'cosine': cosine,
        'levenshtein': levenshtein
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
