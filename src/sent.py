#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:41:43 2026

@author: mdraminski
"""

from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
    "Beginning in 1914, the Yuan Shikai coinage, designed by Tianjin Mint engraver Luigi Giorgi, was struck by the Republic of China to replace imperial and foreign coins then in circulation. They depict Yuan Shikai on the obverse (pictured), and a wreath of grain and the denomination of one yuan on the reverse. Coins were produced by mints across China. Until 1920, all coins were dated Republican Year 3 (1914) regardless of when they were struck. The Nationalist government ordered an end to their production in 1929, but striking continued, with poorer-quality examples produced in Communist-held areas during the 1930s. Later issues were in response to hyperinflation during the Chinese Civil War, including a large run of coins at Canton in 1949. They were struck again in the mid-1950s for use in newly annexed Tibet and rural southwestern China. In total, around 1.1 billion Yuan Shikai dollars were produced from 1914 to 1954, not including local issues produced by warlords or revolutionaries.",    
    "money",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])




########
# ChatGPT

from openai import OpenAI
client = OpenAI(api_key='REDACTED_OPENAI_API_KEY')

response = client.responses.create(
    model="gpt-3.5-turbo",    
    input="Napisz jedno zdanie o Warszawie."
)

print(response.output[0].content[0].text)



    


########
# OLLAMA

from ollama import chat
messages = [{
    "role": "user",
    "content": "Explain what Python is in one sentence.",
    },]
response = chat(model="llama3.2:latest", messages=messages)
print(response.message.content)
#Python is a high-level, interpreted programming language that is widely used
#for its simplicity, readability, and versatility, making it an ideal choice
#for web development, data analysis, machine learning, automation, and more.
