from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY") or not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("Missing API keys!")

from langchain.chat_models import init_chat_model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")



import os
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from networkx.algorithms.community import modularity_max
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load NLP model and sentence transformer model
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Step 1: Load Documents with Metadata Preservation

folder_path = "D:/LECTURE/Y4S2/DSA4265/DSA4265/HDB_docs"  # put your own file path to the HDB docs
all_docs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Add source metadata to all pages
        for doc in docs:
            doc.metadata['source'] = file_path  # Ensure source is preserved
        all_docs.extend(docs)

# Step 2: Text Chunking with Metadata Inheritance

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,  # Reduced from 300
    add_start_index=True,
    separators=["\n\n\n", "\n\n", "(?<=\\. )", " "]  # Better sentence-aware splits
)
all_splits = text_splitter.split_documents(all_docs)

# Verify metadata in splits
print("Sample chunk metadata:", all_splits[0].metadata)

# Step 3: Entity Extraction with Coreference Resolution
def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in {"ORG", "GPE", "PERSON", "EVENT", "PRODUCT"}:
            # Normalize entity text and handle coreferences
            clean_ent = ent.text.strip().replace('\n', ' ')
            entities[clean_ent] = ent.label_
    return entities

# Step 4: Relationship Extraction with Context Awareness
def extract_relationships(chunks):
    relations = []
    for chunk in chunks:
        entities = list(extract_entities(chunk.page_content).keys())
        # Create bidirectional relationships with context
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                relations.append((entities[i], "related_to", entities[j]))
                relations.append((entities[j], "related_to", entities[i]))  # Bidirectional
    return relations

# Step 5: Knowledge Graph Construction with Full Metadata
knowledge_graph = nx.DiGraph()
entity_mapping = {}

for chunk in all_splits:

    # Preserve metadata with fallbacks
    metadata = chunk.metadata.copy()
    metadata.setdefault('source', 'Unknown')
    metadata.setdefault('page', 0)

    # Document node with full metadata
    doc_embedding = sentence_model.encode(chunk.page_content)
    doc_node_id = f"doc_{len(knowledge_graph.nodes)}"
    knowledge_graph.add_node(
        doc_node_id,
        type="document",
        embedding=doc_embedding,
        content=chunk.page_content,
        metadata=metadata,  #  Store all metadata
        label=f"Document {doc_node_id}"
    )

    # Entity handling with normalization
    entities = extract_entities(chunk.page_content)
    for entity, label in entities.items():
        clean_entity = entity.strip().replace('\n', ' ')
        if clean_entity not in entity_mapping:
            knowledge_graph.add_node(
                clean_entity,
                label=label,
                type="entity"
            )
            entity_mapping[clean_entity] = label

        # Connect document to entity
        knowledge_graph.add_edge(
            doc_node_id, clean_entity,
            relation="mentions",
            weight=1.0
        )

    # Add entity relationships
    for source, rel, target in extract_relationships([chunk]):
        clean_source = source.strip().replace('\n', ' ')
        clean_target = target.strip().replace('\n', ' ')
        if clean_source in entity_mapping and clean_target in entity_mapping:
            knowledge_graph.add_edge(
                clean_source, clean_target,
                relation=rel,
                weight=1.0
            )

# Step 6: Community Detection with Enhanced Visualization
def detect_communities():
    return list(nx.algorithms.community.louvain_communities(knowledge_graph))

def visualize_graph():
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(knowledge_graph, k=0.2, seed=42)

    # Node styling
    node_colors = []
    for node in knowledge_graph.nodes():
        if 'document' in knowledge_graph.nodes[node].get('type', ''):
            node_colors.append('lightgreen')
        else:
            node_colors.append('skyblue')

    nx.draw_networkx_nodes(knowledge_graph, pos, node_size=800, node_color=node_colors)
    nx.draw_networkx_edges(knowledge_graph, pos, alpha=0.3, width=1.5)

    # Label formatting
    labels = {n: d.get('label', n) for n, d in knowledge_graph.nodes(data=True)}
    nx.draw_networkx_labels(knowledge_graph, pos, labels, font_size=9)

    plt.title("Knowledge Graph with Document Metadata and Entity Relationships")
    plt.axis('off')
    plt.show()

# if len(knowledge_graph.nodes) > 0:
#     visualize_graph()

import os
import json
import re
import random
import string
from typing import List, Tuple, Optional, TypedDict
from difflib import SequenceMatcher
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory

# External components assumed to be defined elsewhere
from interactive_bot_graph import graph, llm, sentence_model, knowledge_graph

# ---- User Profile ----
user_profile = {
    "age": None,
    "income": None,
    "flat_type": None,
    "relationship_status": None,
}
user_memory_store = {}
chat_history = []
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
PROFILE_PATH = "user_profile.json"

def generate_user_id(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def check_existing_user(serial_code=None):
    if serial_code and serial_code.strip().upper() in user_memory_store:
        serial_code = serial_code.strip().upper()
        user_profile.update(user_memory_store[serial_code])
        return serial_code, f"Welcome back! Found your profile with ID {serial_code}. How can I help you today?"
    else:
        new_id = generate_user_id()
        user_memory_store[new_id] = {}
        return new_id, f"New session started. Your serial code is: {new_id}. Save this to continue later!"

def save_user_profile():
    with open(PROFILE_PATH, "w") as f:
        json.dump(user_profile, f, indent=2)

def load_user_profile():
    global user_profile
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r") as f:
            user_profile.update(json.load(f))

def extract_age(query: str):
    match = re.search(r'\b(?:i am|i’m|im)?\s*(\d{2})\s*(?:years old|y/o|yo|yrs)?\b', query.lower())
    return int(match.group(1)) if match else None

def extract_income(query: str):
    match = re.search(r'\$?\s?(\d{3,5})', query)
    return int(match.group(1)) if match else None

def extract_relationship(query: str):
    q = query.lower()
    if any(w in q for w in ["fiance", "fiancée"]): return "fiance"
    if any(w in q for w in ["married", "spouse", "wife", "husband"]): return "married"
    if "divorced" in q: return "divorced"
    if "widowed" in q or "orphan" in q: return "widowed"
    if "single" in q: return "single"
    return None

def extract_flat_type(query: str):
    q = query.lower()
    if any(word in q for word in ["both", "not sure", "unsure", "either", "any"]):
        return "both"
    if "bto" in q:
        return "bto"
    if "resale" in q:
        return "resale"
    return None

def update_user_profile(query: str):
    user_profile["age"] = extract_age(query) or user_profile["age"]
    user_profile["income"] = extract_income(query) or user_profile["income"]
    user_profile["relationship_status"] = extract_relationship(query) or user_profile["relationship_status"]
    user_profile["flat_type"] = extract_flat_type(query) or user_profile["flat_type"]

def ask_missing_fields():
    prompts = {
        "age": " What is your age? ",
        "income": " What is your monthly income? ",
        "relationship_status": " What is your relationship status? ",
        "flat_type": " Are you interested in a BTO or resale flat? ",
    }
    follow_up_questions = [prompt for key, prompt in prompts.items() if not user_profile[key]]
    return " ".join(follow_up_questions) if follow_up_questions else ""

def interactive_chatbot(user_input, serial_code=None):
    session_id, welcome_message = check_existing_user(serial_code)

    update_user_profile(user_input)
    save_user_profile()

    chat_history.append(f"User: {user_input}")
    result = graph.invoke({
        "question": user_input,
        "hypothetical_doc": None,
        "context": [],
        "answer": None,
        "messages": chat_history
    })

    answer = result["answer"]
    chat_history.append(f"Assistant: {answer}")

    missing_prompt = ask_missing_fields()
    full_response = f"{welcome_message}\n\n{answer}"
    if missing_prompt:
        full_response += f"\n\n{missing_prompt}"
    return full_response

