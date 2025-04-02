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

from typing import List, Tuple, Optional, TypedDict
from langchain_core.documents import Document
import re
from collections import defaultdict
import hashlib
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

# ---- LLM and sentence model (must be defined beforehand) ----
# llm = ChatOpenAI()
# sentence_model = SentenceTransformer(...)
# knowledge_graph = your NetworkX graph with embedded document nodes

# ---- Shared memory structure ----
user_profile = {
    "age": None,
    "income": None,
    "flat_type": None,
    "relationship_status": None,
}

# ---- LangGraph-compatible State ----
class State(TypedDict):
    question: str
    hypothetical_doc: Optional[str]
    context: List[Tuple[Document, float]]
    answer: Optional[str]
    messages: List[str]

# ---- User profile utilities ----
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
    
    # Handle both / open-ended cases
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
    
    for key, prompt in prompts.items():
        if not user_profile[key]:
            response = input(prompt)
            update_user_profile(response)

from langchain.memory import ConversationBufferMemory
import json
import os

# 1. Short-term memory (conversation tracking)
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 2. Long-term memory persistence (user profile)
PROFILE_PATH = "user_profile.json"

def save_user_profile():
    with open(PROFILE_PATH, "w") as f:
        json.dump(user_profile, f)

def load_user_profile():
    global user_profile
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r") as f:
            user_profile = json.load(f)
        print(" User profile loaded.")
    else:
        print(" No saved profile found. Using defaults.")

# Load profile at startup
load_user_profile()

# ---- LangGraph nodes ----
def generate_hypothetical_node(state: State) -> State:
    profile = user_profile
    flat = profile.get("flat_type")

    # Flat type context handling
    if flat == "bto":
        flat_context = "The user is interested in a BTO flat."
    elif flat == "resale":
        flat_context = "The user is interested in a resale flat."
    elif flat == "both":
        flat_context = (
            "The user is open to both BTO and resale flats. "
            "If policies differ, simulate an answer that includes both perspectives."
        )
    else:
        flat_context = (
            "The user has not specified flat type. "
            "Please simulate an answer that covers both BTO and resale paths if relevant."
        )

    # Full profile summary
    profile_summary = f"""User Profile:
- Age: {profile.get('age')}
- Income: {profile.get('income')}
- Relationship: {profile.get('relationship_status')}
- Flat Type: {flat or 'unspecified'}

{flat_context}
"""

    # HyDE generation prompt
    response = llm.invoke([
        {"role": "system", "content": f"Generate a hypothetical answer as if it came from a government housing policy document. Use the user's profile below:\n\n{profile_summary}"},
        {"role": "user", "content": state["question"]}
    ])
    
    state["hypothetical_doc"] = response.content
    return state



def retrieve_node(state: State) -> State:
    hyde_embedding = sentence_model.encode(state["hypothetical_doc"])
    results = []
    seen_hashes = set()
    for node_id, node_data in knowledge_graph.nodes(data=True):
        if node_data.get('type') != 'document': continue
        doc_embed = node_data.get('embedding')
        if doc_embed is None: continue
        score = cosine_similarity(
            hyde_embedding.reshape(1, -1), 
            doc_embed.reshape(1, -1)
        )[0][0]
        doc = Document(
            page_content=node_data.get('content', ''),
            metadata=node_data.get('metadata', {})
        )
        hash_ = hashlib.md5(doc.page_content.encode()).hexdigest()
        if hash_ not in seen_hashes:
            results.append((doc, score))
            seen_hashes.add(hash_)
    results.sort(key=lambda x: x[1], reverse=True)
    state["context"] = results[:3]
    return state

def generate_node(state: State) -> State:
    context_text = "\n\n".join([doc.page_content for doc, _ in state["context"]])
    # history = "\n".join(state.get("messages", [])[-5:])
    history = chat_memory.load_memory_variables({}).get("chat_history", "")
    
    # Add profile logic
    profile = user_profile
    flat = profile.get("flat_type")

    # Flat type context handling
    if flat == "bto":
        flat_context = "The user is interested in a BTO flat."
    elif flat == "resale":
        flat_context = "The user is interested in a resale flat."
    elif flat == "both":
        flat_context = (
            "The user is open to both BTO and resale flats. "
            "If policies differ, explain both options side by side."
        )
    else:
        flat_context = (
            "The user has not specified BTO or resale. "
            "If the answer depends on flat type, explain both options clearly."
        )

    # Full profile summary to include in prompt
    profile_summary = f"""
User Profile:
- Age: {profile.get('age')}
- Income: {profile.get('income')}
- Relationship: {profile.get('relationship_status')}
- Flat Type: {flat or 'unspecified'}

{flat_context}
"""

    # Prompt messages
    prompt = [
        {"role": "system", "content": f"""You are an HDB assistant. Use the following user profile and retrieved documents to answer accurately.

{profile_summary}

Retrieved Context:
{context_text}

Chat History:
{history if history else "No prior chat history."}
""" },
        {"role": "user", "content": state['question']}
    ]
    
    response = llm.invoke(prompt)

    # Track conversation turn in memory
    chat_memory.save_context({"input": state["question"]}, {"output": response.content})

    state["answer"] = response.content
    return state



def fact_check_answer(answer: str, docs: List, threshold: float = 0.6) -> bool:
    answer_sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]

    # Gracefully handle both Document and (Document, score)
    cleaned_docs = [doc[0] if isinstance(doc, tuple) else doc for doc in docs]

    for sentence in answer_sentences:
        for doc in cleaned_docs:
            sim = SequenceMatcher(None, sentence.lower(), doc.page_content.lower()).ratio()
            if sim > threshold:
                return True
    return False


def fact_check_node(state: State) -> State:
    answer = state.get("answer", "")
    docs = [doc[0] if isinstance(doc, tuple) else doc for doc in state.get("context", [])]

    if not fact_check_answer(answer, docs):
        print("Fact-check failed. Switching to safe RAG answer.")

        context_text = "\n\n".join(doc.page_content for doc in docs)

        # Reuse full user profile in fallback
        profile = user_profile
        flat = profile.get("flat_type")
        if flat == "bto":
            flat_context = "The user is interested in a BTO flat."
        elif flat == "resale":
            flat_context = "The user is interested in a resale flat."
        elif flat == "both":
            flat_context = (
                "The user is open to both BTO and resale options. "
                "If policies differ, explain both clearly."
            )
        else:
            flat_context = (
                "The user has not specified flat type. "
                "If relevant, explain both BTO and resale differences."
            )

        profile_summary = f"""
User Profile:
- Age: {profile.get('age')}
- Income: {profile.get('income')}
- Relationship: {profile.get('relationship_status')}
- Flat Type: {flat or 'unspecified'}

{flat_context}
"""

        fallback_prompt = [
            {"role": "system", "content": f"""You are an HDB assistant. Always answer based on the user profile and retrieved documents.

{profile_summary}

Retrieved Context:
{context_text}
""" },
            {"role": "user", "content": state["question"]}
        ]

        fallback = llm.invoke(fallback_prompt)
        state["answer"] = fallback.content
    else:
        print("Answer passed fact-check.")

    return state

# ---- Build LangGraph ----
from langgraph.graph import StateGraph

workflow = StateGraph(State)
workflow.add_node("generate_hypothesis", generate_hypothetical_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("fact_check", fact_check_node)

workflow.set_entry_point("generate_hypothesis")
workflow.add_edge("generate_hypothesis", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "fact_check")
workflow.set_finish_point("fact_check")

graph = workflow.compile()

import random
import string

# A dictionary to store user profiles keyed by serial
user_memory_store = {}

# Generate unique user ID (e.g., 6-char alphanumeric code)
def generate_user_id(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    
def check_existing_user():
    print("Welcome back! If you're a returning user, enter your serial code. Otherwise, press Enter to start fresh.")
    user_input = input("Serial code: ").strip().upper()

    if user_input and user_input in user_memory_store:
        print(f"Found your profile with ID {user_input}. Welcome back!")
        user_profile.update(user_memory_store[user_input])
        return user_input
    else:
        new_id = generate_user_id()
        print(f"New session started. Your serial code is: {new_id}")
        print("Save this code to continue later with the same profile. And what do you want to ask?")
        return new_id

def interactive_chatbot(user_input):
    # process input and return response string
    return "some response"
    
# ---- Interactive chatbot ----
chat_history = []

def def interactive_chatbot(user_input):  # one input string, returns a response string
    print("Hello! I'm your HDB eligibility assistant.")
    session_id = check_existing_user()  # New or returning?

    while True:
        query = input("\n You: ")
        if query.lower() in ["exit", "quit"]:
            # Save user profile
            user_memory_store[session_id] = user_profile.copy()
            print(f"Profile saved under ID: {session_id}")
            print("Goodbye! Use your serial next time to continue.")
            break

        update_user_profile(query)
        ask_missing_fields()

        chat_history.append(f"User: {query}")

        result = graph.invoke({
            "question": query,
            "hypothetical_doc": None,
            "context": [],
            "answer": None,
            "messages": chat_history
        })

        answer = result["answer"]
        chat_history.append(f"Assistant: {answer}")

        print("\n", answer)
        print(f"\n Current user profile (ID: {session_id}):", user_profile)