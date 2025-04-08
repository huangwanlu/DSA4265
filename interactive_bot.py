import hashlib
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pdfplumber
import random
import re
import spacy
import string
import uuid
from collections import defaultdict
from difflib import SequenceMatcher
from dotenv import load_dotenv
from flask import session
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from networkx.algorithms.community import modularity_max
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, TypedDict
from langchain.memory import ConversationSummaryBufferMemory

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY") or not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("Missing API keys!")

from langchain.chat_models import init_chat_model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")


# === New Function: Extract Headings from PDF ===
def extract_headings_from_pdf(pdf_path):
    headings = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            words = page.extract_words(extra_attrs=["size", "fontname"])
            for word in words:
                if word["size"] >= 14 or "Bold" in word["fontname"]:
                    text = word["text"].strip()
                    if len(text) > 3:
                        headings.append({
                            "page": i,
                            "text": text,
                            "size": word["size"],
                            "font": word["fontname"]
                        })
    return headings


# === New Function: Match Heading to Policy Track ===
def infer_policy_track_from_heading(heading):
    heading = heading.lower()
    if "single" in heading and "bto" in heading:
        return "bto_singles"
    elif "couple" in heading and "bto" in heading:
        return "bto_couples"
    elif "single" in heading and "resale" in heading:
        return "resale_singles"
    elif "couple" in heading and "resale" in heading:
        return "resale_couples"
    elif "grant" in heading:
        return "grants"
    elif "loan" in heading:
        return "loan_eligibility"
    return "unknown"

# Load NLP model and sentence transformer model
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Step 1: Load Documents with Metadata Preservation

# === Load and Tag Documents with pdfplumber Layout Awareness ===
folder_path = "D:/LECTURE/Y4S2/DSA4265/DSA4265/HDB_docs"
all_docs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        heading_list = extract_headings_from_pdf(file_path)
        heading_map = {item["page"]: infer_policy_track_from_heading(item["text"]) for item in heading_list}

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            page_num = doc.metadata.get("page", 0)
            heading = heading_map.get(page_num, "")
            policy_track = infer_policy_track_from_heading(heading)
            doc.metadata["source"] = file_path
            doc.metadata["policy_track"] = policy_track
        all_docs.extend(docs)

# Step 2: Text Chunking with Metadata Inheritance

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,  # Reduced from 300
    add_start_index=True,
    separators=["\n\n\n", "\n\n", "(?<=\\. )", " "]  # Better sentence-aware splits
)
all_splits = text_splitter.split_documents(all_docs)


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
        text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        entities = list(extract_entities(text).keys())

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                source = entities[i]
                target = entities[j]
                relations.append((source, "related_to", target))
                relations.append((target, "related_to", source))  # bidirectional
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


# Placeholder: Replace with your actual model imports
# Example:
# from langchain.chat_models import ChatOpenAI
# from sentence_transformers import SentenceTransformer
# import networkx as nx

# llm = ChatOpenAI(model="gpt-3.5-turbo")
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# knowledge_graph = nx.read_gpickle("your_graph.gpickle")

# --- USER PROFILE ---
user_profile = {
    "age": None,
    "income": None,
    "flat_type": None,
    "relationship_status": None,
    "partner_age": None,
    "partner_income": None,
    "partner_citizenship": None
}

user_memory_store = {}
chat_history = []
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
summary_memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="summary",
    return_messages=True
)

PROFILE_PATH = "user_profile.json"


# --- LangGraph State Definition ---
class State(TypedDict):
    question: str
    hypothetical_doc: Optional[str]
    context: List[Tuple[Document, float]]
    answer: Optional[str]
    messages: List[str]

import ast

def extract_user_profile_info_with_llm(query: str) -> dict:
    system_prompt = """You are an assistant that extracts housing eligibility details from user messages.
Return a Python dictionary containing only fields that are explicitly stated or clearly implied.

Extract any of the following if mentioned:
- 'age': int
- 'income': int
- 'relationship_status': one of ['single', 'married', 'fiance', 'divorced', 'widowed']
- 'flat_type': one of ['bto', 'resale', 'both']
- 'partner_age': int
- 'partner_income': int
- 'partner_citizenship': 'Singaporean', 'PR', or 'foreigner'

Examples:
User: I’m 25, my income is 4000, applying with my girlfriend for a BTO.
Output: {'age': 25, 'income': 4000, 'relationship_status': 'fiance', 'flat_type': 'bto'}

User: My partner is a PR earning 3500, and I'm single.
Output: {'relationship_status': 'single', 'partner_income': 3500, 'partner_citizenship': 'PR'}

User: I'm 27, she’s 25 and we're applying for resale.
Output: {'age': 27, 'partner_age': 25, 'relationship_status': 'fiance', 'flat_type': 'resale'}

Respond ONLY with a Python dictionary (no explanation).
"""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])

    try:
        return ast.literal_eval(response.content.strip())
    except Exception:
        return {}


# # --- User Profile Utilities ---
# def extract_age(query: str):
#     q = query.lower()
#     if "partner" in q:
#         return None
#     match = re.search(r'\b(?:i am|i’m|im)?\s*(\d{2})\s*(?:years old|y/o|yo|yrs)?\b', q)
#     return int(match.group(1)) if match else None


# def extract_income(query: str):
#     q = query.lower()

#     # If it's referring to partner, skip extracting user income
#     if "partner" in q:
#         return None

#     match = re.search(r'\$?\s?(\d{3,5})', query)
#     return int(match.group(1)) if match else None


# def extract_relationship(query: str):
#     q = query.lower()
#     if any(phrase in q for phrase in [
#         "my girlfriend", "my boyfriend", "fiance", "fiancée", "fiancé",
#         "partner", "applying together", "we are applying", "with my partner", "applying with"
#     ]):
#         return "fiance"

#     if any(w in q for w in ["married", "spouse", "wife", "husband"]): return "married"
#     if "divorced" in q: return "divorced"
#     if "widowed" in q or "orphan" in q: return "widowed"
#     if "single" in q: return "single"
#     return None


# def extract_flat_type(query: str):
#     q = query.lower()
#     if any(word in q for word in ["both", "not sure", "unsure", "either", "any"]):
#         return "both"
#     if "bto" in q: return "bto"
#     if "resale" in q: return "resale"
#     return None
    

# def extract_partner_info(query: str):
#     q = query.lower()
#     partner = {}

#     # Match income
#     income_match = re.search(r'(?:my\s+)?partner(?:\'s)?\s+(?:income|earnings|salary)?\s*(?:is|earns|earning|makes)?\s*\$?(\d{3,5})', q)
    
#     # Match age
#     age_match = re.search(r'(?:my\s+)?partner(?:\'s)?\s+age\s*(?:is)?\s*(\d{2})', q)

#     # Match citizenship
#     citizenship_match = re.search(r'(?:my\s+)?partner.*?(singapore citizen|citizen|pr|permanent resident|foreigner|non[-\s]?resident)', q)

#     if age_match:
#         partner["age"] = int(age_match.group(1))
#     if income_match:
#         partner["income"] = int(income_match.group(1))
#     if citizenship_match:
#         c = citizenship_match.group(1).strip().lower()
#         if "pr" in c or "permanent" in c:
#             partner["citizenship"] = "PR"
#         elif "citizen" in c:
#             partner["citizenship"] = "Singaporean"
#         elif "foreigner" in c or "non" in c:
#             partner["citizenship"] = "foreigner"

    # return partner

import ast

def extract_partner_info_with_llm(query: str) -> dict:
    system_prompt = """You are an assistant extracting partner-related details from the user's message for a housing eligibility chatbot.

Only return the following fields **if mentioned or strongly implied**:
- 'partner_age': int
- 'partner_income': int
- 'partner_citizenship': one of ['Singaporean', 'PR', 'foreigner']

Examples:
User: My fiancée is 24 and earns $3500. She's Singaporean.
Output: {'partner_age': 24, 'partner_income': 3500, 'partner_citizenship': 'Singaporean'}

User: My girlfriend is 25 and she’s a PR.
Output: {'partner_age': 25, 'partner_citizenship': 'PR'}

User: I'm applying with my non-citizen partner who makes 3000.
Output: {'partner_income': 3000, 'partner_citizenship': 'foreigner'}

Respond ONLY with a valid Python dictionary. Do not explain.
"""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])

    try:
        return ast.literal_eval(response.content.strip())
    except Exception:
        return {}

def format_user_profile():
    user_lines = ["👤 You:"]
    partner_lines = []

    # Group user vs partner info
    for key, value in user_profile.items():
        if value is None:
            continue

        label = key.replace("_", " ").capitalize()

        if key.startswith("partner_"):
            partner_lines.append(f"- {label.replace('Partner ', '')}: {value}")
        else:
            user_lines.append(f"- {label}: {value}")

    if len(user_lines) == 1:
        user_lines.append("⚠️ No user information provided yet.")

    if partner_lines:
        user_lines.append("\n🧑‍🤝‍🧑 Partner:")
        user_lines.extend(partner_lines)

    return "\n".join(user_lines)


def update_user_profile(query: str):
    extracted = extract_user_profile_info_with_llm(query)
    for key, value in extracted.items():
        if value:
            user_profile[key] = value

    partner_info = extract_partner_info_with_llm(query)
    for key, value in partner_info.items():
        if value:
            user_profile_key = f"partner_{key}" if not key.startswith("partner_") else key
            user_profile[user_profile_key] = value






def was_prompt_already_asked(field_key: str):
    question_keywords = {
        "age": ["your age", "how old are you"],
        "income": ["your income", "how much do you earn"],
        "relationship_status": ["your relationship status", "are you single"],
        "flat_type": ["bto or resale", "what flat type"],
        "partner_age": ["your partner's age"],
        "partner_income": ["your partner's income"],
        "partner_citizenship": ["your partner a citizen", "partner's citizenship"]
    }
    asked = any(
        any(keyword in message.lower() for keyword in question_keywords[field_key])
        for message in chat_history if isinstance(message, str)
    )
    return asked


def ask_missing_fields():
    prompts = []

    # Always ask these if missing
    base_fields = {
        "age": " What is your age? ",
        "income": " What is your monthly income? ",
        "relationship_status": " What is your relationship status? ",
        "flat_type": " Are you interested in a BTO or resale flat? ",
    }

    for key, question in base_fields.items():
        if not user_profile.get(key) and not was_prompt_already_asked(key):
            prompts.append(question)

    # Add partner info only if relationship and flat_type support it
    flat = (user_profile.get("flat_type") or "").lower()
    if user_profile.get("relationship_status") == "fiance" and flat in ["bto", "resale", "both"]:
        partner_fields = {
            "partner_age": " What is your partner's age?",
            "partner_income": " What is your partner's monthly income?",
            "partner_citizenship": " Is your partner a Singapore Citizen, PR, or foreigner?"
        }
        for key, question in partner_fields.items():
            if not user_profile.get(key) and not was_prompt_already_asked(key):
                prompts.append(question)

    return " ".join(prompts)



def format_answer_nicely(text):
    import re
    # Break after numbers like "1.", "2.", etc.
    text = re.sub(r"(?<!\n)(\d\.)", r"\n\n\1", text)
    # Convert bullet symbols to consistent dot bullets
    text = re.sub(r"\n[\*\-] ", r"\n• ", text)
    # Compact multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def save_user_profile():
    with open(PROFILE_PATH, "w") as f:
        json.dump(user_profile, f, indent=2)


def load_user_profile():
    global user_profile
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r") as f:
            user_profile.update(json.load(f))


# --- LangGraph Nodes ---
def generate_hypothetical_node(state: State) -> State:
    profile = user_profile
    flat = profile.get("flat_type")

    if flat == "bto":
        flat_context = "The user is interested in a BTO flat."
    elif flat == "resale":
        flat_context = "The user is interested in a resale flat."
    elif flat == "both":
        flat_context = "The user is open to both BTO and resale flats."
    else:
        flat_context = "The user has not specified flat type."

    profile_summary = f"""User Profile:
- Age: {profile.get('age')}
- Income: {profile.get('income')}
- Relationship: {profile.get('relationship_status')}
- Flat Type: {flat or 'unspecified'}
- Partner Age: {profile.get('partner_age')}
- Partner Income: {profile.get('partner_income')}
- Partner Citizenship: {profile.get('partner_citizenship')}

{flat_context}
"""


    system_prompt = f"""Generate a hypothetical answer from a government housing policy document. Use the user's profile below:

{profile_summary}
"""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["question"]}
    ])
    state["hypothetical_doc"] = response.content
    return state


def retrieve_node(state: State) -> State:
    from sklearn.metrics.pairwise import cosine_similarity
    import hashlib

    hyde_embedding = sentence_model.encode(state["hypothetical_doc"])
    results = []
    seen_hashes = set()

    # Step 1: Collect all unique policy_track tags in the graph
    available_tracks = set()
    for node_id, node_data in knowledge_graph.nodes(data=True):
        if node_data.get("type") == "document":
            track = node_data.get("metadata", {}).get("policy_track", "")
            if track:
                available_tracks.add(track)

    # Step 2: Profile-aware scoring of relevance
    def track_score(track: str, profile: dict):
        score = 0
        flat = (profile.get("flat_type") or "").lower()
        status = (profile.get("relationship_status") or "").lower()
        if flat and flat in track:
            score += 1
        if status and status in track:
            score += 1
        if "grants" in track or "loan" in track:
            score += 0.5  # always loosely relevant
        return score

    # Step 3: Determine relevant tags for this user
    relevant_tags = [t for t in available_tracks if track_score(t, user_profile) > 0]

    # Step 4: Filter and rank document nodes
    for node_id, node_data in knowledge_graph.nodes(data=True):
        if node_data.get("type") != "document":
            continue

        track = node_data.get("metadata", {}).get("policy_track", "")
        if track and track not in relevant_tags:
            continue

        doc_embed = node_data.get("embedding")
        if doc_embed is None:
            continue

        score = cosine_similarity(hyde_embedding.reshape(1, -1), doc_embed.reshape(1, -1))[0][0]
        doc = Document(page_content=node_data["content"], metadata=node_data.get("metadata", {}))
        hash_ = hashlib.md5(doc.page_content.encode()).hexdigest()
        if hash_ not in seen_hashes:
            results.append((doc, score))
            seen_hashes.add(hash_)

    # Step 5: Sort and keep top results
    results.sort(key=lambda x: x[1], reverse=True)
    state["context"] = results[:3]
    return state



def generate_node(state: State) -> State:
    context_text = "\n\n".join([doc.page_content for doc, _ in state["context"]])
    history_summary = summary_memory.load_memory_variables({}).get("summary", "")

    profile = user_profile
    flat = profile.get("flat_type")
    flat_context = "The user is interested in " + (flat if flat else "unspecified") + " flats."

    relationship_note = ""
    if profile.get("relationship_status") == "fiance":
        relationship_note = "Note: The user is currently unmarried but is applying with their partner, which qualifies them under the Fiancé/Fiancée Scheme (min age 21)."

    # ✅ Dynamically build partner_note based on available fields
    partner_note = ""
    partner_lines = []
    if profile.get("partner_age"):
        partner_lines.append(f"- Age: {profile['partner_age']}")
    if profile.get("partner_income"):
        partner_lines.append(f"- Income: {profile['partner_income']}")
    if profile.get("partner_citizenship"):
        partner_lines.append(f"- Citizenship: {profile['partner_citizenship']}")
    if partner_lines:
        partner_note = "👫 Partner Info:\n" + "\n".join(partner_lines)

    profile_summary = f"""
    User Profile:
    - Age: {profile.get('age')}
    - Income: {profile.get('income')}
    - Relationship Status: {profile.get('relationship_status')}
    - Flat Type: {flat or 'unspecified'}

    {flat_context}
    {relationship_note}
    {partner_note}
    
    """
    prompt = [
        {
  "role": "system",
  "content": f"""You are an HDB eligibility assistant. Use the user profile and retrieved documents below to answer the user's question.

{profile_summary}

📌 Notes:
- Do not reject users from BTO eligibility just because they identify as "single" — they may be applying under the Fiancé/Fiancée Scheme if mentioned.
- Only base your answer on the retrieved documents and known profile. Do not assume unknown values.
- If key information is missing, mention what’s needed next to confirm eligibility.
- Do not hallucinate schemes or make up rules.
- If the user mentions having a partner, cross-check if relationship status is "fiancé(e)" and extract/update partner-related info from the message.
- Avoid assuming missing details — always prefer asking follow-up questions.

📋 Formatting instructions:
- Structure your answer in clear paragraphs or bullet points.
- Separate sections by eligibility paths (e.g., Singles Scheme, Fiancé/Fiancée Scheme).
- Only list paths that may apply based on current info.
- Keep answers concise if data is incomplete.

📚 Retrieved Context:
{context_text}

🧠 Conversation Summary:
{history_summary or "No prior summary available."}

"""
},
        {"role": "user", "content": state["question"]}
    ]
    response = llm.invoke(prompt)
    summary_memory.save_context({"input": state["question"]}, {"output": response.content})
    state["answer"] = format_answer_nicely(response.content)
    
    return state


def fact_check_answer(answer: str, docs: List, threshold: float = 0.6) -> bool:
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
    cleaned_docs = [doc[0] if isinstance(doc, tuple) else doc for doc in docs]
    for sentence in sentences:
        for doc in cleaned_docs:
            sim = SequenceMatcher(None, sentence.lower(), doc.page_content.lower()).ratio()
            if sim > threshold:
                return True
    return False


def fact_check_node(state: State) -> State:
    answer = state.get("answer", "")
    docs = [doc[0] if isinstance(doc, tuple) else doc for doc in state.get("context", [])]
    if not fact_check_answer(answer, docs):
        context_text = "\n\n".join(doc.page_content for doc in docs)
        profile = user_profile
        profile_summary = f"""
User Profile:
- Age: {profile.get('age')}
- Income: {profile.get('income')}
- Relationship: {profile.get('relationship_status')}
- Flat Type: {profile.get('flat_type') or 'unspecified'}
"""
        fallback_prompt = [
            {"role": "system", "content": f"""You are an HDB assistant. Use this profile and context:

{profile_summary}
Retrieved Context:
{context_text}
""" },
            {"role": "user", "content": state["question"]}
        ]
        fallback = llm.invoke(fallback_prompt)
        state["answer"] = fallback.content
    return state

def reasoning_planner(state: State) -> State:
    from random import shuffle

    context_text = "\n\n".join([doc.page_content for doc, _ in state["context"]])
    profile = user_profile

    prompt = [
        {
            "role": "system",
            "content": f"""You're an HDB reasoning planner. Your job is to explore different *paths* the user might be eligible under, based on incomplete info.

📌 Instructions:
- List multiple eligibility paths (e.g., Singles Scheme, Fiancé/Fiancée Scheme, Joint Singles).
- Briefly explain for each path:
  • Key criteria
  • What is already fulfilled based on profile
  • What info is still missing
- Do NOT decide on the final answer. Just lay out possibilities.

📋 Format:
- Use headers like `🏠 Fiancé/Fiancée Scheme:`
- Use bullet points or short paragraphs under each
- Avoid repetition; don’t decide eligibility unless certain

Known User Profile:
{format_user_profile()}

Retrieved context:
{context_text}
"""
        },
        {"role": "user", "content": state["question"]}
    ]

    response = llm.invoke(prompt)
    state["answer"] = format_answer_nicely(response.content)
    return state


def follow_up_planner(state: State) -> State:
    missing = ask_missing_fields()
    if missing:
        state["answer"] = f"🤔 Before I can give a confident answer, I still need a few things:\n{missing}"
        # If there's something missing, stop here and respond interactively
        return state
    return state  # If no missing info, continue to generate

# ---- Build LangGraph ----
workflow = StateGraph(State)
workflow.add_node("generate_hypothesis", generate_hypothetical_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("reasoning", reasoning_planner)
workflow.add_node("follow_up", follow_up_planner)
workflow.add_node("final_response", generate_node)
workflow.add_node("fact_check", fact_check_node)
 

workflow.set_entry_point("generate_hypothesis")
workflow.add_edge("generate_hypothesis", "retrieve")
workflow.add_edge("retrieve", "reasoning")
workflow.add_edge("reasoning", "follow_up") 
workflow.add_edge("follow_up", "final_response")
workflow.add_edge("final_response", "fact_check")

workflow.set_finish_point("fact_check")

graph = workflow.compile()


def interactive_chatbot(user_input, serial_code=None):
    lower_input = user_input.strip().lower()
    if any(phrase in lower_input for phrase in ["show profile", "profile info", "my profile", "what do you know about me", "current profile", "show my info"]):
        return format_user_profile()
        
    global chat_history, user_profile

    # Handle page load intro
    if user_input.strip() == "__init__":
        session_id = serial_code or str(uuid.uuid4())[:8]
        session["session_id"] = session_id  # <-- Add this
        chat_history.clear()
        return (
            f"👋 Hi! I'm your HDB eligibility assistant.\n\n"
            f"🆔 Your session ID is: {session_id}\n\n"
            "To get started, please tell me:\n"
            "• Your age\n"
            "• Your citizenship\n"
            "• Your monthly income\n"
            "• Your relationship status (single, married, etc.)\n"
            "• Whether you're interested in a BTO or resale flat"
        )

    # Handle reset command
    reset_phrases = ["reset", "start over", "new session", "restart", "begin again", "fresh start"]
    if any(phrase in user_input.lower() for phrase in reset_phrases):
        old_session_id = serial_code or str(uuid.uuid4())[:8]

        # Archive current session
        archive_path = f"session_{old_session_id}.json"
        with open(archive_path, "w") as f:
            json.dump({
                "session_id": old_session_id,
                "chat_history": chat_history,
                "user_profile": user_profile
            }, f, indent=2)

        # Generate and store new session ID
        new_session_id = str(uuid.uuid4())[:8]
        session["session_id"] = new_session_id  

        # Reset state
        chat_history.clear()
        user_profile = {
            "age": None,
            "income": None,
            "flat_type": None,
            "relationship_status": None,
            "partner_age": None,
            "partner_income": None,
            "partner_citizenship": None
        }


        # Return fresh intro message with new ID
        intro_msg = (
            f"🔄 Session has been reset and archived. Let's start fresh!\n\n"
            f"👋 Hi! I'm your HDB eligibility assistant.\n\n"
            f"🆔 Your new session ID is: {new_session_id}\n\n"
            "To get started, please tell me:\n"
            "• Your age\n"
            "• Your citize\n"
            "• Your monthly income\n"
            "• Your relationship status (single, married, etc.)\n"
            "• Whether you're interested in a BTO or resale flat"
        )
        return intro_msg


    # Use existing session
    session_id = serial_code or str(uuid.uuid4())[:8]
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

    answer = format_answer_nicely(result["answer"])  # Apply formatting
    if len(chat_history) <= 2:
        answer = f"🆔 Your session ID is: {session_id}\n\n{answer}"

    chat_history.append(f"Assistant: {answer}")
    missing = ask_missing_fields()
    return f"{answer}\n\n{missing}" if missing else answer
