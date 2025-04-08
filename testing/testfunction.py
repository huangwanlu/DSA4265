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

# Function to calculate monthly payment (EMI)
def calculate_monthly_payment(principal, interest_rate, tenure_years):
    r = interest_rate / 12 / 100  # Monthly interest rate (as a decimal)
    n = tenure_years * 12  # Convert tenure to months
    if r == 0:  # If interest rate is zero, monthly payment is just principal divided by tenure in months
        return principal / n
    monthly_payment = principal * r * (1 + r)**n / ((1 + r)**n - 1)
    return monthly_payment

# Function to compare loans, using the HDB fixed rate (2.6%) and bank loan rate provided by the user
def compare_loans(hdb_loan_amount, hdb_interest_rate, hdb_loan_tenure,
                  bank_loan_amount, bank_interest_rate, bank_loan_tenure):
    
    def calculate_monthly_payment(principal, interest_rate, tenure_years):
        r = interest_rate / 12 / 100  # Monthly interest rate (as a decimal)
        n = tenure_years * 12  # Convert tenure to months
        if r == 0:  # If interest rate is zero, monthly payment is just principal divided by tenure in months
            return principal / n
        monthly_payment = principal * r * (1 + r)**n / ((1 + r)**n - 1)
        return monthly_payment

    # Calculate monthly repayments for both loans
    hdb_monthly_payment = calculate_monthly_payment(hdb_loan_amount, hdb_interest_rate, hdb_loan_tenure)
    bank_monthly_payment = calculate_monthly_payment(bank_loan_amount, bank_interest_rate, bank_loan_tenure)
    
    # Calculate total repayments over the loan tenure
    hdb_total_repayment = hdb_monthly_payment * hdb_loan_tenure * 12
    bank_total_repayment = bank_monthly_payment * bank_loan_tenure * 12
    
    # Calculate total interest paid over the loan period
    hdb_total_interest = hdb_total_repayment - hdb_loan_amount
    bank_total_interest = bank_total_repayment - bank_loan_amount
    
    # Return the comparison results
    return {
        "HDB Loan": {
            "monthly_payment": round(hdb_monthly_payment, 2),
            "total_repayment": round(hdb_total_repayment, 2),
            "total_interest": round(hdb_total_interest, 2)
        },
        "Bank Loan": {
            "monthly_payment": round(bank_monthly_payment, 2),
            "total_repayment": round(bank_total_repayment, 2),
            "total_interest": round(bank_total_interest, 2)
        }
    }


def ask_for_loan_details():
    loan_details_prompt = """
    To compare loans, please provide the following details:

    1. What is the loan amount for the HDB loan (in SGD)?
    2. What is the loan tenure for the HDB loan (in years)?

    3. What is the loan amount for the bank loan (in SGD)?
    4. What is the interest rate for the bank loan (as a percentage)?
    5. What is the loan tenure for the bank loan (in years)?

    Note: For HDB loans, the current interest rate is fixed at 2.6% (tied to the Ordinary Account interest rate).
    """
    return loan_details_prompt


# --- USER PROFILE ---
user_profile = {
    "age": None,
    "income": None,
    "flat_type": None,
    "relationship_status": None,
    "partner_age": None,
    "partner_income": None,
    "partner_citizenship": None,
    "loan_amount_hdb": None,
    "interest_rate_hdb": None,
    "loan_tenure_hdb": None,
    "loan_amount_bank": None,
    "interest_rate_bank": None,
    "loan_tenure_bank": None
}

user_memory_store = {}
chat_history = []
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
PROFILE_PATH = "user_profile.json"


# --- LangGraph State Definition ---
class State(TypedDict):
    question: str
    hypothetical_doc: Optional[str]
    context: List[Tuple[Document, float]]
    answer: Optional[str]
    messages: List[str]


# --- User Profile Utilities ---
def extract_age(query: str):
    q = query.lower()
    if "partner" in q:
        return None
    match = re.search(r'\b(?:i am|i’m|im)?\s*(\d{2})\s*(?:years old|y/o|yo|yrs)?\b', q)
    return int(match.group(1)) if match else None


def extract_income(query: str):
    q = query.lower()

    # If it's referring to partner, skip extracting user income
    if "partner" in q:
        return None

    match = re.search(r'\$?\s?(\d{3,5})', query)
    return int(match.group(1)) if match else None


def extract_relationship(query: str):
    q = query.lower()
    if any(phrase in q for phrase in [
        "my girlfriend", "my boyfriend", "fiance", "fiancée", "fiancé",
        "partner", "applying together", "we are applying", "with my partner", "applying with"
    ]):
        return "fiance"

    if any(w in q for w in ["married", "spouse", "wife", "husband"]): return "married"
    if "divorced" in q: return "divorced"
    if "widowed" in q or "orphan" in q: return "widowed"
    if "single" in q: return "single"
    return None


def extract_flat_type(query: str):
    q = query.lower()
    if any(word in q for word in ["both", "not sure", "unsure", "either", "any"]):
        return "both"
    if "bto" in q: return "bto"
    if "resale" in q: return "resale"
    return None
    

def extract_partner_info(query: str):
    q = query.lower()
    partner = {}

    # Match income
    income_match = re.search(r'(?:my\s+)?partner(?:\'s)?\s+(?:income|earnings|salary)?\s*(?:is|earns|earning|makes)?\s*\$?(\d{3,5})', q)
    
    # Match age
    age_match = re.search(r'(?:my\s+)?partner(?:\'s)?\s+age\s*(?:is)?\s*(\d{2})', q)

    # Match citizenship
    citizenship_match = re.search(r'(?:my\s+)?partner.*?(singapore citizen|citizen|pr|permanent resident|foreigner|non[-\s]?resident)', q)

    if age_match:
        partner["age"] = int(age_match.group(1))
    if income_match:
        partner["income"] = int(income_match.group(1))
    if citizenship_match:
        c = citizenship_match.group(1).strip().lower()
        if "pr" in c or "permanent" in c:
            partner["citizenship"] = "PR"
        elif "citizen" in c:
            partner["citizenship"] = "Singaporean"
        elif "foreigner" in c or "non" in c:
            partner["citizenship"] = "foreigner"

    return partner


# Function to extract loan amount from user input
def extract_loan_amount(query: str):
    q = query.lower()
    match = re.search(r'\$?\s?(\d{3,5})\s*(?:sgd|dollars|amount)?', q)
    return int(match.group(1)) if match else None

# Function to extract interest rate from user input
def extract_interest_rate(query: str):
    q = query.lower()
    match = re.search(r'(\d+(\.\d{1,2})?)\s*%?', q)
    return float(match.group(1)) if match else None

# Function to extract loan tenure from user input
def extract_loan_tenure(query: str):
    q = query.lower()
    match = re.search(r'(\d{1,2})\s*(?:years?|yrs?)', q)
    return int(match.group(1)) if match else None


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
    user_profile["age"] = extract_age(query) or user_profile["age"]
    user_profile["income"] = extract_income(query) or user_profile["income"]
    user_profile["relationship_status"] = extract_relationship(query) or user_profile["relationship_status"]
    user_profile["flat_type"] = extract_flat_type(query) or user_profile["flat_type"]

    # Update partner fields
    partner = extract_partner_info(query)
    user_profile["partner_age"] = partner.get("age") or user_profile.get("partner_age")
    user_profile["partner_income"] = partner.get("income") or user_profile.get("partner_income")
    user_profile["partner_citizenship"] = partner.get("citizenship") or user_profile.get("partner_citizenship")

    # Update loan details
    user_profile["loan_amount_hdb"] = extract_loan_amount(query) or user_profile.get("loan_amount_hdb")
    user_profile["interest_rate_hdb"] = extract_interest_rate(query) or user_profile.get("interest_rate_hdb")
    user_profile["loan_tenure_hdb"] = extract_loan_tenure(query) or user_profile.get("loan_tenure_hdb")
    
    user_profile["loan_amount_bank"] = extract_loan_amount(query) or user_profile.get("loan_amount_bank")
    user_profile["interest_rate_bank"] = extract_interest_rate(query) or user_profile.get("interest_rate_bank")
    user_profile["loan_tenure_bank"] = extract_loan_tenure(query) or user_profile.get("loan_tenure_bank")


def was_prompt_already_asked(field_key: str):
    question_keywords = {
        "age": ["your age", "how old are you"],
        "income": ["your income", "how much do you earn"],
        "relationship_status": ["your relationship status", "are you single"],
        "flat_type": ["bto or resale", "what flat type"],
        "partner_age": ["your partner's age"],
        "partner_income": ["your partner's income"],
        "partner_citizenship": ["your partner a citizen", "partner's citizenship"],
        
        # Add loan-related fields here
        "loan_amount_hdb": ["what is the loan amount for the hdb loan", "how much is the hdb loan"],
        "loan_tenure_hdb": ["what is the loan tenure for the hdb loan", "how long is the hdb loan tenure"],
        "interest_rate_hdb": ["what is the interest rate for the hdb loan", "hdb loan interest rate"],
        "loan_amount_bank": ["what is the loan amount for the bank loan", "how much is the bank loan"],
        "loan_tenure_bank": ["what is the loan tenure for the bank loan", "how long is the bank loan tenure"],
        "interest_rate_bank": ["what is the interest rate for the bank loan", "bank loan interest rate"]
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

    # Add loan-related fields to the prompt
    if not user_profile.get("loan_amount_hdb") and not was_prompt_already_asked("loan_amount_hdb"):
        prompts.append("What is the loan amount for the HDB loan (in SGD)?")
    if not user_profile.get("loan_tenure_hdb") and not was_prompt_already_asked("loan_tenure_hdb"):
        prompts.append("What is the loan tenure for the HDB loan (in years)?")
    if not user_profile.get("loan_amount_bank") and not was_prompt_already_asked("loan_amount_bank"):
        prompts.append("What is the loan amount for the bank loan (in SGD)?")
    if not user_profile.get("interest_rate_bank") and not was_prompt_already_asked("interest_rate_bank"):
        prompts.append("What is the interest rate for the bank loan (as a percentage)?")
    if not user_profile.get("loan_tenure_bank") and not was_prompt_already_asked("loan_tenure_bank"):
        prompts.append("What is the loan tenure for the bank loan (in years)?")

    # Add partner info only if relationship and flat_type support it
    flat = user_profile.get("flat_type", "").lower()
    if user_profile.get("relationship_status") == "fiance" and flat in ["bto", "resale", "both"]:
        partner_fields = {
            "partner_age": " What is your partner's age?",
            "partner_income": " What is your partner's monthly income?",
            "partner_citizenship": " Is your partner a Singapore Citizen, PR, or foreigner?"
        }
        for key, question in partner_fields.items():
            if not user_profile.get(key):
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
    hyde_embedding = sentence_model.encode(state["hypothetical_doc"])
    results = []
    seen_hashes = set()
    for node_id, node_data in knowledge_graph.nodes(data=True):
        if node_data.get("type") != "document": continue
        doc_embed = node_data.get("embedding")
        if doc_embed is None: continue
        score = cosine_similarity(hyde_embedding.reshape(1, -1), doc_embed.reshape(1, -1))[0][0]
        doc = Document(page_content=node_data["content"], metadata=node_data.get("metadata", {}))
        hash_ = hashlib.md5(doc.page_content.encode()).hexdigest()
        if hash_ not in seen_hashes:
            results.append((doc, score))
            seen_hashes.add(hash_)
    results = [(doc, score) for doc, score in results if score > 0.65]
    results.sort(key=lambda x: x[1], reverse=True)
    state["context"] = results[:3]
    return state


def generate_node(state: State) -> State:
    context_text = "\n\n".join([doc.page_content for doc, _ in state["context"]])
    history = chat_memory.load_memory_variables({}).get("chat_history", "")

    profile = user_profile
    flat = profile.get("flat_type")
    flat_context = "The user is interested in " + (flat if flat else "unspecified") + " flats."

    relationship_note = ""
    if profile.get("relationship_status") == "fiance":
        relationship_note = "Note: The user is currently unmarried but is applying with their partner, which qualifies them under the Fiancé/Fiancée Scheme (min age 21)."

    profile_summary = f"""
    User Profile:
    - Age: {profile.get('age')}
    - Income: {profile.get('income')}
    - Relationship Status: {profile.get('relationship_status')}
    - Flat Type: {flat or 'unspecified'}

    {flat_context}
    {relationship_note}
    """

    # Check if we need to perform loan comparison and generate an appropriate response
    if "compare hdb loan and bank loan" in state["question"].lower():
        # Perform the loan comparison
        loan_comparison = compare_loans(
            user_profile["loan_amount_hdb"], 
            user_profile["interest_rate_hdb"], 
            user_profile["loan_tenure_hdb"],
            user_profile["loan_amount_bank"], 
            user_profile["interest_rate_bank"], 
            user_profile["loan_tenure_bank"]
        )

        comparison_result = (
            f"Here is the loan comparison result:\n\n"
            f"**HDB Loan**\n"
            f"Monthly Payment: ${loan_comparison['HDB Loan']['monthly_payment']}\n"
            f"Total Repayment: ${loan_comparison['HDB Loan']['total_repayment']}\n"
            f"Total Interest: ${loan_comparison['HDB Loan']['total_interest']}\n\n"
            f"**Bank Loan**\n"
            f"Monthly Payment: ${loan_comparison['Bank Loan']['monthly_payment']}\n"
            f"Total Repayment: ${loan_comparison['Bank Loan']['total_repayment']}\n"
            f"Total Interest: ${loan_comparison['Bank Loan']['total_interest']}\n"
        )

        # Default processing for other queries using LLM
        prompt = [
            {"role": "system", "content": f"Summarize the following loan comparison in a conversational tone:\n\n{comparison_result}"},
            {"role": "user", "content": "Please explain which loan option is better for me."}
        ]

        # Invoke the LLM to generate the explanation
        response = llm.invoke(prompt)
        state["answer"] = response.content  # Use the LLM's response for the final answer
        return state

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

📋 Formatting instructions:
- Structure your answer in clear paragraphs or bullet points.
- Separate sections by eligibility paths (e.g., Singles Scheme, Fiancé/Fiancée Scheme).
- Only list paths that may apply based on current info.
- Keep answers concise if data is incomplete.

📚 Retrieved Context:
{context_text}

💬 Chat History:
{history if history else "No prior chat history."}
"""
},
        {"role": "user", "content": state["question"]}
    ]
    response = llm.invoke(prompt)
    chat_memory.save_context({"input": state["question"]}, {"output": response.content})
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

# ---- Build LangGraph ----
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


def interactive_chatbot(user_input, serial_code=None):
    lower_input = user_input.strip().lower()
    if any(phrase in lower_input for phrase in ["show profile", "profile info", "my profile", "what do you know about me", "current profile", "show my info"]):
        return format_user_profile()
        
    # Handle loan comparison request
    if "compare hdb loan and bank loan" in lower_input:
        # Ask for loan details if they're missing
        return ask_for_loan_details()
    
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
            "partner_citizenship": None,
            "loan_amount_hdb": None,
            "interest_rate_hdb": None,
            "loan_tenure_hdb": None,
            "loan_amount_bank": None,
            "interest_rate_bank": None,
            "loan_tenure_bank": None
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
