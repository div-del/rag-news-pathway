# DataQuest 2026 — Dynamic RAG Playground  
**Megalith 2026 | IIT Kharagpur (Online Hackathon)**

---

## 🚀 About Pathway

Pathway is a real-time data processing framework designed for building **Live AI systems**.  
It enables AI pipelines that **continuously adapt to changing data** without restarts or re-indexing.

Pathway introduces:
- A post-transformer architecture (BDH)
- The world’s fastest incremental data processing engine
- Native support for real-time Retrieval-Augmented Generation (RAG)

**Key Repositories (Mandatory):**
- https://github.com/pathwaycom/pathway  
- https://github.com/pathwaycom/llm-app  
- https://github.com/pathwaycom/bdh  

---

## 🧠 Hackathon Theme: *Live AI*

### The Problem with Static AI
Traditional LLM-based systems rely on **stale knowledge snapshots**.  
Even RAG systems often fail to reflect **real-time changes** in data.

### The Shift to Live AI
Live AI systems:
- Ingest data continuously
- Update knowledge instantly
- Reason over the *current state of reality*

This hackathon challenges you to build such a system.

---

## 🎯 Core Challenge

### Formal Problem Statement
Build a **Dynamic Retrieval-Augmented Generation (RAG) application** using the **Pathway framework** that:

- Connects to a **live, continuously updating data source**
- Updates its knowledge **incrementally**
- Reflects data changes in responses **almost instantly**
- Requires **no manual restart or batch re-indexing**

---

## 🔑 Key Requirement: Demonstrable Dynamism

This is the **most important evaluation criterion**.

Your system must clearly demonstrate that:
- When data changes
- The system’s answers change immediately

Judges will expect:
- Visible real-time updates
- Low latency between ingestion and response
- End-to-end streaming behavior

---

## 💡 Example Application Ideas (Inspiration Only)

- Live News Analyst  
- Real-Time Stock / Market Analyst  
- Dynamic Documentation Assistant  
- Social Media Trend Tracker  
- Live E-commerce Inventory Assistant  

⚠️ These are examples, not restrictions.

---

## 📡 Data Source Requirements

### Mandatory
Your final project **must use a dynamic data source**.

Static datasets may be used during development, but **live behavior must be demonstrated**.

---


## 🔁 Alternative Dynamic Data Sources (Encouraged)

- Cloud Storage (Google Drive, S3, SharePoint)
- Databases with CDC (Postgres, etc.)
- Kafka / MQTT streams
- Custom Python connectors
- Artificial streaming via Pathway demo utilities

---

## 🧱 Core Pathway Concepts You Must Use

- Streaming connectors
- Tables & transformations
- Incremental joins and filters
- Stateful windowed computations
- Real-time feature engineering

Documentation:
- https://pathway.com/developers/user-guide/introduction/concepts/

---

## 🤖 LLM Integration (RAG)

Use Pathway’s **LLM xPack** for:
- Live RAG
- Summarization
- Reasoning over changing data

Supported integrations:
- OpenAI / Gemini / OpenRouter
- LangChain / LlamaIndex
- Agentic RAG workflows

Resources:
- https://pathway.com/developers/user-guide/llm-xpack/overview

---

## 🧪 Judging Criteria

### 1️⃣ Real-Time Capability & Dynamism — **35%**
- Instant reaction to data changes
- No restarts or re-indexing
- Clear live demo

### 2️⃣ Technical Implementation & Elegance — **30%**
- Idiomatic Pathway usage
- Clean, modular code
- Clear architecture

### 3️⃣ Innovation & UX — **20%**
- Non-generic idea
- Thoughtful prompt engineering
- Functional UI / API / CLI

### 4️⃣ Impact & Feasibility — **15%**
- Real-world relevance
- Scalability considerations
- Clear value proposition

---

## 📦 Submission Requirements

### Required Deliverables
- Public GitHub repository
- Comprehensive `README.md`
- 3-minute demo video
- Clear proof of real-time behavior

### README Must Include
- Project overview
- Architecture diagram
- Setup instructions
- How live updates work
- Prompt & RAG explanation

---

## 🎥 Demo Expectations

Your demo must show:
1. A question asked
2. New data arriving
3. The same question producing a **different answer**

This is **non-negotiable**.

---

## 🧠 Final Notes

You are expected to:
- Exploit Pathway’s streaming engine fully
- Use incremental computation
- Build a true Live AI system

Static RAG systems will **not score well**.

---

**Good luck, and build something that thinks in real time.**
