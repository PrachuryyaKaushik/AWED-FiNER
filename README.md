# 🌍 AWED-FiNER: Agents, Web applications, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers

AWED-FiNER agentic tool is a unified **Agentic AI Tool** designed to bridge the gap between global high-resource languages and vulnerable linguistic communities. It provides high-precision, Fine-grained Named Entity Recognition (FgNER) by routing queries to 36 specialized expert models.

## 🌟 Why AWED-FiNER?
While standard Large Language Models (LLMs) are powerful, they often struggle with:
1.  **Vulnerable Languages:** Languages like Bodo, Manipuri, Bishnupriya, and Mizo are under-represented and considered as vulnerable by UNESCO.
2.  **Fine-grained Accuracy:** AWED-FiNER distinguishes between specific sub-categories (e.g., 'Politician', 'City', 'River') rather than just generic 'Person' or 'Location' tags.
3.  **Digital Preservation:** Providing state-of-the-art AI support for 26 Indian languages and 10 global languages spoken by over 80% of the world's population.

## 🏗️ Technical Architecture
AWED-FiNER acts as a **Routing Agent Tool** callable by LLM agents (e.g., smolagents, LangChain, or custom agents). It functions as a specialized capability hosted on Hugging Face Spaces. Instead of using one general-purpose model, it routes requests to a specialized, fine-tuned "Expert Model" (mBERT, XLM-R, or IndicBERTv2) optimized for each supported language. The tool is exposed via a production-ready Gradio API endpoint (`/predict`) and accessed programmatically using `gradio_client`.

**Core Datasets:**
* **SampurNER:** Comprehensive NER for 22 Indian languages. [Dataset](https://huggingface.co/datasets/prachuryyaIITG/SampurNER) [Paper](https://github.com/PrachuryyaKaushik/SampurNER/blob/main/SampurNER_AAAI_extended.pdf)
* **CLASSER:** Multilingual fine-grained datasets for five low-resource Indian languages. [Dataset](https://huggingface.co/datasets/prachuryyaIITG/CLASSER) [Paper](https://aclanthology.org/2025.ijcnlp-long.94/)
* **MultiCoNER2:** SemEval-23 task contributing extensively to global FgNER research. [Dataset](https://huggingface.co/datasets/MultiCoNER/multiconer_v2) [Paper](https://aclanthology.org/2023.findings-emnlp.134/)
 



## 🛠️ Quick Start (Agentic Tool)

### 1. Installation
```bash
pip install smolagents gradio_client
```
### 2. Usage with smolagents
```python
from smolagents import CodeAgent, HfApiModel
from tool import AWEDFiNERTool

# Initialize the expert tool (connects to Hugging Face Space API)
ner_tool = AWEDFiNERTool(
    space_id="prachuryyaIITG/AWED-FiNER"
)

# Initialize the agent (using a model of your choice)
agent = CodeAgent(
    tools=[ner_tool],
    model=HfApiModel()
)

# The agent will automatically use AWED-FiNER via the /predict API endpoint
# Case: Processing a vulnerable language (Bodo)
agent.run("Recognize the named entities in this Bodo sentence: 'बिथाङा दिल्लियाव थाङो।'")
```
### 3. Direct Tool Usage (without agent)
```python
from tool import AWEDFiNERTool

tool = AWEDFiNERTool(
    space_id="prachuryyaIITG/AWED-FiNER"
)

result = tool.forward(
    text="Jude Bellingham joined Real Madrid in 2023.",
    language="English"
)

print(result)
```

## 🔗 Project Links

* Interactive Demo: [AWED-FiNER](https://huggingface.co/spaces/prachuryyaIITG/AWED-FiNER)

* Model Hub: [Language specific Models](https://huggingface.co/prachuryyaIITG/models) for Fine-grained NER in HuggingFace

* Author Profile: Prachuryya Kaushik: [LinkedIn](https://www.linkedin.com/in/pkabundant/)  &emsp;  [Google Scholar](https://scholar.google.com/citations?user=dyGLivYAAAAJ&hl=en)  &emsp;  [Hugging Face](https://huggingface.co/prachuryyaIITG)



## Citations

If you use this tool, please cite the following papers:

```bibtex
@misc{kaushik2026awedfiner,
  title        = {AWED-FiNER: Agents, Web Applications, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers},
  author       = {Kaushik, Prachuryya and Anand, Ashish},
  year         = {2026},
  note         = {arXiv preprint, submitted},
  archivePrefix= {arXiv},
  eprint       = {submit/7163987}
}

@inproceedings{kaushik2026sampurner,
  title={SampurNER: Fine-grained Named Entity Recognition Dataset for 22 Indian Languages},
  author={Kaushik, Prachuryya and Anand, Ashish},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  year={2026}
}

@inproceedings{kaushik-anand-2025-classer,
    title = "{CLASSER}: Cross-lingual Annotation Projection enhancement through Script Similarity for Fine-grained Named Entity Recognition",
    author = "Kaushik, Prachuryya  and
      Anand, Ashish",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.94/",
    pages = "1745--1760",
    ISBN = "979-8-89176-298-5",
}

@inproceedings{fetahu2023multiconer,
  title={MultiCoNER v2: a Large Multilingual dataset for Fine-grained and Noisy Named Entity Recognition},
  author={Fetahu, Besnik and Chen, Zhiyu and Kar, Sudipta and Rokhlenko, Oleg and Malmasi, Shervin},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={2027--2051},
  year={2023}
}
```
