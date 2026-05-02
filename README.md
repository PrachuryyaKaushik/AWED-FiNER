# 🌍 AWED-FiNER: Agent, Web application, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers

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
* **FewNERD:** Largest manually annotated FgNER dataset in English. [Dataset](https://huggingface.co/datasets/DFKI-SLT/few-nerd) [Paper](https://aclanthology.org/2021.acl-long.248/)
* **FiNERVINER:** Multilingual fine-grained datasets for three vulnerable Indian languages. [Dataset](https://huggingface.co/datasets/prachuryyaIITG/FiNERVINER) [Paper](https://lrec.elra.info/lrec2026-main-607)
* **APTFiNER:** Multilingual fine-grained datasets for six low-resource Indian languages. [Dataset](https://huggingface.co/datasets/prachuryyaIITG/APTFiNER) [Paper](https://lrec.elra.info/lrec2026-main-608)
* **FiNE-MiBBiC:** Multilingual fine-grained datasets for four low-resource Indian languages. [Dataset](https://github.com/PrachuryyaKaushik/FiNE-MiBBiC) [Paper(under review in ACM TALLIP)](https://dl.acm.org/journal/tallip)
* **TAFSIL:** Multilingual fine-grained datasets for six low-resource Indian languages in four different taxonomies. [Dataset](https://huggingface.co/datasets/prachuryyaIITG/TAFSIL) [Paper](https://dl.acm.org/doi/10.1145/3726302.3730341)
 
## 🔗 Important Links

* Interactive Demo: [AWED-FiNER](https://huggingface.co/spaces/prachuryyaIITG/AWED-FiNER)

* Collection of [49 Best-performing Expert Detector Models](https://huggingface.co/collections/prachuryyaIITG/awed-finer) for Fine-grained NER in HuggingFace

* [AWED-FiNER paper](https://arxiv.org/abs/2601.10161)

* Author Profile: Prachuryya Kaushik: [LinkedIn](https://www.linkedin.com/in/pkabundant/)  &emsp;  [Google Scholar](https://scholar.google.com/citations?user=dyGLivYAAAAJ&hl=en)  &emsp;  [Hugging Face](https://huggingface.co/prachuryyaIITG) <br>
Prof. Ashish Anand: [LinkedIn](https://www.linkedin.com/in/anandashish/)  &emsp;  [Google Scholar](https://scholar.google.co.in/citations?user=W7nidBQAAAAJ&hl=en&oi=ao)  &emsp;  [Personal Website](https://www.iitg.ac.in/anand.ashish/)

## 🛠️ Quick Start (Agentic Tool)

### 1. Installation
```bash
pip install smolagents gradio_client
```
### 2. Tool Usage
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
### 3. Example usage with Gemini
```python
import os
import json
from google import genai
from google.genai import types
from gradio_client import Client

# 1. Setup
# Ensure your API key is in the environment or pass it to the Client
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
client_gemini = genai.Client()
client_gradio = Client("prachuryyaIITG/AWED-FiNER")

# 2. Define the Tool Function
def awed_finer_ner(text: str, language: str):
    """
    Extracts fine-grained named entities from text across 36 languages.
    """
    result = client_gradio.predict(
        text=text,
        language=language,
        api_name="/predict"
    )
    return json.dumps(result)

# 3. Use the Tool with Gemini
response = client_gemini.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Identify the entities in this Bodo sentence: 'अमिताभ बच्चनआ सासे मुंदांखा फावखुंगुर।'",
    config=types.GenerateContentConfig(
        tools=[awed_finer_ner]
    )
)

print(f"Gemini's Response:\n{response.text}")
```

## Experimental Results

### Fine-tuned models on various languages in MultiCoNER2 Taxonomy
**Performance (Macro-F1) of the best expert models across 22 languages under the MultiCoNER2 taxonomy.** This includes the MultiCoNER2 benchmark, CLASSER, FiNERVINER, and APTFiNER datasets.

| Language | Base Encoder | Dataset | Precision | Recall | Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Assamese | MuRIL | CLASSER | 74.88 | 75.62 | 75.25 |
| Bengali | XLM-RoBERTa | MultiCoNER2 | 77.74 | 79.36 | 78.54 |
| Bodo | MuRIL | CLASSER | 73.83 | 76.37 | 75.08 |
| Chinese | XLM-RoBERTa | MultiCoNER2 | 64.66 | 69.42 | 66.95 |
| English | XLM-RoBERTa | MultiCoNER2 | 78.29 | 80.94 | 79.59 |
| Farsi | XLM-RoBERTa | MultiCoNER2 | 76.05 | 78.86 | 77.43 |
| French | XLM-RoBERTa | MultiCoNER2 | 81.83 | 83.03 | 82.43 |
| German | XLM-RoBERTa | MultiCoNER2 | 74.51 | 76.29 | 75.38 |
| Hindi | XLM-RoBERTa | MultiCoNER2 | 76.07 | 79.42 | 77.71 |
| Italian | XLM-RoBERTa | MultiCoNER2 | 85.67 | 85.98 | 85.83 |
| Manipuri | IndicBERTv2 | FiNERVINER | 60.49 | 64.42 | 62.39 |
| Marathi | MuRIL | CLASSER | 79.24 | 81.00 | 80.11 |
| Mizo | XLM-RoBERTa | FiNERVINER | 80.33 | 81.83 | 81.07 |
| Nepali | MuRIL | CLASSER | 76.92 | 79.50 | 78.19 |
| Portuguese | XLM-RoBERTa | MultiCoNER2 | 80.07 | 81.98 | 81.01 |
| Sanskrit | MuRIL | CLASSER | 77.62 | 78.99 | 78.30 |
| Spanish | XLM-RoBERTa | MultiCoNER2 | 79.51 | 81.42 | 80.45 |
| Swedish | XLM-RoBERTa | MultiCoNER2 | 85.10 | 84.19 | 84.64 |
| Tamil | XLM-RoBERTa | APTFiNER | 75.51 | 77.28 | 76.38 |
| Telugu | MuRIL | APTFiNER | 75.72 | 77.26 | 76.49 |
| Ukrainian | XLM-RoBERTa | MultiCoNER2 | 79.78 | 81.51 | 80.61 |
| Urdu | XLM-RoBERTa | CLASSER | 73.48 | 74.92 | 73.93 |

---

### Fine-tuned models on various languages in FewNERD Taxonomy
**Performance (Macro-F1) of the best expert models across 27 languages under the FewNERD taxonomy.** This includes the FewNERD benchmark, SampurNER, and FiNE-MiBBiC datasets.

| Language | Base Encoder | Dataset | Precision | Recall | Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Assamese | MuRIL | SampurNER | 65.54 | 67.73 | 66.26 |
| Bengali | MuRIL | SampurNER | 66.54 | 70.08 | 68.26 |
| Bhojpuri | MuRIL | FiNE-MiBBiC | 66.12 | 69.51 | 67.78 |
| Bishnupriya | MuRIL | FiNE-MiBBiC | 61.85 | 64.13 | 62.97 |
| Bodo | IndicBERTv2 | SampurNER | 64.17 | 67.14 | 65.62 |
| Chhattisgarhi | MuRIL | FiNE-MiBBiC | 64.40 | 67.87 | 66.09 |
| Dogri | IndicBERTv2 | SampurNER | 60.34 | 64.79 | 62.49 |
| English | MuRIL | FewNERD | 66.21 | 69.98 | 68.04 |
| Gujarati | IndicBERTv2 | SampurNER | 64.77 | 67.88 | 66.29 |
| Hindi | IndicBERTv2 | SampurNER | 62.08 | 66.01 | 63.99 |
| Kannada | IndicBERTv2 | SampurNER | 63.97 | 67.41 | 65.63 |
| Kashmiri | IndicBERTv2 | SampurNER | 60.83 | 64.79 | 62.74 |
| Konkani | IndicBERTv2 | SampurNER | 61.52 | 65.32 | 63.36 |
| Maithili | IndicBERTv2 | SampurNER | 60.52 | 64.76 | 62.56 |
| Malayalam | IndicBERTv2 | SampurNER | 62.09 | 65.80 | 63.89 |
| Manipuri | IndicBERTv2 | SampurNER | 57.68 | 61.14 | 59.35 |
| Marathi | IndicBERTv2 | SampurNER | 65.09 | 68.47 | 66.73 |
| Mizo | MuRIL | FiNE-MiBBiC | 64.85 | 68.70 | 66.71 |
| Nepali | MuRIL | SampurNER | 67.18 | 70.01 | 68.57 |
| Odia | IndicBERTv2 | SampurNER | 63.29 | 66.71 | 64.96 |
| Punjabi | IndicBERTv2 | SampurNER | 61.53 | 65.44 | 63.42 |
| Sanskrit | IndicBERTv2 | SampurNER | 61.48 | 65.19 | 63.27 |
| Santali | IndicBERTv2 | SampurNER | 51.21 | 52.51 | 52.84 |
| Sindhi | IndicBERTv2 | SampurNER | 60.33 | 62.08 | 61.13 |
| Tamil | IndicBERTv2 | SampurNER | 62.29 | 65.67 | 63.72 |
| Telugu | IndicBERTv2 | SampurNER | 62.04 | 65.82 | 63.89 |
| Urdu | IndicBERTv2 | SampurNER | 62.17 | 65.89 | 63.97 |






## Citations

If you use this tool, please cite the following papers:

```bibtex
@misc{kaushik2026awedfineragentswebapplications,
      title={AWED-FiNER: Agents, Web applications, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers}, 
      author={Prachuryya Kaushik and Ashish Anand},
      year={2026},
      eprint={2601.10161},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.10161}, 
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

@inproceedings{kaushik-etal-2026-finerviner,
  title = {FiNERVINER: Fine-grained Named Entity Recognition for Vulnerable Languages of India's North Eastern Region},
  author = {Kaushik, Prachuryya and Anand, Ashish},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {7655--7667},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  doi = {10.63317/3rs5mcedzvss}
}

@inproceedings{kaushik-etal-2026-aptfiner,
  title = {APTFiNER: Annotation Preserving Translation for Fine-grained Named Entity Recognition},
  author = {Kaushik, Prachuryya and Gupta, Adittya and Maurya, Ajanta and Sharma, Gautam and Saradhi, V. V. and Anand, Ashish},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {7668--7680},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  doi = {10.63317/3w7rv4rg7nty}
}

@inproceedings{fetahu2023multiconer,
  title={MultiCoNER v2: a Large Multilingual dataset for Fine-grained and Noisy Named Entity Recognition},
  author={Fetahu, Besnik and Chen, Zhiyu and Kar, Sudipta and Rokhlenko, Oleg and Malmasi, Shervin},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={2027--2051},
  year={2023}
}
```
