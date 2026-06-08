# CausalAudit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Causal-Guard is a **neuro-symbolic verification layer** that audits LLM-generated explanations against five causal admissibility constraints.

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **C₃ Recall** | 100% (catches all physical impossibilities) |
| **C₃ Precision** | 21.1% (safety-first design) |
| **C₅ F1 Score** | 66.1% |
| **Latency** | <500ms per explanation |

## 🚀 Quick Start
```bash
git clone https://github.com/yahyaatuom/CausalAudit.git
cd causalaudit
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python main.py