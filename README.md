# 🎗️ Predição e Classificação Adaptativa de Câncer de Mama com Machine Learning
🧬 Pipeline de Machine Learning para classificação e diagnóstico preditivo de câncer de mama (Maligno/Benigno) utilizando Regressão Logística e Árvores de Decisão com foco em interpretabilidade de atributos biomédicos.

# Objetivo do Projeto

Este projeto desenvolve um pipeline de dados e inteligência artificial focado em saúde preditiva, com o objetivo de auxiliar na identificação precoce e classificação de tumores de mama entre Malignos e Benignos.

Utilizando dados clínicos e características celulares extraídos de imagens digitalizadas de biópsias aspirativas por agulha fina (PAAF), o projeto aborda desde a análise exploratória (EDA), tratamento e escalonamento de atributos, até o treinamento comparativo de modelos preditivos. Foram aplicados algoritmos de Regressão Logística e Árvores de Decisão, alcançando métricas sólidas de acurácia (até 95.6%) e fornecendo interpretabilidade clínica através da extração das variáveis celulares com maior peso estatístico no diagnóstico.

# Pré-processamento de dados

**Foi Removida colunas irrelevantes**: A coluna 'id' é apenas um identificador e 'Sem nome: 32' contém apenas valores nulos.

**Codifiquei a variável alvo:** A coluna 'diagnosis', que indica 'Maligno' ou 'Benigno', foi convertida para formato numérico (por exemplo, M=1, B=0).

<img width="1368" height="554" alt="Image" src="https://github.com/user-attachments/assets/bfbc6d6c-5e16-4f8e-bcfd-58d4747c7bf7" />
