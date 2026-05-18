# 🎗️ Predição e Classificação Adaptativa de Câncer de Mama com Machine Learning
🧬 Pipeline de Machine Learning para classificação e diagnóstico preditivo de câncer de mama (Maligno/Benigno) utilizando Regressão Logística e Árvores de Decisão com foco em interpretabilidade de atributos biomédicos.

# 🎯 Objetivo do Projeto

Este projeto desenvolve um pipeline de dados e inteligência artificial focado em saúde preditiva, com o objetivo de auxiliar na identificação precoce e classificação de tumores de mama entre Malignos e Benignos.

Utilizando dados clínicos e características celulares extraídos de imagens digitalizadas de biópsias aspirativas por agulha fina (PAAF), o projeto aborda desde a análise exploratória (EDA), tratamento e escalonamento de atributos, até o treinamento comparativo de modelos preditivos. Foram aplicados algoritmos de Regressão Logística e Árvores de Decisão, alcançando métricas sólidas de acurácia (até 95.6%) e fornecendo interpretabilidade clínica através da extração das variáveis celulares com maior peso estatístico no diagnóstico.

# Pré-processamento de dados

**Foi Removida colunas irrelevantes**: A coluna 'id' é apenas um identificador e 'Sem nome: 32' contém apenas valores nulos.

**Codifiquei a variável alvo:** A coluna 'diagnosis', que indica 'Maligno' ou 'Benigno', foi convertida para formato numérico (por exemplo, M=1, B=0).

<img width="1368" height="554" alt="Image" src="https://github.com/user-attachments/assets/bfbc6d6c-5e16-4f8e-bcfd-58d4747c7bf7" />

# Divisão em Conjuntos de Treinamento e Teste

Por fim, dividirei os dados pré-processados em conjuntos de treinamento e teste para avaliar o desempenho do modelo em dados não vistos. Utilizarei a função train_test_split com um test_size de 0.2 (20% para o conjunto de teste) e random_state=42 para garantir a reprodutibilidade dos resultados. Após a divisão, os dados de treino e teste serão armazenados nas variáveis X_train_actual, X_test_actual, y_train_actual e y_test_actual, permitindo validar as dimensões (shape) de cada matriz antes de iniciar a modelagem.

<img width="1321" height="315" alt="Image" src="https://github.com/user-attachments/assets/291965c2-3d87-472d-9e65-868c18947863" />
