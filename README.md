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

# Avaliando a Precisão do Modelo

Regressão Logística: Alcançou o desempenho superior com uma acurácia de 95.61%.

Árvore de Decisão: Apresentou excelente robustez, atingindo uma acurácia de 93.86%.

**Superioridade Linear Confiável:** Ambas as abordagens demonstraram um desempenho extremamente sólido na classificação de casos malignos e benignos. No entanto, a Regressão Logística obteve uma acurácia ligeiramente superior e maior estabilidade neste conjunto de teste específico. Isso sugere que, após a padronização e o escalonamento dos recursos (StandardScaler), as fronteiras de decisão do problema ganharam uma separabilidade linear muito bem definida.

<img width="1016" height="347" alt="Image" src="https://github.com/user-attachments/assets/0b5029ba-1654-4350-9718-63970f302c1b" />

# 🔍 Comparação visual das previsões

O que este gráfico nos mostra?
Sobreposição Perfeita (Roxo/Centro do Alvo): Sempre que o X vermelho fica exatamente centralizado dentro do círculo azul, significa que o modelo realizou uma predição correta (Verdadeiro Positivo ou Verdadeiro Negativo).

Pontos Isolados: Casos onde o círculo azul e o X vermelho aparecem separados na mesma amostra vertical indicam um erro de classificação (Falsos Positivos ou Falsos Negativos), permitindo rastrear visualmente quais amostras clínicas geraram incerteza nos algoritmos.

<img width="1390" height="480" alt="Image" src="https://github.com/user-attachments/assets/ad6c5665-e942-478f-9d51-3570e8e63bc6" />

# Matriz de Confusão

Para avaliar melhor o desempenho dos nossos classificadores, calculamos e exibimos a matriz de confusão para os modelos de Árvore de Decisão e Regressão Logística. Isso nos forneceu informações sobre o número de previsões verdadeiras positivas, verdadeiras negativas, falsas positivas e falsas negativas.

<img width="528" height="470" alt="Image" src="https://github.com/user-attachments/assets/5f480ec3-3756-44fd-9942-14ce154e55fe" />

<img width="528" height="470" alt="Image" src="https://github.com/user-attachments/assets/7270be9e-2bd1-43e8-8af5-11d2e1dc41a3" />

### **Matriz de Confusão para a Árvore de Decisão:**

```
[[68  3]
 [ 4 39]]
```

-   **68 Verdadeiros Negativos (VN):** O modelo previu corretamente que 68 casos eram **Benignos** e eles realmente eram Benignos. (Canto superior esquerdo)
-   **3 Falsos Positivos (FP):** O modelo previu que 3 casos eram **Malignos**, mas eles eram realmente Benignos. (Canto superior direito - Erro Tipo I)
-   **4 Falsos Negativos (FN):** O modelo previu que 4 casos eram **Benignos**, mas eles eram realmente Malignos. (Canto inferior esquerdo - Erro Tipo II)
-   **39 Verdadeiros Positivos (VP):** O modelo previu corretamente que 39 casos eram **Malignos** e eles realmente eram Malignos. (Canto inferior direito)

--- 

#### **Matriz de Confusão para a Regressão Logística:**

```
[[70  1]
 [ 4 39]]
```

-   **70 Verdadeiros Negativos (VN):** O modelo previu corretamente que 70 casos eram **Benignos** e eles realmente eram Benignos. (Canto superior esquerdo)
-   **1 Falso Positivo (FP):** O modelo previu que 1 caso era **Maligno**, mas ele era realmente Benigno. (Canto superior direito - Erro Tipo I)
-   **4 Falsos Negativos (FN):** O modelo previu que 4 casos eram **Benignos**, mas eles eram realmente Malignos. (Canto inferior esquerdo - Erro Tipo II)
-   **39 Verdadeiros Positivos (VP):** O modelo previu corretamente que 39 casos eram **Malignos** e eles realmente eram Malignos. (Canto inferior direito)

--- 

**Comparação:**

*   A **Regressão Logística** teve um desempenho ligeiramente melhor na identificação de casos Benignos, com apenas 1 Falso Positivo (contra 3 da Árvore de Decisão).
*   Ambos os modelos tiveram o mesmo número de Falsos Negativos (4), o que significa que ambos erraram em 4 casos de câncer (Maligno) predizendo que eram Benignos.
*   Ambos os modelos acertaram o mesmo número de Verdadeiros Positivos (39), identificando corretamente os casos Malignos.

# Calculando a acurácia, a precisão e a revocação

Na imagem abaixo Calculamos a acurácia, a precisão e a revocação para os modelos de Árvore de Decisão e Regressão Logística. Essas métricas fornecem uma avaliação mais abrangente do desempenho do modelo, especialmente em conjuntos de dados desbalanceados.

<img width="1237" height="656" alt="Image" src="https://github.com/user-attachments/assets/d4afa752-246d-4b38-a459-b9bfc0d9c562" />

# Comparação das métricas de desempenho do modelo

Com base nas métricas de desempenho que calculamos e visualizamos (Acurácia, Precisão, e Recall), podemos observar o seguinte:

Acurácia: A Regressão Logística (0.9561) superou ligeiramente a Árvore de Decisão (0.9386), indicando que a Regressão Logística fez uma proporção maior de previsões corretas no geral.

Precisão: A Regressão Logística (0.9750) mostrou uma precisão significativamente maior do que a Árvore de Decisão (0.9286). Isso significa que, quando a Regressão Logística previu um caso como maligno, a probabilidade de estar correto foi maior.

Recall (Sensibilidade): Ambos os modelos tiveram o mesmo recall (0.9070). Isso indica que ambos foram igualmente bons em identificar a proporção de todos os casos malignos que foram corretamente identificados. Ou seja, ambos os modelos deixaram de identificar 4 casos malignos, como visto nas matrizes de confusão.

<img width="990" height="590" alt="Image" src="https://github.com/user-attachments/assets/aa77404d-792c-4511-94bf-70344307802f" />

# Curvas ROC

Para avaliar melhor o desempenho dos nossos modelos, especialmente considerando a relação entre a taxa de verdadeiros positivos (sensibilidade) e a taxa de falsos positivos (1 - especificidade), vamos plotar a curva ROC (Receiver Operating Characteristic) para os modelos de Árvore de Decisão e Regressão Logística. Também calcularemos a Área Sob a Curva (AUC), que resume a capacidade do modelo de distinguir entre as classes.

<img width="846" height="624" alt="Image" src="https://github.com/user-attachments/assets/7b8b601f-a1d1-4259-9975-e3fbd67c82ca" />

# Relatório de Classificação

Para obter uma análise mais detalhada do desempenho do modelo, além da acurácia, precisão e recall, usamos o classification_report. Este relatório fornece essas métricas por classe, juntamente com a pontuação F1 e o suporte (número de ocorrências reais da classe no conjunto de dados especificado).

<img width="1216" height="653" alt="Image" src="https://github.com/user-attachments/assets/e3b1846b-e271-4e9f-be49-404809f4c692" />

# Conclusão do Modelo

Com base na análise realizada, segue um resumo do desempenho do modelo no conjunto de teste de 114 amostras:

* **Classificador de Árvore de Decisão:**

* Acurácia: 0,9386

* Precisão: 0,9286

* Revocação: 0,9070

* **Regressão Logística:**

* Acurácia: 0,9561

* Precisão: 0,9750

* Revocação: 0,9070

Ambos os modelos demonstram um desempenho sólido na classificação de casos malignos e benignos. A Regressão Logística apresenta acurácia e precisão ligeiramente superiores neste conjunto de teste específico.

