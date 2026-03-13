# Lab P1-03 – Implementação do Transformer Decoder

Este projeto implementa componentes do **Decoder de um Transformer**, conforme descrito no artigo científico *Attention Is All You Need*.

O laboratório inclui a implementação de três elementos principais do Decoder: a máscara causal (Look-Ahead Mask), o mecanismo de Cross Attention entre Encoder e Decoder e uma simulação do loop de geração auto-regressiva de tokens.

A biblioteca utilizada para as operações matriciais foi o NumPy.

---

## Requisitos

- Python 3.x  
- NumPy  

Instalação do NumPy:

pip install numpy

---

## Execução

Para executar o exemplo de teste, utilize:

python main.py

O script executa três testes principais:

- aplicação da máscara causal  
- cálculo do Cross Attention entre Encoder e Decoder  
- simulação de geração auto-regressiva de tokens  

Os resultados são exibidos diretamente no terminal.

---

## Máscara Causal (Look-Ahead Mask)

A máscara causal impede que uma posição da sequência tenha acesso a tokens futuros durante o cálculo da atenção.

A função implementada gera uma matriz quadrada onde a diagonal principal e a parte inferior possuem valor **0**, enquanto a parte superior possui valor **-infinito**.

Essa máscara é adicionada aos scores de atenção antes da aplicação do **Softmax**, garantindo que a probabilidade de tokens futuros seja anulada.

---

## Cross Attention

No Decoder, o mecanismo de Cross Attention permite que o modelo utilize a representação gerada pelo Encoder.

Nesse processo:

- o estado do Decoder é projetado para gerar as **Queries (Q)**
- a saída do Encoder gera as **Keys (K)** e **Values (V)**

O cálculo segue a fórmula do **Scaled Dot-Product Attention**:

Attention(Q, K, V) = softmax((QK^T) / sqrt(dk)) V

Diferente do Self-Attention do Decoder, nesta etapa não é aplicada máscara causal, pois o modelo pode acessar toda a saída do Encoder.

---

## Loop de Inferência Auto-Regressivo

Modelos baseados em Transformer geram texto de forma iterativa.

O processo inicia com o token `<START>`. A cada iteração, o modelo calcula uma distribuição de probabilidade para o próximo token, seleciona o mais provável utilizando **argmax** e o adiciona à sequência.

O loop continua até que o token `<EOS>` seja gerado ou até atingir um limite máximo de tokens.
