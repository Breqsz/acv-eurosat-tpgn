# Contexto — Global Solution 2026/1 · Applied Computer Vision (ACV)

> Documento-fonte de contexto do projeto. Tudo aqui é a base de verdade para
> as decisões técnicas e a implementação da entrega de Visão Computacional.

---

## 1. Identificação

| Campo | Valor |
|---|---|
| **Evento** | Global Solution 2026 · 1º Semestre |
| **Desafio** | Indústria Espacial: O Código que Move o Universo |
| **Curso** | Engenharia de Software · 4º Ano · Presencial · FIAP |
| **Disciplina (esta entrega)** | Applied Computer Vision (ACV) |
| **Grupo** | **TPGN** — TechPulse GlobalNetwork |
| **Início** | 25/05/2026 |
| **Entrega** | 09/06/2026 (até 23h55) |
| **Valor da disciplina** | 14 pontos · entrega técnica vale 100 pts internos |

### Integrantes do grupo TPGN
- **GUILHERME ROCHA BIANCHINI** — RM97974
- **NIKOLAS RODRIGUES MOURA DOS SANTOS** — RM551566
- **PEDRO HENRIQUE PEDROSA TAVARES** — RM97877
- **RODRIGO BRASILEIRO** — RM98952
- **THIAGO JARDIM DE OLIVEIRA** — RM551624

### Convenção de autoria (IMPORTANTE)
> Em todos os arquivos de código, notebooks e cabeçalhos, o campo **`Author`**
> deve ser **GUILHERME ROCHA BIANCHINI**. Toda autoria de código é atribuída a
> ele por padrão.

---

## 2. Contexto do Desafio (tema-mãe da Global Solution)

A nova corrida espacial é, fundamentalmente, um desafio de **engenharia de
software**. Empresas privadas (SpaceX, Blue Origin, Rocket Lab, Planet Labs,
Amazon Kuiper, D-Orbit) operam no espaço como empresas de tecnologia. Dados
satelitais hoje orientam agronegócio, antecipam desastres, monitoram
desmatamento e guiam decisões em tempo real.

**Economia espacial:** > US$ 630 bi em 2023, projeção > US$ 1 tri até 2030.
**Dados abertos disponíveis:** NASA, ESA, INPE, Copernicus — gratuitos.

### Frentes do desafio
- **Sistemas de Missão** — controle e telemetria de satélites/veículos.
- **Dados Orbitais** — imagens satelitais, sensoriamento remoto, APIs abertas.
- **Edge em Órbita** — computação embarcada, latência crítica, radiação.
- **Conectividade Global** — IoT via satélite, redes mesh orbitais.

### Regras gerais da GS
- Projeto **único por equipe**; cada disciplina entrega uma **camada** do todo.
- O resultado final deve ser **coeso**, não peças soltas. As entregas se complementam.
- Equipe de até 5 alunos. Envio por disciplina feito por **um** integrante no portal FIAP.

### ODS prioritários (escolher ≥1)
ODS 2 (Agricultura sustentável) · ODS 9 (Inovação e infraestrutura) ·
ODS 10 (Redução de desigualdades) · ODS 11 (Cidades inteligentes) ·
ODS 13 (Ação climática).

### O que os avaliadores esperam
- Problema real com clareza (quem sofre, quanto custa, por que importa).
- Solução tecnológica coerente (arquitetura, stack, componentes, fluxos).
- Conexão com a Indústria Espacial (dado orbital, infraestrutura ou inspiração).
- Integração entre disciplinas (o projeto é um todo).
- Viabilidade técnica (não precisa estar 100% pronto, mas precisa ser possível).
- Impacto real (conexão com ODS, benefício mensurável).

---

## 3. Contexto da Demanda Específica — Applied Computer Vision

### Descrição da entrega
Desenvolver uma solução de **Visão Computacional** aplicada à Indústria
Espacial, usando **redes neurais convolucionais (CNNs) treinadas do zero** para
resolver um problema de **classificação de imagens** ligado à solução proposta
no projeto integrado da GS.

### Restrições NÃO-NEGOCIÁVEIS
1. **Tarefa = classificação de imagens** (não detecção, não segmentação).
2. **≥ 2 arquiteturas de CNN criadas do ZERO** — implementadas pela equipe em
   **PyTorch** ou **TensorFlow/Keras**. **PROIBIDO usar modelos pré-treinados**
   (sem pesos ImageNet, sem transfer learning).
3. **Dataset** criado/coletado/adaptado/selecionado pela equipe, relacionado ao
   tema. Exemplos válidos: imagens satelitais, aéreas, áreas urbanas/agrícolas/
   ambientais, classificação de solo/vegetação/água/queimadas/nuvens/áreas
   construídas, imagens simuladas de sensores/satélites/componentes espaciais.
4. **Acurácia mínima de referência: 88% no conjunto de teste.** Se não atingir,
   exige justificativa técnica consistente (limitações + melhorias futuras).
5. Comparar as arquiteturas com métricas: **acurácia, loss, matriz de confusão,
   análise de erros** — e justificar tecnicamente as diferenças entre os modelos.

### Conteúdo obrigatório do projeto
1. **Definição do problema** de VC e conexão com a solução espacial.
2. **Dataset**: origem, nº de classes, nº de imagens/classe, split treino/val/
   teste, pré-processamento.
3. **Treinamento de CNNs do zero**: criar/treinar/comparar ≥2 redes próprias,
   mostrando evolução de acurácia e loss por época.
4. **Avaliação**: métricas quantitativas + análise qualitativa (acertos/erros).
5. **Comparação entre arquiteturas**: qual foi melhor e por quê.
6. **Demonstração funcional**: modelo rodando em imagens novas — notebook, API,
   web simples, **Streamlit**, **Gradio** ou **FastAPI**.

### Critérios de avaliação (total 100 pts)
| Critério | Pts |
|---|---|
| Definição do problema e conexão com a Indústria Espacial | 10 |
| Dataset, preparação dos dados e pré-processamento | 15 |
| **Implementação das CNNs do zero** | **20** |
| Treinamento, validação e acompanhamento das métricas | 15 |
| **Comparação entre modelos e análise técnica dos resultados** | **20** |
| Demonstração funcional da solução | 10 |
| Documentação, organização e clareza técnica | 10 |

> **Onde a nota se concentra:** CNNs do zero (20) + comparação/análise (20) = 40 pts.
> É o coração técnico da entrega.

### Formato e instrução de entrega
Repositório **GitHub público** contendo no mínimo:
- notebook de treinamento (`.ipynb`);
- scripts Python (se houver);
- arquivo com a arquitetura dos modelos;
- pesos do melhor modelo (quando aplicável);
- imagens de teste / amostra do dataset (quando permitido);
- `requirements.txt` ou instruções de instalação;
- aplicação de demonstração (caso exista);
- **README** com instruções de execução **e nomes dos integrantes**.

**Vídeo de até 3 minutos** apresentando a proposta e o funcionamento. Demo pode
ser: notebook executável / API local (FastAPI, Flask) / app (Streamlit, Gradio).

> **ATENÇÃO:** A entrega deve conter o **link do GitHub público** + **link do
> vídeo no YouTube** (não enviar o arquivo de vídeo).

---

## 4. Restrições de prazo e risco

- **Hoje: 02/06/2026 · Entrega: 09/06/2026 → ~7 dias.**
- **Risco crítico:** atingir **88% com CNN do zero** depende quase inteiramente
  da **escolha do dataset/problema**. Problema bem separado (poucas classes,
  classes visualmente distintas) → 88% confortável. Dataset cru, multiclasse e
  desbalanceado → difícil passar de 70%. **A escolha do problema é ~80% da nota.**
- Correção ocorre 1 semana após o encerramento.

---

## 5. Status

- [x] Contexto registrado
- [ ] Tema-mãe da GS definido (integração entre disciplinas)
- [ ] Problema de VC + dataset escolhidos
- [ ] Arquiteturas das 2+ CNNs definidas
- [ ] Pipeline de treino/avaliação
- [ ] Demo funcional
- [ ] README + vídeo + repo público
