# Design / Spec — Solução ACV · Global Solution 2026/1 · Grupo TPGN

**Data:** 2026-06-02 · **Entrega:** 2026-06-09 · **Disciplina:** Applied Computer Vision
**Autoria do código:** GUILHERME ROCHA BIANCHINI (convenção `Author` em todo cabeçalho)
**Princípio-guia:** pragmatismo brutal — qualidade com todas as entregas completas, **zero overengineering**.

> Contexto completo do desafio e da disciplina em [`contexto.md`](./contexto.md).

---

## 1. Problema de Visão Computacional & conexão com a GS

**Tarefa:** classificação de imagens — **tiles Sentinel-2 → 10 classes de cobertura/uso do solo**:
`AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake`.

**Conexão com o tema-mãe (TPGN):** a camada de VC é o "olho" da plataforma de **dados de
precisão para agro e cidades** — converte pixel orbital cru em rótulo acionável (quanto de
plantação, floresta, área construída, água numa região). Dado orbital → informação útil.

**ODS:** 2 (agro sustentável), 11 (cidades inteligentes), 9 (inovação/infra).

*Atende:* Definição do problema + conexão espacial (10 pts).

---

## 2. Dataset — EuroSAT (RGB)

| Item | Decisão |
|---|---|
| Fonte | EuroSAT (Helber et al.), tiles Sentinel-2 · 27.000 imgs · 64×64 · 10 classes |
| Carregamento | `tensorflow_datasets` (`eurosat/rgb`) — "selecionar/adaptar dataset" é permitido |
| Split | **Estratificado, criado pela equipe**: 70% treino / 15% val / 15% teste |
| Pré-proc. | normalização `[0,1]` (rescale 1/255); EDA com contagem por classe + grade de amostras |
| Balanceamento | EuroSAT ~2–3k/classe (razoavelmente balanceado) — reportar contagens |
| Augmentation | só nos modelos que pedem: flip H/V, rotação, zoom, contraste (válido p/ vista aérea) |

*Atende:* Dataset/preparação/pré-processamento (15 pts).

---

## 3. As 3 CNNs do zero (Keras) — a história técnica

Cada modelo isola **uma variável** para justificar tecnicamente a diferença.

| Modelo | Arquitetura | Variável isolada | Esperado |
|---|---|---|---|
| **M1 — Baseline** | 2× (Conv→MaxPool) → Flatten → Dense(128) → softmax. Sem BN/Dropout/aug | piso de referência; deve **overfittar** | ~80–85% |
| **M2 — Deep+Reg** | 4 blocos [Conv-BN-ReLU ×2 → MaxPool → Dropout] (32→64→128→128) → Dense(256)→Dropout→softmax + **augmentation** + EarlyStopping + ReduceLROnPlateau | profundidade + regularização + aug | **~90–95% (vencedor)** |
| **M3 — Variação GAP** | mesmo backbone do M2, **GlobalAveragePooling** no lugar de Flatten+Dense | papel da cabeça de classificação; **menos parâmetros** | ~89–93% |

**Sem pré-treinado, sem transfer learning.** Todas implementadas do zero.
**Narrativa de comparação:** M1→M2 = efeito de profundidade/regularização/augmentation (fecha o
gap treino/val); M2→M3 = trade-off nº de parâmetros × acurácia (GAP).

*Atende:* Implementação das CNNs do zero (20 pts) + base da comparação (20 pts).

---

## 4. Treino, avaliação & comparação

- **Ambiente:** notebook `.ipynb` rodável no **Google Colab (GPU grátis)**; roda local também.
- **Acompanhamento:** curvas de **accuracy e loss** (treino vs val) por modelo — overfitting do M1 visível.
- **Avaliação no teste:** **matriz de confusão** + `classification_report` (precision/recall/F1 por
  classe) + **análise de erros** (imagens mal classificadas; pares difíceis: Highway×River,
  AnnualCrop×PermanentCrop).
- **Tabela comparativa:** nº de parâmetros · acurácia teste · loss teste · tempo de treino → escolha
  justificada do melhor (M2).
- **Meta:** ≥88% no teste. EuroSAT RGB do zero bate ~95% em CNNs simples publicadas → meta realista.
  *Fallback se (improvável) não atingir:* mais épocas/augmentation ou justificativa técnica de limitações.
- **Reprodutibilidade:** seeds fixas (tf/numpy), `requirements.txt` pinado.

*Atende:* Treino/validação/métricas (15 pts) + Comparação/análise técnica (20 pts).

---

## 5. Demonstração funcional — FastAPI + página única (gated)

- **Backend:** `api/main.py` (FastAPI) → `POST /predict` recebe imagem, carrega o **melhor modelo
  salvo (M2)**, retorna `{classe, probabilidades[]}`. Serve também a `index.html`.
- **Front:** **uma página HTML** (vanilla JS + CSS dark premium) servida pela própria API:
  upload/drag-drop → predição + **barras de confiança** por classe + tiles de amostra. **Sem Next.js,
  sem build, sem deploy separado.**
- **Regra de prazo:** só começa **depois** do modelo treinado e salvo.
- **Fallback garantido:** célula de predição em imagens novas no próprio notebook (cumpre a regra
  mesmo se o front não fechar).

*Atende:* Demonstração funcional (10 pts).

---

## 6. Estrutura do repositório (enxuta)

```
computer-vision/
├── contexto.md            ✓
├── design.md              ✓ (este arquivo)
├── README.md              # integrantes TPGN + instruções + resultados + prints
├── requirements.txt
├── notebooks/
│   └── treinamento.ipynb  # fonte única: EDA → preprocess → M1/M2/M3 → treino → eval → comparação → demo fallback
├── models/
│   └── best_model.keras   # pesos do melhor (M2)
├── reports/figures/       # curvas, matriz de confusão, análise de erros (PNGs p/ README)
├── api/
│   ├── main.py            # FastAPI: /predict + serve index.html      (Author: GUILHERME ROCHA BIANCHINI)
│   └── static/index.html  # front single-page, dark premium, vanilla JS
└── samples/               # imagens novas p/ testar
```

*Atende:* Documentação/organização/clareza (10 pts).

---

## 7. Stack & dependências

`tensorflow` · `tensorflow-datasets` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`
(métricas/matriz de confusão) · `fastapi` · `uvicorn` · `python-multipart` · `pillow`. Versões pinadas.

---

## 8. Ordem de execução (7 dias)

1. Repo + `requirements.txt` + carregamento/split EuroSAT + EDA
2. M1 baseline (treino/aval)
3. M2 + M3 (treino/aval)
4. Comparação + figuras + análise de erros + salvar `best_model.keras`
5. FastAPI + página única (gated no modelo pronto)
6. README + reprodutibilidade + **gravar vídeo ≤3min → YouTube**
7. Buffer

---

## 9. Decisões anti-overengineering (o que deliberadamente NÃO faremos)

- **Sem pacote `src/`** — notebook é a fonte única. Scripts só "se houver".
- **Sem Next.js** — página HTML única servida pela FastAPI; mesma aparência no vídeo, 1/5 do trabalho.
- **Sem suíte de testes/TDD** — projeto de notebook ML; a verificação é a métrica no teste.
- **Sem Docker/deploy em nuvem** — demo roda local; entrega é repo + vídeo.
- **Sem tuning exagerado** — 3 arquiteturas que contam uma história, não um grid search.
- **Sem dataset customizado coletado à mão** — EuroSAT selecionado/adaptado (permitido) garante os 88%.

---

## 10. Riscos

| Risco | Mitigação |
|---|---|
| Front consome tempo do ML | Demo é gated; fallback = predição no notebook |
| Treino lento em CPU | Colab GPU grátis |
| Não atingir 88% (improvável) | EuroSAT do zero ~95%; senão, mais épocas/aug ou justificativa técnica |
| Repo GitHub não criado a tempo | Git init + push é etapa explícita do plano (push só com confirmação) |
```
