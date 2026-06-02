# 🛰️ Classificação de Cobertura do Solo — EuroSAT (CNNs do zero)

**Global Solution 2026/1 · Applied Computer Vision · Grupo TPGN (TechPulse GlobalNetwork)**
Engenharia de Software · 4º Ano · FIAP

Camada de **Visão Computacional** da solução integrada do TPGN: uma plataforma de **dados de
precisão para agro e cidades** alimentada por imagens de satélite. Este módulo classifica tiles
**Sentinel-2** em **10 classes de cobertura/uso do solo**, usando **redes neurais convolucionais
treinadas do zero** (sem modelos pré-treinados, sem transfer learning).

**Conexão com a Indústria Espacial:** transforma dado orbital cru (Sentinel-2 / Copernicus) em
informação acionável sobre o território. **ODS:** 2 (agricultura sustentável), 9 (inovação e
infraestrutura), 11 (cidades inteligentes).

---

## 👥 Integrantes — TPGN

| Nome | RM |
|---|---|
| GUILHERME ROCHA BIANCHINI | RM97974 |
| NIKOLAS RODRIGUES MOURA DOS SANTOS | RM551566 |
| PEDRO HENRIQUE PEDROSA TAVARES | RM97877 |
| RODRIGO BRASILEIRO | RM98952 |
| THIAGO JARDIM DE OLIVEIRA | RM551624 |

---

## 🎥 Vídeo de apresentação

➡️ **[LINK DO YOUTUBE — preencher após gravar]**

---

## 📊 Dataset — EuroSAT (RGB)

- **Fonte:** EuroSAT (Helber et al.), tiles Sentinel-2 · **27.000 imagens · 64×64 · 10 classes**.
- **Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
  PermanentCrop, Residential, River, SeaLake.
- **Carregamento:** `tensorflow_datasets` (`eurosat/rgb`).
- **Split estratificado (criado pela equipe):** 70% treino / 15% validação / 15% teste.
- **Pré-processamento:** normalização `[0,1]` via camada `Rescaling` dentro do modelo;
  data augmentation (flip, rotação, zoom, contraste) nos modelos regularizados.

---

## 🧠 Modelos (3 CNNs do zero, Keras)

| Modelo | Arquitetura | Variável isolada |
|---|---|---|
| **M1 — Baseline** | 2× (Conv→MaxPool) → Flatten → Dense(128) → softmax | piso de referência (overfitting) |
| **M2 — Deep+Reg** | 4 blocos [Conv-BN-ReLU ×2 → MaxPool → Dropout] + augmentation + Dense(256) | profundidade + regularização |
| **M3 — GAP** | mesmo backbone do M2 com GlobalAveragePooling no lugar de Flatten+Dense | papel da cabeça densa / nº de params |

---

## 📈 Resultados

> Preencher após rodar o notebook (valores da tabela `results`).

| Modelo | Parâmetros | Acurácia (teste) | Loss (teste) |
|---|---|---|---|
| M1 | _ | _ | _ |
| M2 | _ | _ | _ |
| M3 | _ | _ | _ |

**Melhor modelo:** _ · **Meta:** ≥ 88% no teste.

**Figuras** (em `reports/figures/`):
- `class_distribution.png` — distribuição por classe
- `samples.png` — amostras do dataset
- `history_M1/M2/M3.png` — curvas de accuracy/loss
- `confusion_matrix.png` — matriz de confusão (melhor modelo)
- `errors.png` — exemplos mal classificados

---

## ▶️ Como executar

### Opção A — Google Colab (recomendado, GPU grátis)
1. Suba `notebooks/treinamento.ipynb` no Colab.
2. *Ambiente de execução → Alterar tipo de ambiente → GPU*.
3. Rode todas as células (o `tensorflow_datasets` baixa o EuroSAT automaticamente).
4. Ao final, o melhor modelo é salvo em `models/best_model.keras`.

### Opção B — Local
```bash
pip install -r requirements.txt
jupyter notebook notebooks/treinamento.ipynb
```

### Demo (API + página) — após treinar e salvar o modelo
```bash
uvicorn api.main:app --reload --port 8000
# abra http://localhost:8000/
```
A página permite enviar uma imagem e ver a classe prevista + barras de confiança por classe.
Há também uma **demonstração no próprio notebook** (seção 7), caso prefira não subir a API.

---

## 📁 Estrutura

```
computer-vision/
├── contexto.md / design.md / plan.md   # contexto, spec e plano
├── requirements.txt
├── notebooks/treinamento.ipynb          # pipeline completo de ML
├── models/                              # best_model.keras + classes.json (gerados)
├── reports/figures/                     # gráficos (gerados)
├── api/main.py + api/static/index.html  # demo FastAPI + página
├── samples/                             # imagens de teste (geradas)
└── README.md
```

---

*Author do código: GUILHERME ROCHA BIANCHINI.*
