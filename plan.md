# ACV / TPGN — Plano de Implementação

> **Para workers:** este plano adapta o formato de checkboxes do superpowers a um projeto
> **notebook-ML** (sem suíte pytest, por decisão do spec). Verificação = saída de célula / métrica.
> Steps usam `- [ ]` para tracking. Implementar tarefa-a-tarefa.

**Goal:** Classificar cobertura do solo (EuroSAT, 10 classes) com 3 CNNs treinadas do zero em Keras, comparar arquiteturas, atingir ≥88% no teste e expor o melhor modelo numa demo FastAPI + página única.

**Architecture:** Notebook único como fonte da verdade (EDA → split estratificado → M1/M2/M3 → treino → avaliação → comparação). Melhor modelo salvo em `.keras` é consumido por uma API FastAPI que serve uma página HTML de demonstração. Sem `src/`, sem Next.js, sem Docker.

**Tech Stack:** Python · TensorFlow/Keras · tensorflow-datasets · scikit-learn · matplotlib/seaborn · FastAPI · uvicorn · Pillow · HTML/CSS/JS vanilla.

**Autoria:** todo cabeçalho de código → `Author: GUILHERME ROCHA BIANCHINI`.

---

## Mapa de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `requirements.txt` | dependências pinadas |
| `notebooks/treinamento.ipynb` | pipeline completo de ML (fonte única) |
| `models/best_model.keras` | pesos do melhor modelo (gerado pelo notebook) |
| `models/classes.json` | ordem das 10 classes (gerado) |
| `reports/figures/*.png` | curvas, matriz de confusão, erros (gerados) |
| `api/main.py` | FastAPI: `/predict` + serve a página |
| `api/static/index.html` | front single-page (dark premium) |
| `samples/` | imagens novas p/ testar a demo |
| `README.md` | integrantes, instruções, resultados |

---

## FASE 0 — Setup

### Task 0.1: Esqueleto + requirements

**Files:**
- Create: `requirements.txt`
- Create (dirs): `notebooks/`, `models/`, `reports/figures/`, `api/static/`, `samples/`

- [ ] **Step 1: Criar diretórios**

```powershell
New-Item -ItemType Directory -Force -Path notebooks,models,reports\figures,api\static,samples | Out-Null
```

- [ ] **Step 2: Escrever `requirements.txt`**

```
tensorflow==2.16.1
tensorflow-datasets==4.9.4
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
pillow==10.3.0
```

- [ ] **Step 3: Verificar**

Run: `Test-Path notebooks,models,reports\figures,api\static,samples,requirements.txt`
Expected: seis `True`.

---

## FASE 1 — Dados (no `notebooks/treinamento.ipynb`)

> Cada "step" abaixo é uma célula do notebook. Cabeçalho markdown no topo com título, integrantes
> TPGN e `Author: GUILHERME ROCHA BIANCHINI`.

### Task 1.1: Imports + reprodutibilidade

- [ ] **Step 1: Célula de setup**

```python
# Author: GUILHERME ROCHA BIANCHINI — TPGN / Global Solution 2026
import os, random, json, io
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TF:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
```

- [ ] **Step 2: Rodar** → Expected: imprime versão do TF; lista GPU (Colab) ou `[]` (CPU local).

### Task 1.2: Carregar EuroSAT em memória

- [ ] **Step 1: Célula de load**

```python
ds, info = tfds.load("eurosat/rgb", split="train", as_supervised=True,
                     with_info=True, batch_size=-1)
images, labels = tfds.as_numpy(ds)           # (27000,64,64,3) uint8 ; (27000,)
CLASSES = list(info.features["label"].names)
print(images.shape, labels.shape, "| classes:", len(CLASSES))
print(CLASSES)
```

- [ ] **Step 2: Rodar** → Expected: `(27000, 64, 64, 3) (27000,) | classes: 10` e a lista das 10 classes.

### Task 1.3: EDA (distribuição + amostras)

- [ ] **Step 1: Distribuição por classe**

```python
counts = np.bincount(labels)
plt.figure(figsize=(10,4))
sns.barplot(x=CLASSES, y=counts)
plt.xticks(rotation=45, ha="right"); plt.title("Imagens por classe — EuroSAT")
plt.tight_layout(); plt.savefig("reports/figures/class_distribution.png", dpi=120); plt.show()
print(dict(zip(CLASSES, counts.tolist())))
```

- [ ] **Step 2: Grade de amostras**

```python
plt.figure(figsize=(12,5))
for i, c in enumerate(range(10)):
    idx = np.where(labels == c)[0][0]
    plt.subplot(2,5,i+1); plt.imshow(images[idx]); plt.axis("off"); plt.title(CLASSES[c], fontsize=9)
plt.tight_layout(); plt.savefig("reports/figures/samples.png", dpi=120); plt.show()
```

- [ ] **Step 3: Rodar ambas** → Expected: 2 figuras salvas; barras ~2000–3000 por classe (relatar balanceamento no README).

### Task 1.4: Split estratificado 70/15/15

- [ ] **Step 1: Split**

```python
X_tmp, X_test, y_tmp, y_test = train_test_split(
    images, labels, test_size=0.15, stratify=labels, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=SEED)  # ~70/15/15
for n, y in [("train",y_train),("val",y_val),("test",y_test)]:
    print(f"{n}: {len(y)}  dist={np.bincount(y)}")
```

- [ ] **Step 2: Rodar** → Expected: train ≈18900, val ≈4050, test ≈4050; distribuição equilibrada entre classes nos três conjuntos (confirma estratificação).

- [ ] **Step 3: Commit (fim da Fase 1)**

```powershell
git add notebooks/treinamento.ipynb reports/figures
git commit -m "data: load EuroSAT, EDA e split estratificado 70/15/15"
```

---

## FASE 2 — Modelos (no notebook)

### Task 2.1: M1 — Baseline simples

- [ ] **Step 1: Célula `build_m1`**

```python
def build_m1(input_shape=(64,64,3), n_classes=10):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="M1_baseline")
    return m

build_m1().summary()
```

- [ ] **Step 2: Rodar** → Expected: `summary()` imprime camadas; sem BatchNormalization/Dropout (proposital).

### Task 2.2: Bloco convolucional + augmentation (compartilhados por M2/M3)

- [ ] **Step 1: Célula**

```python
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augment")

def conv_block(x, filters):
    for _ in range(2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    return x
```

- [ ] **Step 2: Rodar** → Expected: sem erro (definições).

### Task 2.3: M2 — Deep + Regularizado (cabeça Dense)

- [ ] **Step 1: Célula `build_m2`**

```python
def build_m2(input_shape=(64,64,3), n_classes=10):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = data_aug(inp)
    x = tf.keras.layers.Rescaling(1./255)(x)
    for f in [32, 64, 128, 128]:
        x = conv_block(x, f)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="M2_deep_reg")

build_m2().summary()
```

- [ ] **Step 2: Rodar** → Expected: ~1–2M params; presença de BatchNormalization e Dropout.

### Task 2.4: M3 — Variação GAP (mesmo backbone, cabeça Global Average Pooling)

- [ ] **Step 1: Célula `build_m3`**

```python
def build_m3(input_shape=(64,64,3), n_classes=10):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = data_aug(inp)
    x = tf.keras.layers.Rescaling(1./255)(x)
    for f in [32, 64, 128, 128]:
        x = conv_block(x, f)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="M3_gap")

build_m3().summary()
```

- [ ] **Step 2: Rodar** → Expected: nº de params **bem menor** que M2 (sem Flatten+Dense256) — esse é o ponto da comparação.

- [ ] **Step 3: Commit (fim da Fase 2)**

```powershell
git add notebooks/treinamento.ipynb
git commit -m "models: define M1 baseline, M2 deep+reg, M3 GAP (todas do zero)"
```

---

## FASE 3 — Treino (no notebook)

### Task 3.1: Função de treino reutilizável

- [ ] **Step 1: Célula**

```python
def compile_and_fit(model, epochs, use_callbacks):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = []
    if use_callbacks:
        cbs = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)]
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=64, callbacks=cbs, verbose=2)
    return hist
```

- [ ] **Step 2: Rodar** → Expected: sem erro.

### Task 3.2: Treinar M1 (sem callbacks → expor overfitting)

- [ ] **Step 1: Célula**

```python
m1 = build_m1()
h1 = compile_and_fit(m1, epochs=30, use_callbacks=False)
```

- [ ] **Step 2: Rodar** → Expected: acurácia de treino sobe perto de ~1.0 enquanto a de val estaciona (gap = overfitting). Teste ~80–85%.

### Task 3.3: Treinar M2

- [ ] **Step 1: Célula**

```python
m2 = build_m2()
h2 = compile_and_fit(m2, epochs=50, use_callbacks=True)
```

- [ ] **Step 2: Rodar** → Expected: val accuracy ≥ ~0.90; gap treino/val pequeno (regularização funcionando).

### Task 3.4: Treinar M3

- [ ] **Step 1: Célula**

```python
m3 = build_m3()
h3 = compile_and_fit(m3, epochs=50, use_callbacks=True)
```

- [ ] **Step 2: Rodar** → Expected: val accuracy ~0.89–0.93 com muito menos parâmetros que M2.

### Task 3.5: Curvas de treino (acc/loss)

- [ ] **Step 1: Célula**

```python
def plot_history(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(11,4))
    ax[0].plot(hist.history["accuracy"], label="treino")
    ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].set_title(f"{name} — accuracy"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="treino")
    ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].set_title(f"{name} — loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"reports/figures/history_{name}.png", dpi=120); plt.show()

for h, n in [(h1,"M1"),(h2,"M2"),(h3,"M3")]:
    plot_history(h, n)
```

- [ ] **Step 2: Rodar** → Expected: 3 figuras salvas; gap visível no M1, controlado no M2/M3.

- [ ] **Step 3: Commit (fim da Fase 3)**

```powershell
git add notebooks/treinamento.ipynb reports/figures
git commit -m "train: M1/M2/M3 treinados + curvas de accuracy/loss"
```

---

## FASE 4 — Avaliação, comparação e salvar melhor

### Task 4.1: Métricas no teste + tabela comparativa

- [ ] **Step 1: Célula**

```python
def eval_model(model, name):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return {"modelo": name, "params": model.count_params(),
            "test_acc": round(float(acc),4), "test_loss": round(float(loss),4)}

results = [eval_model(m, n) for m, n in [(m1,"M1"),(m2,"M2"),(m3,"M3")]]
for r in results: print(r)
best_name = max(results, key=lambda r: r["test_acc"])["modelo"]
best_model = {"M1":m1, "M2":m2, "M3":m3}[best_name]
print("Melhor:", best_name)
```

- [ ] **Step 2: Rodar** → Expected: dict por modelo; melhor com `test_acc ≥ 0.88`. Se < 0.88, adicionar célula markdown com justificativa técnica (limitações + melhorias).

### Task 4.2: Matriz de confusão (melhor modelo)

- [ ] **Step 1: Célula**

```python
y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predito"); plt.ylabel("Real"); plt.title(f"Matriz de confusão — {best_name}")
plt.tight_layout(); plt.savefig("reports/figures/confusion_matrix.png", dpi=120); plt.show()
print(classification_report(y_test, y_pred, target_names=CLASSES))
```

- [ ] **Step 2: Rodar** → Expected: heatmap salvo; `classification_report` com precision/recall/F1 por classe.

### Task 4.3: Análise de erros (exemplos mal classificados)

- [ ] **Step 1: Célula**

```python
wrong = np.where(y_pred != y_test)[0]
sel = wrong[:10]
plt.figure(figsize=(13,5))
for i, idx in enumerate(sel):
    plt.subplot(2,5,i+1); plt.imshow(X_test[idx]); plt.axis("off")
    plt.title(f"R:{CLASSES[y_test[idx]]}\nP:{CLASSES[y_pred[idx]]}", fontsize=8)
plt.tight_layout(); plt.savefig("reports/figures/errors.png", dpi=120); plt.show()
print("Total de erros:", len(wrong), "de", len(y_test))
```

- [ ] **Step 2: Rodar** → Expected: grade de erros salva; comentar pares confusos (ex.: Highway×River) em célula markdown.

### Task 4.4: Salvar melhor modelo + classes

- [ ] **Step 1: Célula**

```python
best_model.save("models/best_model.keras")
with open("models/classes.json", "w") as f:
    json.dump(CLASSES, f, ensure_ascii=False, indent=2)
print("Salvo:", best_name)
```

- [ ] **Step 2: Rodar** → Expected: `models/best_model.keras` e `models/classes.json` criados.

- [ ] **Step 3: Célula markdown de comparação** — tabela final (params × acc × loss), qual venceu e **por quê** (M1→M2: profundidade+regularização+aug fecham o gap; M2→M3: GAP corta params com acurácia próxima).

- [ ] **Step 4: Commit (fim da Fase 4)**

```powershell
git add notebooks/treinamento.ipynb reports/figures models/best_model.keras models/classes.json
git commit -m "eval: matriz de confusao, analise de erros, comparacao e melhor modelo salvo"
```

---

## FASE 5 — Demo FastAPI + página única (gated: só após Fase 4)

### Task 5.1: Backend FastAPI

**Files:** Create `api/main.py`

- [ ] **Step 1: Escrever `api/main.py`**

```python
"""
TPGN — Global Solution 2026/1 — Applied Computer Vision
API de inferencia de cobertura do solo (EuroSAT).
Author: GUILHERME ROCHA BIANCHINI
"""
import io, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
ROOT = BASE.parent
model = tf.keras.models.load_model(ROOT / "models" / "best_model.keras")
CLASSES = json.loads((ROOT / "models" / "classes.json").read_text(encoding="utf-8"))

app = FastAPI(title="TPGN — EuroSAT Land Cover")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((64, 64))
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo de imagem invalido")
    arr = np.expand_dims(np.array(img), 0)          # Rescaling esta dentro do modelo
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "classe": CLASSES[idx],
        "confianca": float(probs[idx]),
        "probabilidades": {c: float(p) for c, p in zip(CLASSES, probs)},
    }

# montar estaticos por ultimo para nao sombrear /predict
app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
```

- [ ] **Step 2: Verificação (smoke test)**

Run: `uvicorn api.main:app --port 8000` (em outro terminal) e
`curl.exe -X POST -F "file=@samples/exemplo.jpg" http://localhost:8000/predict`
Expected: JSON com `classe`, `confianca`, `probabilidades` (10 chaves). (Copiar 1–2 imagens de teste do dataset para `samples/` antes — célula opcional no notebook usando `X_test`.)

### Task 5.2: Front single-page (dark premium)

**Files:** Create `api/static/index.html`

- [ ] **Step 1: Escrever `api/static/index.html`**

```html
<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TPGN · Cobertura do Solo — EuroSAT</title>
<style>
  :root{--bg:#0a0e14;--card:#121826;--accent:#4f8cff;--txt:#e6edf3;--muted:#8b97a8}
  *{box-sizing:border-box;font-family:system-ui,Segoe UI,Roboto,sans-serif}
  body{margin:0;background:var(--bg);color:var(--txt);display:flex;min-height:100vh;
       align-items:center;justify-content:center;padding:24px}
  .card{background:var(--card);border:1px solid #1e2940;border-radius:16px;padding:28px;
        width:min(520px,100%);box-shadow:0 20px 60px rgba(0,0,0,.5)}
  h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);font-size:13px;margin:0 0 20px}
  .drop{border:2px dashed #2a3956;border-radius:12px;padding:32px;text-align:center;
        cursor:pointer;transition:.2s}.drop:hover{border-color:var(--accent);background:#0e1626}
  img#preview{max-width:160px;border-radius:10px;margin-top:14px;display:none}
  .bar{height:22px;background:#0e1626;border-radius:6px;overflow:hidden;margin:6px 0}
  .bar>div{height:100%;background:linear-gradient(90deg,#4f8cff,#7aa7ff);
           display:flex;align-items:center;padding-left:8px;font-size:12px;white-space:nowrap}
  .row{display:flex;justify-content:space-between;font-size:13px;margin-top:10px}
  .top{font-size:18px;font-weight:700;margin:18px 0 4px;color:var(--accent)}
</style>
</head>
<body>
  <div class="card">
    <h1>Cobertura do Solo · EuroSAT</h1>
    <p class="sub">TPGN — TechPulse GlobalNetwork · Global Solution 2026 · CNN do zero</p>
    <label class="drop" id="drop">
      Clique ou arraste uma imagem de satélite (64×64)
      <input type="file" id="file" accept="image/*" hidden>
      <br><img id="preview">
    </label>
    <div id="out"></div>
  </div>
<script>
const file=document.getElementById('file'),drop=document.getElementById('drop'),
      preview=document.getElementById('preview'),out=document.getElementById('out');
drop.addEventListener('dragover',e=>{e.preventDefault()});
drop.addEventListener('drop',e=>{e.preventDefault();file.files=e.dataTransfer.files;go()});
file.addEventListener('change',go);
async function go(){
  if(!file.files.length)return;
  preview.src=URL.createObjectURL(file.files[0]);preview.style.display='inline-block';
  out.innerHTML='<p class="sub">Classificando…</p>';
  const fd=new FormData();fd.append('file',file.files[0]);
  const r=await fetch('/predict',{method:'POST',body:fd});
  if(!r.ok){out.innerHTML='<p class="sub">Erro na predição.</p>';return}
  const d=await r.json();
  const probs=Object.entries(d.probabilidades).sort((a,b)=>b[1]-a[1]);
  out.innerHTML='<div class="top">'+d.classe+' · '+(d.confianca*100).toFixed(1)+'%</div>'+
    probs.map(([c,p])=>'<div class="row"><span>'+c+'</span><span>'+(p*100).toFixed(1)+'%</span></div>'+
      '<div class="bar"><div style="width:'+(p*100).toFixed(1)+'%"></div></div>').join('');
}
</script>
</body>
</html>
```

- [ ] **Step 2: Verificação manual**

Abrir `http://localhost:8000/` → arrastar imagem → ver classe + barras de confiança ordenadas.

- [ ] **Step 3: Commit (fim da Fase 5)**

```powershell
git add api samples
git commit -m "demo: FastAPI /predict + pagina unica de classificacao"
```

---

## FASE 6 — Entrega final

### Task 6.1: README

**Files:** Create `README.md`

- [ ] **Step 1: Escrever `README.md`** com:
  - Título + descrição (problema de VC + conexão GS/espacial + ODS).
  - **Integrantes TPGN** (nomes + RMs — copiar de `contexto.md`).
  - Dataset (EuroSAT, 10 classes, split 70/15/15).
  - As 3 arquiteturas + **tabela de resultados** (params/acc/loss) com prints de `reports/figures/`.
  - Como rodar: Colab (notebook) **e** local (`pip install -r requirements.txt`; `uvicorn api.main:app`).
  - Link do vídeo (YouTube) — preencher no fim.

- [ ] **Step 2: Verificação** → README abre e renderiza tabela + imagens de `reports/figures/`.

### Task 6.2: Reprodutibilidade + repositório

- [ ] **Step 1: Criar `.gitignore`** (ignorar `__pycache__/`, `.ipynb_checkpoints/`, venvs).
- [ ] **Step 2: `git init` + primeiro push**

```powershell
git init; git add .; git commit -m "feat: entrega ACV TPGN — EuroSAT CNNs do zero + demo"
```

> **Push para GitHub público só após confirmação do Breq** (regra global). Conferir antes que
> nenhum secret/PII esteja versionado.

### Task 6.3: Vídeo (≤3 min)

- [ ] **Step 1:** Roteiro: problema/conexão GS (30s) → dataset+3 modelos (60s) → comparação/melhor (45s) → demo ao vivo na página (30s).
- [ ] **Step 2:** Gravar, subir no YouTube, colar link no README e na entrega do portal FIAP.

---

## Self-Review (cobertura do spec)

- §1 Problema/conexão → README + notebook markdown ✓
- §2 Dataset/split/pré-proc → Fase 1 ✓
- §3 três CNNs do zero → Fase 2 ✓
- §4 treino/curvas/avaliação/comparação/88% → Fases 3–4 ✓
- §5 demo FastAPI + página (fallback notebook) → Fase 5 (+ Task 4 prediz em imagens novas) ✓
- §6 estrutura repo → Task 0.1 + 6.2 ✓
- §7 stack/requirements → Task 0.1 ✓
- Autoria GUILHERME em cabeçalhos → Tasks 1.1, 5.1 ✓
- Anti-overengineering (sem src/, sem Next, sem testes) → respeitado ✓

Sem placeholders de código. Nomes consistentes (`build_m1/2/3`, `compile_and_fit`, `best_model`, `CLASSES`).
