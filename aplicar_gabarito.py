"""
aplicar_gabarito.py

Lê silver_nf_inconsistentes.csv (revisado) e atualiza:
  - resultados_avaliacao_dupla.csv  → recalcula similarity e y_pred
  - Gera: resultados_avaliacao_dupla_corrigido.csv

Depois basta rodar analizar_resultados.py apontando para o arquivo corrigido.
NÃO precisa re-rodar o main.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

base = Path(r"C:\DATA\salva_3\data_D\LARC\Artigonovo\Debentures - Copia\Data_Geral")

SIMILARIDADE_THRESHOLD_TP = 0.85

# ============================================================
# 1. CARREGAR ARQUIVOS
# ============================================================

print("🔹 Carregando arquivos...")
df_result   = pd.read_csv(base / "resultados_avaliacao_dupla.csv")
df_gabarito = pd.read_csv(base / "silver_nf_inconsistentes.csv", encoding='utf-8-sig')

# Diagnóstico: mostrar colunas disponíveis
print(f"\n  Colunas no CSV de gabarito:")
print(f"  {df_gabarito.columns.tolist()}\n")

# Garantir que as colunas de anotação existem
if 'revisao_humana' not in df_gabarito.columns:
    print("⚠️  Coluna 'revisao_humana' não encontrada — criando vazia.")
    df_gabarito['revisao_humana'] = ""

if 'gabarito' not in df_gabarito.columns:
    print("⚠️  Coluna 'gabarito' não encontrada — criando vazia.")
    df_gabarito['gabarito'] = ""

# Salvar CSV atualizado com as colunas para preenchimento
csv_para_anotar = base / "silver_nf_inconsistentes.csv"
df_gabarito.to_csv(csv_para_anotar, index=False, encoding='utf-8-sig')
print(f"✅ CSV atualizado com colunas de anotação: {csv_para_anotar.name}")
print(f"   → Abra o arquivo, preencha 'revisao_humana' (TRUE/FALSE) e 'gabarito'")
print(f"   → Depois rode este script novamente.\n")

# Filtrar só linhas revisadas
df_rev = df_gabarito[
    df_gabarito['revisao_humana'].astype(str).str.strip().str.upper().isin(['TRUE','FALSE'])
].copy()

print(f"  Revisões encontradas : {len(df_rev)}")

if len(df_rev) == 0:
    print("⚠️  Nenhuma revisão encontrada ainda.")
    print("   Abra silver_nf_inconsistentes.csv, preencha e rode novamente.")
    exit()

# ============================================================
# 2. CARREGAR SBERT (só se houver correções FALSE)
# ============================================================

correcoes = df_rev[df_rev['revisao_humana'].str.upper() == 'FALSE'].copy()

if len(correcoes) > 0:
    print(f"\n🔹 Carregando SBERT para recalcular {len(correcoes)} silver corrigidos...")
    model_sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# ============================================================
# 3. APLICAR CORREÇÕES NO df_result
# ============================================================

df_corrigido = df_result.copy()
n_atualizados = 0

for _, row in correcoes.iterrows():
    doc     = row['documento']
    topico  = row['topico']
    dataset = row['dataset_legivel']
    novo_silver = str(row['gabarito']).strip()

    if not novo_silver or novo_silver.lower() in ['nan', '']:
        print(f"  ⚠️  Gabarito vazio para {doc} / {topico} — pulando")
        continue

    # Máscara: todas as linhas desse (doc, tópico, dataset)
    mask = (
        (df_corrigido['documento']       == doc)    &
        (df_corrigido['topico']          == topico) &
        (df_corrigido['dataset_legivel'] == dataset)
    )

    if mask.sum() == 0:
        print(f"  ⚠️  Não encontrado no resultado: {doc} / {topico} / {dataset}")
        continue

    # Recalcular similarity de cada modelo contra o novo silver
    silver_emb = model_sbert.encode([novo_silver])[0]
    respostas  = df_corrigido.loc[mask, 'resposta'].tolist()
    resp_embs  = model_sbert.encode(respostas)

    novas_sims = [float(util.cos_sim(silver_emb, e)[0][0]) for e in resp_embs]
    novos_pred = [1 if s >= SIMILARIDADE_THRESHOLD_TP else 0 for s in novas_sims]

    # Atualizar silver_label, silver_is_not_found, similarity, y_pred
    df_corrigido.loc[mask, 'silver_label']        = novo_silver
    df_corrigido.loc[mask, 'silver_is_not_found'] = False
    df_corrigido.loc[mask, 'similarity']          = novas_sims
    df_corrigido.loc[mask, 'y_pred']              = novos_pred

    # quality_applicable: agora pode ser True se resposta também é textual
    NOT_FOUND_TOKENS = {"não encontrado","nao encontrado","não informado","nao informado","not found","n/a",""}
    resp_col = df_corrigido.loc[mask, 'resposta'].str.strip().str.lower()
    df_corrigido.loc[mask, 'quality_applicable'] = ~resp_col.isin(NOT_FOUND_TOKENS)

    n_atualizados += mask.sum()
    print(f"  ✅ Corrigido: {doc[:30]} / {topico} / {dataset}")

print(f"\n✅ Total de linhas atualizadas: {n_atualizados}")

# ============================================================
# 4. SALVAR ARQUIVO CORRIGIDO
# ============================================================

out = base / "resultados_avaliacao_dupla_corrigido.csv"
df_corrigido.to_csv(out, index=False, encoding='utf-8-sig')
print(f"✅ Salvo: {out.name}")

# ============================================================
# 5. RELATÓRIO DE IMPACTO
# ============================================================

print("\n📊 Impacto da correção:")
cols = ['similarity', 'y_pred']
for col in cols:
    antes = df_result[col].mean()
    depois = df_corrigido[col].mean()
    print(f"  {col}: {antes:.4f} → {depois:.4f}  (Δ {depois-antes:+.4f})")

print("\n➡️  Próximo passo: rode analizar_resultados.py")
print("   Troque a linha de carregamento para:")
print('   df = pd.read_csv(base / "resultados_avaliacao_dupla_corrigido.csv")')