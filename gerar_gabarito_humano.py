"""
gerar_gabarito_humano.py — versão corrigida
Substitui pivot_table por pivot() explícito com colunas garantidas
"""

import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"C:\DATA\salva_3\data_D\LARC\Artigonovo\Debentures - Copia\Data_Geral")

df = pd.read_csv(base / "resultados_avaliacao_dupla.csv")

# Garantir que strings estejam limpas
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()

modelos = sorted(df['modelo'].unique())
print(f"Modelos encontrados: {modelos}")

# ============================================================
# 1. MONTAR TABELA BASE (uma linha por doc+tópico+dataset)
# ============================================================

# Silver é igual para todas as linhas do mesmo (doc, tópico, dataset)
# então pegamos uma linha por grupo para ter o silver
df_silver = (
    df[['dataset_legivel', 'documento', 'topico', 'silver_label', 'silver_is_not_found']]
    .drop_duplicates(subset=['dataset_legivel', 'documento', 'topico'])
    .reset_index(drop=True)
)

# ============================================================
# 2. PIVOTAR RESPOSTAS: uma coluna por modelo
# ============================================================

df_resp = (
    df[['dataset_legivel', 'documento', 'topico', 'modelo', 'resposta']]
    .copy()
)

# Criar coluna de nome seguro para cada modelo
modelo_col_map = {m: f"resp_{m.replace(' ', '_').replace('.', '').replace('ã', 'a').replace('í', 'i')[:30]}" for m in modelos}
df_resp['col_nome'] = df_resp['modelo'].map(modelo_col_map)

resp_pivot = df_resp.pivot_table(
    index=['dataset_legivel', 'documento', 'topico'],
    columns='col_nome',
    values='resposta',
    aggfunc='first'
).reset_index()

# Garantir que colunas não tenham MultiIndex
resp_pivot.columns = [str(c).strip() for c in resp_pivot.columns]
resp_pivot.columns.name = None

print(f"Colunas após pivot respostas: {resp_pivot.columns.tolist()}")

# ============================================================
# 3. PIVOTAR SIMILARIDADES: uma coluna por modelo
# ============================================================

df_sim = (
    df[['dataset_legivel', 'documento', 'topico', 'modelo', 'similarity']]
    .copy()
)

sim_col_map = {m: f"sim_{m.replace(' ', '_').replace('.', '').replace('ã', 'a').replace('í', 'i')[:30]}" for m in modelos}
df_sim['col_nome'] = df_sim['modelo'].map(sim_col_map)

sim_pivot = df_sim.pivot_table(
    index=['dataset_legivel', 'documento', 'topico'],
    columns='col_nome',
    values='similarity',
    aggfunc='first'
).reset_index()

sim_pivot.columns = [str(c).strip() for c in sim_pivot.columns]
sim_pivot.columns.name = None

print(f"Colunas após pivot similaridades: {sim_pivot.columns.tolist()}")

# ============================================================
# 4. JUNTAR TUDO
# ============================================================

df_gabarito = (
    df_silver
    .merge(resp_pivot, on=['dataset_legivel', 'documento', 'topico'], how='left')
    .merge(sim_pivot,  on=['dataset_legivel', 'documento', 'topico'], how='left')
)

print(f"Colunas finais: {df_gabarito.columns.tolist()}")
print(f"Shape: {df_gabarito.shape}")

# ============================================================
# 5. CALCULAR DIVERGÊNCIA (para priorizar casos mais úteis)
# ============================================================

sim_cols = [c for c in df_gabarito.columns if c.startswith('sim_')]
if sim_cols:
    df_gabarito['divergencia'] = df_gabarito[sim_cols].apply(
        lambda row: pd.to_numeric(row, errors='coerce').std(), axis=1
    ).fillna(0)
else:
    df_gabarito['divergencia'] = 0

df_gabarito = df_gabarito.sort_values(
    ['dataset_legivel', 'divergencia'],
    ascending=[True, False]
).reset_index(drop=True)

# ============================================================
# 6. COLUNAS PARA ANOTAÇÃO HUMANA
# ============================================================

df_gabarito['gabarito_humano'] = ""   # resposta correta segundo o documento
df_gabarito['gabarito_is_nf']  = ""   # TRUE se o campo não existe no doc
df_gabarito['silver_correto']  = ""   # TRUE/FALSE: silver automático estava certo?
df_gabarito['melhor_modelo']   = ""   # qual modelo foi mais preciso
df_gabarito['notas_anotador']  = ""   # observações livres

# Remover coluna auxiliar
df_gabarito = df_gabarito.drop(columns=['divergencia'])

# ============================================================
# 7. SALVAR COMPLETO
# ============================================================

output_completo = base / "gabarito_humano_completo.csv"
df_gabarito.to_csv(output_completo, index=False, encoding='utf-8-sig')
print(f"\n✅ Gabarito completo : {len(df_gabarito)} linhas → {output_completo.name}")

# ============================================================
# 8. AMOSTRA ESTRATIFICADA (5 casos por tópico+dataset)
# ============================================================

N_POR_TOPICO_DATASET = 5

# Garantir que topico está como coluna antes do groupby
assert 'topico' in df_gabarito.columns, f"Colunas disponíveis: {df_gabarito.columns.tolist()}"

# Abordagem segura: usar rank por grupo em vez de groupby+apply
df_gabarito['_rank'] = (
    df_gabarito
    .groupby(['dataset_legivel', 'topico'], sort=False)
    .cumcount()
)

amostra = (
    df_gabarito[df_gabarito['_rank'] < N_POR_TOPICO_DATASET]
    .drop(columns=['_rank'])
    .reset_index(drop=True)
)

# Também remover _rank do df_gabarito
df_gabarito = df_gabarito.drop(columns=['_rank'])

output_amostra = base / "gabarito_humano_amostra.csv"
amostra.to_csv(output_amostra, index=False, encoding='utf-8-sig')
print(f"✅ Gabarito amostra  : {len(amostra)} linhas → {output_amostra.name}")
print(f"   ({N_POR_TOPICO_DATASET} casos × {amostra['topico'].nunique()} tópicos × {amostra['dataset_legivel'].nunique()} datasets)")

# ============================================================
# 9. RELATÓRIO DE COBERTURA
# ============================================================

print("\n📊 Cobertura da amostra por tópico:")
cob_topico = amostra['topico'].value_counts().sort_index()
print(cob_topico.to_string())

print("\n📊 Cobertura da amostra por dataset:")
cob_dataset = amostra['dataset_legivel'].value_counts().sort_index()
print(cob_dataset.to_string())

print("\n✅ Concluído.")