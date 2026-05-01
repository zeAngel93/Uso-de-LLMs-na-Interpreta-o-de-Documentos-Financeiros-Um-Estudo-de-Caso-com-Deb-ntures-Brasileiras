"""
identificar_casos_dubios.py
Filtra casos onde o silver automático é menos confiável,
ordenados por prioridade de revisão.
"""

import pandas as pd
from pathlib import Path

base = Path(r"C:\DATA\salva_3\data_D\LARC\Artigonovo\Debentures - Copia\Data_Geral")

df = pd.read_csv(base / "resultados_avaliacao_dupla.csv")

# ============================================================
# CRITÉRIOS DE DUBIEDADE (cada um gera uma flag)
# ============================================================

# 1. Empate exato entre "Não encontrado" e respostas textuais
#    Ex: 2 LLMs disseram NF e 2 disseram algo → silver pode estar errado
contagem = (
    df.groupby(['dataset_legivel', 'documento', 'topico'])
    .agg(
        total=('modelo', 'count'),
        n_nf=('resposta_is_not_found', 'sum'),
        n_txt=('resposta_is_not_found', lambda x: (~x.astype(bool)).sum()),
        sim_std=('similarity', 'std'),
        sim_mean=('similarity', 'mean'),
        silver=('silver_label', 'first'),
        silver_is_nf=('silver_is_not_found', 'first')
    )
    .reset_index()
)

contagem['flag_empate']      = contagem['n_nf'] == contagem['n_txt']
contagem['flag_divergencia'] = contagem['sim_std'] > 0.15   # alta variação semântica
contagem['flag_baixa_sim']   = contagem['sim_mean'] < 0.75  # todos distantes do silver
contagem['flag_silver_unico']= contagem['n_txt'] == 1       # silver com só 1 voto textual

# Score de prioridade: quantas flags ativas
contagem['n_flags'] = (
    contagem['flag_empate'].astype(int) +
    contagem['flag_divergencia'].astype(int) +
    contagem['flag_baixa_sim'].astype(int) +
    contagem['flag_silver_unico'].astype(int)
)

# ============================================================
# PIVOTAR RESPOSTAS LADO A LADO (só para os casos duvidosos)
# ============================================================

dubios = contagem[contagem['n_flags'] >= 1].copy()

resp_pivot = df.pivot_table(
    index=['dataset_legivel', 'documento', 'topico'],
    columns='modelo',
    values='resposta',
    aggfunc='first'
).reset_index()
resp_pivot.columns.name = None

# Renomear colunas de forma segura
modelos = [c for c in resp_pivot.columns if c not in ['dataset_legivel','documento','topico']]
for m in modelos:
    abrev = m.replace(' ', '_').replace('.', '').replace('ã','a').replace('í','i')[:20]
    resp_pivot = resp_pivot.rename(columns={m: f'resp_{abrev}'})

dubios = pd.merge(dubios, resp_pivot,
                  on=['dataset_legivel', 'documento', 'topico'],
                  how='left')

# Ordenar: mais flags primeiro, depois maior desvio
dubios = dubios.sort_values(['n_flags', 'sim_std'], ascending=[False, False])

# Colunas para anotador
dubios['gabarito_humano'] = ""
dubios['silver_correto']  = ""   # TRUE / FALSE
dubios['notas']           = ""

# ============================================================
# SALVAR
# ============================================================

output = base / "casos_para_revisao_humana.csv"
dubios.to_csv(output, index=False, encoding='utf-8-sig')

print(f"Total de casos duvidosos: {len(dubios)} / {len(contagem)} ({100*len(dubios)/len(contagem):.1f}%)")
print(f"\nDistribuição por número de flags:")
print(dubios['n_flags'].value_counts().sort_index(ascending=False).to_string())
print(f"\nDistribuição por tipo de flag:")
for flag in ['flag_empate','flag_divergencia','flag_baixa_sim','flag_silver_unico']:
    print(f"  {flag}: {dubios[flag].sum()} casos")
print(f"\n✅ Salvo: {output.name}")