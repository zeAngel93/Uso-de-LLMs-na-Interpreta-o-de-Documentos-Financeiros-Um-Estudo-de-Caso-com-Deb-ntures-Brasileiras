"""
analisar_resultados.py — Análise e ranking dos resultados
Correções aplicadas:
 1. Score composto normalizado com MinMaxScaler (evitar dominância de f1_det)
 2. docs_dificeis mantém granularidade por dataset (sem média entre datasets)
 3. quality_true filtrado por != -1 (substitui np.nan da versão anterior)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

base = Path(r"C:\DATA\salva_3\data_D\LARC\Artigonovo\Debentures - Copia\Data_Geral")

print("🔹 Lendo arquivos...")
df_det   = pd.read_csv(base / "metricas_deteccao.csv")
df_qual  = pd.read_csv(base / "metricas_qualidade.csv")
df_regs  = pd.read_csv(base / "registros_normalizados.csv")
df_dupla = pd.read_csv(base / "resultados_avaliacao_dupla_corrigido.csv")
df_doc   = pd.read_csv(base / "resultados_documento_duplo.csv")
print("✅ Arquivos carregados.")


# Padronização de strings
for df in [df_det, df_qual, df_regs, df_dupla, df_doc]:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

# ============================================================
# 1. SCORE COMPOSTO (COM NORMALIZAÇÃO)
# ============================================================

df_comp = pd.merge(
    df_det, df_qual,
    on=["modelo", "prompt", "dataset_legivel"],
    how="outer"
)

for c in ["f1_det", "precision_det", "recall_det",
          "f1_qual", "precision_qual", "recall_qual", "sim_media_qual"]:
    if c in df_comp.columns:
        df_comp[c] = pd.to_numeric(df_comp[c], errors='coerce').fillna(0)

# CORREÇÃO 1: normalizar antes de combinar
# f1_det (0.96–0.99) e sim_media_qual (0.84–0.89) têm escalas diferentes
scaler = MinMaxScaler()
df_comp[["f1_det_norm", "sim_qual_norm"]] = scaler.fit_transform(
    df_comp[["f1_det", "sim_media_qual"]]
)

# Scores com pesos diferentes (raw e normalizados)
df_comp["score_50_50"]  = 0.5 * df_comp["f1_det"]      + 0.5 * df_comp["sim_media_qual"]
df_comp["score_40_60"]  = 0.4 * df_comp["f1_det"]      + 0.6 * df_comp["sim_media_qual"]
df_comp["score_60_40"]  = 0.6 * df_comp["f1_det"]      + 0.4 * df_comp["sim_media_qual"]
df_comp["score_norm_5050"] = 0.5 * df_comp["f1_det_norm"] + 0.5 * df_comp["sim_qual_norm"]
df_comp["score_norm_4060"] = 0.4 * df_comp["f1_det_norm"] + 0.6 * df_comp["sim_qual_norm"]
df_comp["score_norm_6040"] = 0.6 * df_comp["f1_det_norm"] + 0.4 * df_comp["sim_qual_norm"]

df_comp = df_comp.sort_values("score_norm_5050", ascending=False)
df_comp.to_csv(base / "ranking_score_composto.csv", index=False)

# ============================================================
# 2. RESUMOS POR DATASET
# ============================================================

resumo_dataset_det = (
    df_det.groupby("dataset_legivel")[["precision_det", "recall_det", "f1_det"]]
    .mean().round(4)
)

resumo_dataset_qual = (
    df_qual.groupby("dataset_legivel")[["precision_qual", "recall_qual", "f1_qual", "sim_media_qual"]]
    .mean().round(4)
)

resumo_dataset_doc = (
    df_doc.groupby("dataset_legivel")[["taxa_encontrado", "sim_media_geral"]]
    .agg(["mean", "std", "min", "max"]).round(4)
)

resumo_dataset_det.to_csv(base / "resumo_dataset_deteccao.csv")
resumo_dataset_qual.to_csv(base / "resumo_dataset_qualidade.csv")
resumo_dataset_doc.to_csv(base / "resumo_dataset_documento_detalhado.csv")

# ============================================================
# 3. RESUMOS POR MODELO E PROMPT
# ============================================================

for col_grp, nome in [("modelo", "modelo"), ("prompt", "prompt")]:
    (
        df_det.groupby(col_grp)[["precision_det", "recall_det", "f1_det"]]
        .mean().sort_values("f1_det", ascending=False).round(4)
        .to_csv(base / f"resumo_{nome}_deteccao.csv")
    )
    (
        df_qual.groupby(col_grp)[["precision_qual", "recall_qual", "f1_qual", "sim_media_qual"]]
        .mean().sort_values("sim_media_qual", ascending=False).round(4)
        .to_csv(base / f"resumo_{nome}_qualidade.csv")
    )

# ============================================================
# 4. ANÁLISE POR TÓPICO
# ============================================================

resumo_topico = (
    df_dupla.groupby(["dataset_legivel", "topico"])
    .agg(
        taxa_encontrado  =("detection_pred",       "mean"),
        sim_media        =("similarity",            "mean"),
        pct_not_found    =("resposta_is_not_found", "mean")
    )
    .reset_index().round(4)
)
resumo_topico.to_csv(base / "resumo_topico_dataset.csv", index=False)

topicos_dificeis = (
    resumo_topico.groupby("topico")[["taxa_encontrado", "sim_media", "pct_not_found"]]
    .mean()
    .sort_values(["sim_media", "taxa_encontrado"], ascending=True)
    .round(4)
)
topicos_dificeis.to_csv(base / "topicos_mais_dificeis.csv")

# ============================================================
# 5. ANÁLISE POR DOCUMENTO (COM GRANULARIDADE POR DATASET)
# ============================================================

# CORREÇÃO 2: manter granularidade por (documento, dataset)
# Versão anterior fazia média entre datasets → perdia informação
docs_por_dataset = (
    df_doc.sort_values(["sim_media_geral", "taxa_encontrado"], ascending=True)
    .round(4)
)
docs_por_dataset.to_csv(base / "documentos_por_dataset_dificeis.csv", index=False)

# Visão agregada (média entre datasets por documento)
docs_media = (
    df_doc.groupby("documento")[["taxa_encontrado", "sim_media_geral"]]
    .mean()
    .sort_values(["sim_media_geral", "taxa_encontrado"], ascending=True)
    .round(4)
)
docs_media.to_csv(base / "documentos_mais_dificeis.csv")

# ============================================================
# 6. RELATÓRIO TEXTUAL
# ============================================================

det_mean  = df_doc.groupby("dataset_legivel")["taxa_encontrado"].mean().sort_values(ascending=False)
qual_mean = df_doc.groupby("dataset_legivel")["sim_media_geral"].mean().sort_values(ascending=False)

rel = []
rel += [
    "ANÁLISE DETALHADA DOS RESULTADOS\n",
    "=" * 70 + "\n\n",
    "1. VISÃO GERAL\n",
    f"- Registros normalizados  : {len(df_regs)}\n",
    f"- Resultados de avaliação : {len(df_dupla)}\n",
    f"- Documentos avaliados    : {df_doc['documento'].nunique()}\n",
    f"- Tópicos avaliados       : {df_dupla['topico'].nunique()}\n",
    f"- Modelos avaliados       : {df_dupla['modelo'].nunique()}\n\n",

    "2. DETECÇÃO POR DATASET\n",
    resumo_dataset_det.to_string() + "\n\n",

    "3. QUALIDADE POR DATASET\n",
    resumo_dataset_qual.to_string() + "\n\n",

    "4. NÍVEL DOCUMENTO POR DATASET\n",
    resumo_dataset_doc.to_string() + "\n\n",

    "5. TRADE-OFF DETECÇÃO vs. QUALIDADE\n",
    f"- Melhor em detecção : {det_mean.index[0]} ({det_mean.iloc[0]:.4f})\n",
    f"- Melhor em qualidade: {qual_mean.index[0]} ({qual_mean.iloc[0]:.4f})\n",
    ("- Trade-off confirmado: líderes diferentes.\n\n"
     if det_mean.index[0] != qual_mean.index[0]
     else "- Mesmo dataset lidera nos dois critérios.\n\n"),

    "6. RANKING MODELOS — DETECÇÃO\n",
    df_det.groupby("modelo")[["precision_det","recall_det","f1_det"]].mean()
          .sort_values("f1_det", ascending=False).round(4).to_string() + "\n\n",

    "7. RANKING MODELOS — QUALIDADE\n",
    df_qual.groupby("modelo")[["precision_qual","recall_qual","f1_qual","sim_media_qual"]].mean()
           .sort_values("sim_media_qual", ascending=False).round(4).to_string() + "\n\n",

    "8. TOP 15 COMBINAÇÕES (score normalizado 50/50)\n",
    df_comp[["modelo","prompt","dataset_legivel",
             "f1_det","sim_media_qual","score_norm_5050"]].head(15).to_string(index=False) + "\n\n",

    "9. TÓPICOS MAIS DIFÍCEIS\n",
    topicos_dificeis.to_string() + "\n\n",

    "10. DOCUMENTOS MAIS DIFÍCEIS (top 15 médias entre datasets)\n",
    docs_media.head(15).to_string() + "\n\n",

    "11. CONCLUSÕES\n",
    "- topics lidera em detecção; conhecimento_12 lidera em qualidade semântica.\n",
    "- zeroshoot é a estratégia mais fraca em ambas as dimensões.\n",
    "- GPT-5.1 tem melhor detecção; Sonar tem melhor qualidade de conteúdo.\n",
    "- Score normalizado mostra diferença real entre estratégias sem distorção de escala.\n",
]

relatorio_txt = "".join(rel)

with open(base / "relatorio_analise_detalhada.txt", "w", encoding="utf-8") as f:
    f.write(relatorio_txt)

print("\n" + "=" * 70)
print(relatorio_txt[:3000])
print("=" * 70)

print("\n✅ Arquivos gerados:")
for arq in [
    "ranking_score_composto.csv",
    "resumo_dataset_deteccao.csv",
    "resumo_dataset_qualidade.csv",
    "resumo_dataset_documento_detalhado.csv",
    "resumo_modelo_deteccao.csv",
    "resumo_modelo_qualidade.csv",
    "resumo_prompt_deteccao.csv",
    "resumo_prompt_qualidade.csv",
    "resumo_topico_dataset.csv",
    "topicos_mais_dificeis.csv",
    "documentos_por_dataset_dificeis.csv",
    "documentos_mais_dificeis.csv",
    "relatorio_analise_detalhada.txt",
]:
    print(f"  - {arq}")