
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================

NOT_FOUND_TOKENS = {
    "não encontrado",
    "nao encontrado",
    "não informado",
    "nao informado",
    "not found",
    "n/a",
    ""
}

SIMILARIDADE_MIN_CONSENSO = 0.80  # threshold para consenso entre LLMs no silver
SIMILARIDADE_THRESHOLD_TP = 0.85  # threshold para acerto binário de qualidade

# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

print("🔹 Carregando JSONs...")

with open('final_basic.json', 'r', encoding='utf-8-sig') as f:
    data_basic = json.load(f)
with open('final_geral.json', 'r', encoding='utf-8-sig') as f:
    data_geral = json.load(f)
with open('final_topics.json', 'r', encoding='utf-8-sig') as f:
    data_topics = json.load(f)

DATASETS_RAW = {
    'prompt_geral_0_shoot': ('zeroshoot', data_basic),
    'prompt_geral_conhecimento': ('conhecimento_12', data_geral),
    'prompt_geral_conhecimento_por_topic': ('topics', data_topics)
}

# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def normalizar_texto(v):
    if v is None:
        return ""
    return str(v).strip()

def is_not_found(v):
    return normalizar_texto(v).lower() in NOT_FOUND_TOKENS

def extrair_registros(data_list, dataset_legivel, dataset_raw_name):
    """
    Extrai um registro por (doc, tópico, modelo).
    - Ignora campos *_normalized (campo bruto vs. campo normalizado)
    - Trata vazio e None como "Não encontrado"
    - "Não encontrado" é resposta VÁLIDA (conta na avaliação de detecção)
    """
    registros = []
    for item in data_list:
        meta = item.get('extractions_meta', {})
        nome_doc = meta.get('nome_arquivo_anexo', 'unknown').strip()
        prompt = meta.get('prompt', dataset_raw_name)
        model = meta.get('llm_model', 'unknown')

        for campo, valor in item.items():
            if campo == 'extractions_meta':
                continue
            if campo.endswith('_normalized'):
                # Evitar contar DATA_DE_EMISSAO e DATA_DE_EMISSAO_normalized
                # como dois tópicos diferentes
                continue

            valor_str = normalizar_texto(valor)
            if valor_str == "":
                valor_str = "Não encontrado"

            registros.append({
                'documento': nome_doc,
                'topico': campo,
                'resposta': valor_str,
                'modelo': model,
                'prompt': prompt,
                'dataset_legivel': dataset_legivel,
                'dataset_raw': dataset_raw_name,
                'is_not_found': is_not_found(valor_str)
            })
    return registros

# ============================================================
# 3. NORMALIZAÇÃO DOS REGISTROS
# ============================================================

print("🔹 Normalizando registros...")
registros = []
for raw_name, (ds_legivel, data_list) in DATASETS_RAW.items():
    registros.extend(extrair_registros(data_list, ds_legivel, raw_name))

df_raw = pd.DataFrame(registros)
df_raw.to_csv('registros_normalizados.csv', index=False)
print(f"Total de registros: {len(df_raw)}")
print(df_raw[['dataset_legivel']].value_counts())

# ============================================================
# 4. SILVER STANDARD + AVALIAÇÃO DUAL
# ============================================================

print("🔹 Carregando modelo Sentence-BERT...")
sbert = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("🔹 Calculando silver standard e métricas duais...")
resultados = []

grupos = list(df_raw.groupby(['documento', 'topico']))

for (doc, topico), df_grupo in tqdm(grupos, desc="Grupos (doc x tópico)"):

    respostas = df_grupo['resposta'].tolist()
    flags_nf  = df_grupo['is_not_found'].tolist()

    nf_count  = sum(flags_nf)
    txt_count = len(flags_nf) - nf_count

    # CORREÇÃO 1: nf_count >= txt_count → silver = "Não encontrado"
    # Antes: apenas nf_count > txt_count
    # Motivo: em empate (ex: 2 NF vs 2 textuais), campo provavelmente não existe
    if nf_count >= txt_count:
        silver        = "Não encontrado"
        silver_is_nf  = True
        silver_emb    = None
        resps_txt_emb = {}  # sem embeddings textuais
    else:
        silver_is_nf        = False
        respostas_textuais  = [r for r, nf in zip(respostas, flags_nf) if not nf]
        indices_textuais    = [i for i, nf in enumerate(flags_nf) if not nf]

        # CORREÇÃO 2: calcular TODOS os embeddings textuais do grupo de uma vez
        embs_txt = sbert.encode(respostas_textuais, convert_to_tensor=True)

        if len(respostas_textuais) == 1:
            silver    = respostas_textuais[0]
            silver_emb = embs_txt[0].cpu().numpy()
        else:
            sim_matrix = util.cos_sim(embs_txt, embs_txt).cpu().numpy()
            max_consenso = -1
            silver_idx   = 0
            for i in range(len(respostas_textuais)):
                # Número de outras respostas com similaridade > threshold de consenso
                # Descontar a auto-similaridade (diagonal = 1.0)
                consenso = np.sum(sim_matrix[i] > SIMILARIDADE_MIN_CONSENSO) - 1
                if consenso > max_consenso:
                    max_consenso = consenso
                    silver_idx   = i
            silver     = respostas_textuais[silver_idx]
            silver_emb = embs_txt[silver_idx].cpu().numpy()

        # Guardar embeddings por índice para reutilizar abaixo
        resps_txt_emb = {
            idx_orig: embs_txt[k].cpu().numpy()
            for k, idx_orig in enumerate(indices_textuais)
        }

    # Avaliar cada resposta do grupo
    for local_idx, (_, row) in enumerate(df_grupo.iterrows()):
        resp       = row['resposta']
        resp_is_nf = row['is_not_found']

        # --- DETECÇÃO ---
        # detection_true = 1 se o campo EXISTE (silver tem conteúdo)
        # detection_pred = 1 se o modelo RESPONDEU com conteúdo
        detection_true = 0 if silver_is_nf else 1
        detection_pred = 0 if resp_is_nf  else 1

        # --- QUALIDADE ---
        # Aplicável somente quando silver tem conteúdo E resposta tem conteúdo
        quality_applicable = (not silver_is_nf) and (not resp_is_nf)

        if silver_is_nf and resp_is_nf:
            # Ambos concordam: campo não existe → similaridade máxima
            sim = 1.0
        elif silver_is_nf and not resp_is_nf:
            # Modelo "alucionou" conteúdo onde não devia → penalidade total
            sim = 0.0
        elif not silver_is_nf and resp_is_nf:
            # Modelo falhou em encontrar campo que existe → penalidade total
            sim = 0.0
        else:
            # CORREÇÃO 2: reutilizar embedding já calculado (não recalcular)
            orig_idx = list(df_grupo.index).index(row.name)
            if orig_idx in resps_txt_emb:
                resp_emb = resps_txt_emb[orig_idx]
            else:
                # fallback seguro (não deveria ocorrer)
                resp_emb = sbert.encode([resp])[0]
            sim = float(util.cos_sim(silver_emb, resp_emb)[0][0])

        quality_pred = 1 if (quality_applicable and sim >= SIMILARIDADE_THRESHOLD_TP) else 0

        # CORREÇÃO 3: usar -1 em vez de np.nan quando qualidade não é aplicável
        # Evita ambiguidade em análises futuras no CSV
        quality_true = 1 if quality_applicable else -1

        resultados.append({
            'documento'            : doc,
            'topico'               : topico,
            'modelo'               : row['modelo'],
            'prompt'               : row['prompt'],
            'dataset_legivel'      : row['dataset_legivel'],
            'dataset_raw'          : row['dataset_raw'],
            'resposta'             : resp,
            'silver_label'         : silver,
            'silver_is_not_found'  : silver_is_nf,
            'resposta_is_not_found': resp_is_nf,
            'detection_true'       : detection_true,
            'detection_pred'       : detection_pred,
            'quality_applicable'   : quality_applicable,
            'quality_true'         : quality_true,  # -1 quando não aplicável
            'quality_pred'         : quality_pred,
            'similarity'           : sim
        })

df_result = pd.DataFrame(resultados)
df_result.to_csv('resultados_avaliacao_dupla.csv', index=False)
print("✅ Salvo: resultados_avaliacao_dupla.csv")

# ============================================================
# 5. MÉTRICAS DE DETECÇÃO
# ============================================================

print("🔹 Calculando métricas de detecção...")
rows_det = []
for idx, g in df_result.groupby(['modelo', 'prompt', 'dataset_legivel']):
    p, r, f1, support = precision_recall_fscore_support(
        g['detection_true'], g['detection_pred'],
        average='binary', zero_division=0
    )
    rows_det.append({
        'modelo'        : idx[0],
        'prompt'        : idx[1],
        'dataset_legivel': idx[2],
        'precision_det' : round(p, 4),
        'recall_det'    : round(r, 4),
        'f1_det'        : round(f1, 4),
        'support_det'   : int(support) if support is not None else 0
    })

df_det = pd.DataFrame(rows_det)
df_det.to_csv('metricas_deteccao.csv', index=False)
print("✅ Salvo: metricas_deteccao.csv")

# ============================================================
# 6. MÉTRICAS DE QUALIDADE
# ============================================================

print("🔹 Calculando métricas de qualidade...")
rows_qual = []
# FILTRO: apenas linhas onde qualidade é aplicável (ambos têm conteúdo)
df_quality = df_result[df_result['quality_applicable'] == True].copy()

for idx, g in df_quality.groupby(['modelo', 'prompt', 'dataset_legivel']):
    # quality_true sempre = 1 aqui (filtramos quality_applicable = True)
    p, r, f1, support = precision_recall_fscore_support(
        g['quality_true'], g['quality_pred'],
        average='binary', zero_division=0
    )
    rows_qual.append({
        'modelo'          : idx[0],
        'prompt'          : idx[1],
        'dataset_legivel' : idx[2],
        'precision_qual'  : round(p, 4),
        'recall_qual'     : round(r, 4),
        'f1_qual'         : round(f1, 4),
        'support_qual'    : int(support) if support is not None else 0,
        'sim_media_qual'  : round(g['similarity'].mean(), 4)
    })

df_qual = pd.DataFrame(rows_qual)
df_qual.to_csv('metricas_qualidade.csv', index=False)
print("✅ Salvo: metricas_qualidade.csv")

# ============================================================
# 7. AGREGAÇÃO POR DOCUMENTO
# ============================================================

print("🔹 Agregando por documento...")
df_doc = (
    df_result.groupby(['dataset_legivel', 'documento'])
    .agg(
        taxa_encontrado=('detection_pred', 'mean'),
        sim_media_geral =('similarity',     'mean')
    )
    .reset_index()
)
df_doc.to_csv('resultados_documento_duplo.csv', index=False)
print("✅ Salvo: resultados_documento_duplo.csv")

# ============================================================
# 8. RELATÓRIOS GLOBAIS
# ============================================================

print("\n📋 Relatório global de detecção:")
print(classification_report(
    df_result['detection_true'],
    df_result['detection_pred'],
    zero_division=0
))

print("\n📋 Resumo por dataset (nível documento):")
print(
    df_doc.groupby('dataset_legivel')[['taxa_encontrado', 'sim_media_geral']]
    .agg(['mean', 'std', 'min', 'max'])
    .round(4)
)

print("\n✅ Arquivos gerados:")
for arq in [
    "registros_normalizados.csv",
    "resultados_avaliacao_dupla.csv",
    "metricas_deteccao.csv",
    "metricas_qualidade.csv",
    "resultados_documento_duplo.csv"
]:
    print(f"  - {arq}")