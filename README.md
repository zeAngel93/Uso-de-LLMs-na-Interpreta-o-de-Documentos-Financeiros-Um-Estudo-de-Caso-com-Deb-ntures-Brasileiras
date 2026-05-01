# Uso de LLMs na Interpretação de Documentos Financeiros: Um Estudo de Caso com Debêntures Brasileiras

## Visão geral

Repositório com dados, prompts e scripts do estudo sobre o uso de LLMs na extração de informações de 100 escrituras de debêntures brasileiras.

O repositório inclui prompts, datasets gerados, resultados intermediários e scripts de avaliação e validação do *silver standard*, permitindo a reprodução completa do pipeline experimental.

## Acesso aos PDFs originais

> **Observação:** os arquivos PDF das debêntures não estão incluídos neste repositório devido ao tamanho dos documentos.  
> Os PDFs podem ser consultados por meio do link público do Google Drive abaixo:

[Documentos originais em PDF no Google Drive](https://drive.google.com/drive/folders/1v8rXeFe7gVbefWOg0pf9XB7-77VXbm_O?usp=sharing)

Este repositório contém apenas os materiais necessários para reproduzir os experimentos: scripts, prompts, datasets processados e resultados gerados.

## Estrutura do repositório

Este repositório reúne os scripts, arquivos intermediários, resultados finais e materiais de apoio utilizados no estudo sobre extração de informações de debêntures brasileiras com LLMs.

A organização abaixo foi pensada para facilitar a reprodutibilidade do pipeline experimental e a interpretação dos arquivos gerados.

## Pipeline resumido

```text
final_basic.json / final_geral.json / final_topics.json
                ↓
             main.py
                ↓
registros_normalizados.csv
resultados_avaliacao_dupla.csv
                ↓
   identificar_casos_dubios.py
                ↓
casos_para_revisao_humana.csv
silver_nf_inconsistentes.csv
                ↓
        aplicar_gabarito.py
                ↓
resultados_avaliacao_dupla_corrigido.csv
                ↓
      analizar_resultados.py
                ↓
métricas, resumos, rankings e relatório final
```

## Scripts principais

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Script principal do pipeline. Lê os arquivos JSON com as extrações dos LLMs, normaliza os registros, constrói o *silver standard* automático, calcula similaridades semânticas e gera os resultados iniciais de avaliação. |
| `identificar_casos_dubios.py` | Identifica os casos em que o *silver standard* automático é menos confiável, priorizando registros para revisão humana com base em empate, divergência semântica, baixa similaridade e outros critérios. |
| `aplicar_gabarito.py` | Aplica o gabarito manual aos casos revisados, recalcula as similaridades necessárias e gera a versão corrigida dos resultados para análise final. |
| `analizar_resultados.py` | Consolida os resultados finais e produz métricas agregadas por dataset, modelo, prompt, tópico e documento, além de rankings e relatórios textuais. |

## Arquivos de entrada

| Arquivo | Descrição |
|---------|-----------|
| `final_basic.json` | Saída bruta da estratégia *zero-shot*, em que todos os tópicos são extraídos sem conhecimento adicional de domínio. |
| `final_geral.json` | Saída bruta da estratégia com conhecimento dos 12 tópicos, em uma única chamada por documento. |
| `final_topics.json` | Saída bruta da estratégia com extração individual por tópico, usando prompts específicos com conhecimento e exemplos. |

## Arquivos de revisão humana

| Arquivo | Descrição |
|---------|-----------|
| `casos_para_revisao_humana.csv` | Lista priorizada de casos duvidosos do *silver standard* para inspeção manual. |
| `silver_nf_inconsistentes.csv` | Casos em que o *silver* foi marcado como “Não encontrado”, mas há inconsistência entre as respostas dos modelos, exigindo revisão e possível correção. |

## Arquivos intermediários e resultados brutos

| Arquivo | Descrição |
|---------|-----------|
| `registros_normalizados.csv` | Base tabular normalizada com um registro por documento, tópico, modelo e estratégia de prompt. |
| `resultados_avaliacao_dupla.csv` | Resultado principal da avaliação automática antes da correção manual, contendo *silver label*, similaridades e indicadores de detecção e qualidade. |
| `resultados_avaliacao_dupla_corrigido.csv` | Versão corrigida do arquivo de avaliação após aplicação do gabarito humano nos casos problemáticos. |
| `resultados_documento_duplo.csv` | Resultados agregados no nível de documento, úteis para comparação de desempenho entre debêntures e estratégias. |

## Métricas e arquivos de síntese

| Arquivo | Descrição |
|---------|-----------|
| `metricas_deteccao.csv` | Métricas de detecção, como precisão, recall e F1, agregadas por combinação de modelo, prompt e dataset. |
| `metricas_qualidade.csv` | Métricas de qualidade semântica das respostas textuais, incluindo similaridade média e indicadores derivados. |
| `ranking_score_composto.csv` | Ranking das combinações modelo × prompt × dataset com base em um score composto de detecção e qualidade. |
| `resumo_dataset_deteccao.csv` | Resumo agregado das métricas de detecção por dataset. |
| `resumo_dataset_qualidade.csv` | Resumo agregado das métricas de qualidade por dataset. |
| `resumo_dataset_documento_detalhado.csv` | Estatísticas detalhadas por documento e por dataset, incluindo cobertura e similaridade média. |
| `resumo_modelo_deteccao.csv` | Comparação agregada entre os modelos na tarefa de detecção. |
| `resumo_modelo_qualidade.csv` | Comparação agregada entre os modelos na tarefa de qualidade semântica. |
| `resumo_prompt_deteccao.csv` | Resumo das métricas de detecção por estratégia de prompt. |
| `resumo_prompt_qualidade.csv` | Resumo das métricas de qualidade por estratégia de prompt. |
| `resumo_topico_dataset.csv` | Resumo por tópico e dataset, útil para identificar campos mais fáceis e mais difíceis. |
| `topicos_mais_dificeis.csv` | Lista dos tópicos com pior desempenho médio no conjunto de experimentos. |

## Relatórios finais

| Arquivo | Descrição |
|---------|-----------|
| `relatorio_analise_detalhada.txt` | Relatório textual consolidado com os principais achados da análise, incluindo comparação entre estratégias de prompt, modelos e métricas de desempenho. |

## Sugestão de leitura dos arquivos

Para entender o repositório na ordem correta, recomenda-se seguir esta sequência:

1. `final_basic.json`, `final_geral.json` e `final_topics.json`
2. `main.py`
3. `registros_normalizados.csv`
4. `resultados_avaliacao_dupla.csv`
5. `identificar_casos_dubios.py`
6. `casos_para_revisao_humana.csv` e `silver_nf_inconsistentes.csv`
7. `aplicar_gabarito.py`
8. `resultados_avaliacao_dupla_corrigido.csv`
9. `analizar_resultados.py`
10. `metricas_*.csv`, `resumo_*.csv`, `ranking_score_composto.csv` e `relatorio_analise_detalhada.txt`
