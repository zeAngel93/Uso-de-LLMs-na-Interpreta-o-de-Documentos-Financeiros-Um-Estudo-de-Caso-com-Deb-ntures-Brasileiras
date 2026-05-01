# Uso de LLMs na Interpreta o de Documentos Financeiros: Um Estudo de Caso com Debêntures Brasileiras
Repositório com dados, prompts e scripts do estudo sobre uso de LLMs na extração de informações de 100 escrituras de debêntures brasileiras. Inclui os documentos, prompts, datasets gerados, resultados intermediários e scripts de avaliação e validação do silver standard, permitindo a reprodução completa do pipeline experimental.
Observação: os arquivos PDF das debêntures não estão incluídos neste repositório devido ao tamanho dos documentos. Em vez disso, os PDFs podem ser consultados por meio de um link público do Google Drive disponibilizado abaixo. O repositório contém apenas os materiais necessários para reproduzir os experimentos: scripts, prompts, datasets processados e resultados gerados. (https://drive.google.com/drive/folders/1v8rXeFe7gVbefWOg0pf9XB7-77VXbm_O?usp=sharing)
 Metodologia de uso:
  1) GERAÇÃO DO SILVER STANDARD (main.py). 
  2) IDENTIFICAÇÃO DE SILVERS PROBLEMÁTICOS (identificar_casos_dubios.py).
  3) APLICAÇÃO DE GABARITO MANUAL (aplicar_gabarito.py). 
  4) ANÁLISE DE RESULTADOS (analizar_resultados.py)

Os demais arquivos (.CSV)  são gerados pelos scripts. 

