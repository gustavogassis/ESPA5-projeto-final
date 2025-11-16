import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2

# Configuração para evitar truncamento na exibição
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 1. Funções Auxiliares ---

# Função de conversão Web Mercator (X, Y) para Lat/Lon (WGS84)
def mercator_to_wgs84(x, y):
    # Raio da Terra (em metros)
    R = 6378137
    # Conversão de X para Longitude
    lon = np.degrees(x / R)
    # Conversão de Y para Latitude
    lat = np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi/2)
    return lat, lon

# Função para calcular a distância Haversine (em km) entre dois pontos (Lat/Lon)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Raio da Terra em km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# --- 2. Carregamento e Pré-processamento de Dados ---

# 2.1. Carregar Places_of_Worship.csv (Locais de Culto)
try:
    df_worship = pd.read_csv('/home/ubuntu/upload/Places_of_Worship.csv', encoding='utf-8')
    df_worship.columns = df_worship.columns.str.replace('^\\ufeff', '', regex=True)
    df_worship.rename(columns={'NAME': 'NOME', 'RELIGION': 'RELIGIAO', 'ADDRESS': 'ENDERECO', 'PLACE_OF_WORSHIP': 'TIPO_LOCAL'}, inplace=True)
    
    # Filtrar apenas as colunas de coordenadas (X, Y) e Religião
    df_worship = df_worship[['X', 'Y', 'RELIGIAO', 'TIPO_LOCAL']].copy()
    df_worship.dropna(subset=['X', 'Y'], inplace=True)
    df_worship['RELIGIAO'].fillna('NAO_INFORMADA', inplace=True)
    
    # Agrupamento de religiões (simplificado)
    def agrupar_religiao(religiao):
        religiao = str(religiao).upper().strip()
        if 'CHRISTIAN' in religiao or 'CATHOLIC' in religiao or 'PROTESTANT' in religiao or 'BAPTIST' in religiao or 'EPISCOPAL' in religiao or 'METHODIST' in religiao or 'PRESBYTERIAN' in religiao or 'LUTHERAN' in religiao or 'PENTECOSTAL' in religiao:
            return 'CRISTIANISMO'
        elif 'JEWISH' in religiao or 'JUDAISM' in religiao:
            return 'JUDAISMO'
        elif 'ISLAMIC' in religiao or 'MUSLIM' in religiao:
            return 'ISLAMISMO'
        else:
            return 'OUTRAS'
    
    df_worship['RELIGIAO_AGRUPADA'] = df_worship['RELIGIAO'].apply(agrupar_religiao)
    
    # Converter coordenadas Web Mercator para Lat/Lon
    df_worship['LAT'], df_worship['LON'] = mercator_to_wgs84(df_worship['X'], df_worship['Y'])
    
    print(f"Locais de Culto carregados: {len(df_worship)} registros.")
except Exception as e:
    print(f"Erro ao carregar Places_of_Worship.csv: {e}")
    exit()

# 2.2. Carregar DC_Demographics.csv (Dados Demográficos - ACS DP05)
try:
    df_demo = pd.read_csv('/home/ubuntu/DC_Demographics.csv', encoding='utf-8')
    df_demo.columns = df_demo.columns.str.replace('^\\ufeff', '', regex=True)
    
    # Selecionar colunas relevantes
    colunas_selecionadas = ['GEOID', 'INTPTLAT', 'INTPTLON', 'DP05_0001E', 'DP05_0071E', 'DP05_0072E', 'DP05_0074E', 'DP05_0070E', 'DP05_0018E']
    df_demo = df_demo[colunas_selecionadas].copy()
    
    # Renomear colunas
    df_demo.rename(columns={
        'DP05_0001E': 'POP_TOTAL',
        'DP05_0018E': 'POP_MAIOR_18',
        'DP05_0071E': 'POP_BRANCA',
        'DP05_0072E': 'POP_NEGRA',
        'DP05_0074E': 'POP_ASIATICA',
        'DP05_0070E': 'POP_HISPANA'
    }, inplace=True)
    
    # Limpeza de dados
    df_demo.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_demo.dropna(subset=['POP_TOTAL'], inplace=True)
    df_demo = df_demo[df_demo['POP_TOTAL'] > 0]
    
    print(f"Dados Demográficos carregados: {len(df_demo)} registros (Census Tracts).")
except Exception as e:
    print(f"Erro ao carregar DC_Demographics.csv: {e}")
    exit()

# 2.3. Mapeamento de Locais de Culto para Census Tracts (Nearest Neighbor)
print("\n--- Mapeamento de Locais de Culto para Census Tracts (Nearest Neighbor) ---")

tract_coords = df_demo[['INTPTLAT', 'INTPTLON', 'GEOID']].values
geoid_proximo = []
distancia_km = []

for i in range(len(df_worship)):
    lat_worship = df_worship.iloc[i]['LAT']
    lon_worship = df_worship.iloc[i]['LON']
    
    distances = [haversine(lat_worship, lon_worship, tract[0], tract[1]) for tract in tract_coords]
    
    min_dist_index = np.argmin(distances)
    
    geoid_proximo.append(tract_coords[min_dist_index][2])
    distancia_km.append(distances[min_dist_index])

df_worship['GEOID_PROXIMO'] = geoid_proximo
df_worship['DISTANCIA_KM'] = distancia_km

print("Mapeamento concluído.")

# 2.4. Agregação e Junção dos Dados
culto_por_tract = df_worship.groupby('GEOID_PROXIMO').size().reset_index(name='CONTAGEM_CULTOS')

df_analise = df_demo.merge(culto_por_tract, left_on='GEOID', right_on='GEOID_PROXIMO', how='left')
df_analise['CONTAGEM_CULTOS'].fillna(0, inplace=True)

# --- 3. Análise de Correlação (Correlação Positiva Forte) ---

# Correlação entre CONTAGEM_CULTOS e POP_TOTAL
print("\n--- Análise de Correlação de Pearson (Contagem vs População) ---")

df_corr = df_analise.dropna(subset=['CONTAGEM_CULTOS', 'POP_TOTAL'])
r_pop_culto = np.corrcoef(df_corr['CONTAGEM_CULTOS'], df_corr['POP_TOTAL'])[0, 1]

print(f"Correlação entre Contagem de Cultos e População Total: r = {r_pop_culto:.4f}")

# --- 4. Análises Estatísticas Adicionais e Gráficos ---

# 4.1. Estatística Descritiva Adicional: Distribuição de Religiões (Gráfico 1)
print("\n--- Geração de Gráfico 1: Distribuição de Religiões ---")
plt.figure(figsize=(10, 6))
sns.countplot(y='RELIGIAO_AGRUPADA', data=df_worship, order=df_worship['RELIGIAO_AGRUPADA'].value_counts().index, palette='viridis')
plt.title('Figura 1: Distribuição de Locais de Culto por Religião Agrupada', fontsize=14)
plt.xlabel('Contagem', fontsize=12)
plt.ylabel('Religião Agrupada', fontsize=12)
plt.tight_layout()
plt.savefig('figura_1_distribuicao_religiao.png')
plt.close()

# 4.2. Probabilidade Adicional: Distribuição de Locais por Tipo (Gráfico 2 - BARRA)
print("\n--- Geração de Gráfico 2: Distribuição de Locais por Tipo (Barra) ---")
plt.figure(figsize=(10, 6))
tipo_counts = df_worship['TIPO_LOCAL'].value_counts()
sns.barplot(x=tipo_counts.values, y=tipo_counts.index, palette='magma')
plt.title('Figura 2: Distribuição de Locais de Culto por Tipo', fontsize=14)
plt.xlabel('Contagem', fontsize=12)
plt.ylabel('Tipo de Local de Culto', fontsize=12)
plt.tight_layout()
plt.savefig('figura_2_distribuicao_tipo.png')
plt.close()

# 4.3. Correlação Positiva (Gráfico 3)
print("\n--- Geração de Gráfico 3: Correlação Positiva (Contagem vs População) ---")
plt.figure(figsize=(10, 6))
sns.regplot(x='POP_TOTAL', y='CONTAGEM_CULTOS', data=df_analise, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title(f'Figura 3: Correlação (r={r_pop_culto:.2f}): Contagem de Cultos vs População Total', fontsize=14)
plt.xlabel('População Total do Census Tract', fontsize=12)
plt.ylabel('Contagem de Locais de Culto', fontsize=12)
plt.tight_layout()
plt.savefig('figura_3_correlacao_populacao.png')
plt.close()

# 4.4. Inferência Adicional: Teste t (População Total) (Gráfico 4)
print("\n--- Geração de Gráfico 4: Boxplot para Inferência (População) ---")

# Hipótese: A contagem média de locais de culto é diferente em Census Tracts com alta População Total (acima da mediana)
mediana_pop = df_analise['POP_TOTAL'].median()
grupo_alta_pop = df_analise[df_analise['POP_TOTAL'] > mediana_pop]['CONTAGEM_CULTOS'].dropna()
grupo_baixa_pop = df_analise[df_analise['POP_TOTAL'] <= mediana_pop]['CONTAGEM_CULTOS'].dropna()

# Cálculo do Teste t (Manual)
mean1 = grupo_alta_pop.mean()
mean2 = grupo_baixa_pop.mean()
std1 = grupo_alta_pop.std()
std2 = grupo_baixa_pop.std()
n1 = len(grupo_alta_pop)
n2 = len(grupo_baixa_pop)

sp2 = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
t_stat = (mean1 - mean2) / np.sqrt(sp2 * (1/n1 + 1/n2))
df_liberdade = n1 + n2 - 2

print(f"Teste t (População Total): t-statistic = {t_stat:.4f}, df = {df_liberdade}")

df_analise['GRUPO_POP'] = np.where(df_analise['POP_TOTAL'] > mediana_pop, 'Alta População', 'Baixa População')

plt.figure(figsize=(8, 6))
sns.boxplot(x='GRUPO_POP', y='CONTAGEM_CULTOS', data=df_analise)
plt.title('Figura 4: Contagem de Locais de Culto por Grupo de População Total', fontsize=14)
plt.xlabel('Grupo de População Total do Census Tract', fontsize=12)
plt.ylabel('Contagem de Locais de Culto', fontsize=12)
plt.tight_layout()
plt.savefig('figura_4_boxplot_contagem_culto_populacao.png')
plt.close()

# 4.5. Gráfico 5: Distribuição da Contagem de Locais de Culto
print("\n--- Geração de Gráfico 5: Histograma da Contagem de Locais de Culto ---")
plt.figure(figsize=(10, 6))
sns.histplot(df_analise['CONTAGEM_CULTOS'], bins=range(int(df_analise['CONTAGEM_CULTOS'].max()) + 2), kde=False, discrete=True)
plt.title('Figura 5: Distribuição de Frequência da Contagem de Locais de Culto por Census Tract', fontsize=14)
plt.xlabel('Contagem de Locais de Culto', fontsize=12)
plt.ylabel('Frequência (Número de Census Tracts)', fontsize=12)
plt.xticks(range(int(df_analise['CONTAGEM_CULTOS'].max()) + 1))
plt.tight_layout()
plt.savefig('figura_5_histograma_contagem.png')
plt.close()

# 4.6. Gráfico 6: Distribuição da População Total
print("\n--- Geração de Gráfico 6: Histograma da População Total ---")
plt.figure(figsize=(10, 6))
sns.histplot(df_analise['POP_TOTAL'], bins=20, kde=True)
plt.title('Figura 6: Distribuição de Frequência da População Total por Census Tract', fontsize=14)
plt.xlabel('População Total', fontsize=12)
plt.ylabel('Frequência (Número de Census Tracts)', fontsize=12)
plt.tight_layout()
plt.savefig('figura_6_histograma_populacao.png')
plt.close()

# Fim do script de análise
print("\nAnálise avançada de dados concluída.")
