# relatorio_fastgreedy.py

import os
import tempfile

import pandas as pd
import gower
import igraph as ig
import matplotlib.pyplot as plt
from fpdf import FPDF

# ==============================================
# Constantes
# ==============================================
SALARIO_MINIMO = 1518.0
SIMILARIDADE_LIMIAR = 0.5
EXCEL_PATH = "./Solicitantes de Auxílio Estudantil - 2018.xlsx"
SHEET_NAME = "2018"
OUTPUT_PDF = "relatorio_fastgreedy.pdf"
OUTPUT_CSV = "stats_fastgreedy.csv"


# ==============================================
# 1. Funções de cálculo de pontos
# ==============================================
def pontos_renda(r):
    """
    Retorna pontuação de vulnerabilidade com base na Renda per capita (r).
    """
    if pd.isna(r):
        return 0  # Considera zero se não houver informação
    if r < 0.5 * SALARIO_MINIMO:
        return 40
    elif r < 1.0 * SALARIO_MINIMO:
        return 30
    elif r <= 1.5 * SALARIO_MINIMO:
        return 20
    return 0


def pontos_moradia(m):
    """
    Retorna pontuação de vulnerabilidade com base na descrição da moradia (m).
    """
    if not isinstance(m, str):
        return 0
    m_lower = m.strip().lower()
    if "alugada" in m_lower:
        return 15
    elif "pagamento" in m_lower:
        return 12
    elif "quitada" in m_lower:
        return 5
    elif "herança" in m_lower or "heranca" in m_lower:
        return 2
    return 0


def pontos_despesas(dp, rp):
    """
    Retorna pontuação de vulnerabilidade com base na relação Despesas per capita (dp) / Renda per capita (rp).
    """
    if pd.isna(dp) or pd.isna(rp) or rp == 0:
        return 10
    rel = dp / rp
    if rel > 2.0:
        return 10
    elif rel > 1.5:
        return 7
    elif rel > 1.0:
        return 4
    return 0


def pontos_bens(v):
    """
    Retorna pontuação de vulnerabilidade com base no Valor Total dos bens familiares (v).
    """
    if pd.isna(v):
        return 0
    if v == 0:
        return 15
    elif v < 10000:
        return 10
    elif v < 100000:
        return 5
    return 0


# ==============================================
# 2. Carregar e preparar os dados
# ==============================================
def carregar_dados(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Lê o arquivo Excel na planilha especificada e retorna um DataFrame 
    contendo apenas as colunas de interesse, sem valores ausentes
    em colunas críticas.
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {excel_path}")
    except Exception as e:
        raise RuntimeError(f"Erro ao ler o Excel: {e}")

    # Seleção de colunas importantes
    colunas = [
        'Qual a situação da MORADIA DO GRUPO FAMILIAR?',
        'Qual o principal meio de transporte que você utiliza para vir até a Universidade?',
        'Renda per capita',
        'Despesas per capita',
        'Valor Total dos bens familiares'
    ]

    # Verifica se todas as colunas existem no DataFrame
    for col in colunas:
        if col not in df.columns:
            raise KeyError(f"Coluna esperada não encontrada: {col}")

    # Filtrar apenas as colunas escolhidas
    df = df[colunas].copy()

    # Remover espaços em branco de campos de texto
    df[colunas[0]] = df[colunas[0]].astype(str).str.strip()
    df[colunas[1]] = df[colunas[1]].astype(str).str.strip()

    # Verificar se há valores ausentes em colunas críticas
    colunas_criticas = ['Renda per capita', 'Despesas per capita', 'Valor Total dos bens familiares']
    linhas_invalidas = []
    for col in colunas_criticas:
        invalidos = df[df[col].isna()].index.tolist()
        linhas_invalidas.extend(invalidos)
    linhas_invalidas = sorted(set(linhas_invalidas))

    if linhas_invalidas:
        raise ValueError(
            f"Existem valores ausentes nas colunas críticas nas linhas: {linhas_invalidas}. "
            "Corrija esses valores antes de prosseguir."
        )

    return df


# ==============================================
# 3. Calcular Índice de Vulnerabilidade
# ==============================================
def calcular_indice_vulnerabilidade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acrescenta ao DataFrame uma coluna 'Indice Vulnerabilidade' calculada
    pela soma de pontos de renda, moradia, despesas e bens.
    """
    df = df.copy()
    df['Indice Vulnerabilidade'] = df.apply(
        lambda row: (
            pontos_renda(row['Renda per capita']) +
            pontos_moradia(row['Qual a situação da MORADIA DO GRUPO FAMILIAR?']) +
            pontos_despesas(row['Despesas per capita'], row['Renda per capita']) +
            pontos_bens(row['Valor Total dos bens familiares'])
        ), axis=1
    )
    return df


# ==============================================
# 4. Construir Grafo a partir da matriz de similaridade (Gower)
# ==============================================
def construir_grafo(df: pd.DataFrame, limiar: float = SIMILARIDADE_LIMIAR) -> (ig.Graph, list):
    """
    Gera um grafo não direcionado em que cada nó representa um estudante, 
    e há arestas apenas quando a similaridade (Gower) entre dois estudantes >= limiar.
    Retorna o grafo e a lista de pesos correspondentes às arestas.
    """
    # Remover coluna de Índice de Vulnerabilidade para cálculo de similaridade
    df_sim = df.drop(columns=['Indice Vulnerabilidade']).copy()

    # Calcular matriz de similaridade de Gower
    matriz_sim = gower.gower_matrix(df_sim.values)

    n = len(df_sim)
    edges = []
    weights = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = matriz_sim[i][j]
            if sim >= limiar:
                edges.append((i, j))
                weights.append(sim)

    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights
    # Rotular os vértices com seu índice
    g.vs['label'] = [str(i) for i in range(n)]

    return g, weights


# ==============================================
# 5. Executar Algoritmo Fast Greedy
# ==============================================
def rodar_fastgreedy(g: ig.Graph, weights: list) -> dict:
    """
    Executa o método Fast Greedy no grafo g, atribui cores conforme comunidades,
    salva a imagem do grafo e retorna um dicionário com informações:
    {
        'num_comunidades': int,
        'modularidade': float,
        'caminho_imagem': str,
        'membership': List[int]
    }
    """
    layout = g.layout("fruchterman_reingold")

    # --- Fast Greedy ---
    g_greedy = g.copy()
    dendro = g_greedy.community_fastgreedy(weights='weight')
    clusters_greedy = dendro.as_clustering()
    g_greedy.vs['community'] = clusters_greedy.membership
    palette_greedy = ig.drawing.colors.ClusterColoringPalette(len(clusters_greedy))
    g_greedy.vs['color'] = [palette_greedy[c] for c in clusters_greedy.membership]

    caminho_fgreedy = tempfile.mktemp(suffix="_fgreedy.png")
    ig.plot(
        g_greedy,
        layout=layout,
        vertex_size=20,
        edge_width=[0.1 + 2 * w for w in weights],
        bbox=(800, 800),
        margin=40,
        target=caminho_fgreedy
    )

    return {
        'num_comunidades': len(clusters_greedy),
        'modularidade': clusters_greedy.modularity,
        'caminho_imagem': caminho_fgreedy,
        'membership': clusters_greedy.membership
    }


# ==============================================
# 6. Calcular Estatísticas por Grupo Fast Greedy
# ==============================================
def calcular_estatisticas_por_grupo(df: pd.DataFrame, membership: list) -> (pd.DataFrame, pd.DataFrame):
    """
    Recebe o DataFrame original (com 'Indice Vulnerabilidade') e a lista 'membership' 
    (um inteiro por linha indicando a qual grupo cada estudante pertence).
    Devolve:
    - df_annotated: DataFrame com uma coluna adicional 'Grupo FastGreedy'.
    - stats_por_grupo: DataFrame contendo colunas ['Grupo', 'Media Vulnerabilidade', 'Desvio Padrão', 'N de Estudantes'].
    """
    df_annotated = df.copy()
    df_annotated['Grupo FastGreedy'] = membership

    agrup = df_annotated.groupby('Grupo FastGreedy')['Indice Vulnerabilidade']
    stats = agrup.agg(mean='mean', std='std', count='count').reset_index()
    stats.columns = ['Grupo', 'Media Vulnerabilidade', 'Desvio Padrão', 'N de Estudantes']

    return df_annotated, stats


# ==============================================
# 7. Gerar Histograma de Médias por Grupo Fast Greedy
# ==============================================
def gerar_histograma_medias_por_grupo(stats_por_grupo: pd.DataFrame) -> str:
    """
    Gera um gráfico de barras mostrando a 'Media Vulnerabilidade' para cada grupo
    e retorna o caminho do arquivo PNG gerado.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        stats_por_grupo['Grupo'].astype(str),
        stats_por_grupo['Media Vulnerabilidade']
    )
    ax.set_xlabel("Grupo Fast Greedy")
    ax.set_ylabel("Média Índice Vulnerabilidade")
    ax.set_title("Média de Vulnerabilidade por Grupo (Fast Greedy)")
    plt.tight_layout()

    caminho_hist = tempfile.mktemp(suffix="_histogr.png")
    fig.savefig(caminho_hist)
    plt.close(fig)
    return caminho_hist


# ==============================================
# 8. Classe para montagem do PDF (FPDF)
# ==============================================
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Análise de Vulnerabilidade Estudantil - 2018 (Fast Greedy)", ln=True, align='C')
        self.ln(4)

    def chapter_title(self, title: str):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, text: str):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_image(self, image_path: str, title: str):
        """
        Insere um título de seção e em seguida a imagem (ajustada à largura do PDF).
        """
        self.chapter_title(title)
        if os.path.isfile(image_path):
            self.image(image_path, w=180)
            self.ln(4)
        else:
            self.chapter_body(f"Erro: imagem não encontrada em {image_path}")


# ==============================================
# 9. Gerar Relatório em PDF (Fast Greedy)
# ==============================================
def gerar_relatorio_pdf_fastgreedy(
    resultado_fg: dict,
    df_annotated: pd.DataFrame,
    stats_por_grupo: pd.DataFrame,
    caminho_hist: str,
    caminho_saida: str = OUTPUT_PDF
):
    """
    Recebe:
    - resultado_fg: Saída de rodar_fastgreedy()
    - df_annotated: DataFrame com coluna 'Indice Vulnerabilidade' e 'Grupo FastGreedy'
    - stats_por_grupo: DataFrame de estatísticas por grupo Fast Greedy
    - caminho_hist: Caminho do histograma de médias por grupo
    - caminho_saida: Nome do arquivo PDF a ser gerado.
    """
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # -- Seção 1: Resumo do Algoritmo Fast Greedy --
    pdf.chapter_title("1. Resumo do Algoritmo Fast Greedy")
    fg = resultado_fg
    texto_resumo = (
        f"Algoritmo Fast Greedy:\n"
        f"  - Número de Comunidades: {fg['num_comunidades']}\n"
        f"  - Modularidade: {fg['modularidade']:.4f}\n"
    )
    pdf.chapter_body(texto_resumo)

    # -- Seção 2: Gráfico Fast Greedy --
    pdf.add_image(fg['caminho_imagem'], "2. Grafo - Fast Greedy")

    # -- Seção 3: Histograma de Médias por Grupo Fast Greedy --
    pdf.add_image(caminho_hist, "3. Média de Vulnerabilidade por Grupo (Fast Greedy)")

    # -- Seção 4: Tabela detalhada de estudantes --
    pdf.chapter_title("4. Tabela de Índice de Vulnerabilidade por Estudante (com Grupo FastGreedy)")
    pdf.chapter_body("A seguir, listam-se todos os estudantes com seu índice e o grupo Fast Greedy correspondente:")

    tabela = df_annotated.copy()
    tabela = tabela[['Indice Vulnerabilidade', 'Grupo FastGreedy']].reset_index()
    tabela = tabela.rename(columns={'index': 'ID', 'Indice Vulnerabilidade': 'Índice', 'Grupo FastGreedy': 'Grupo'})
    tabela = tabela.sort_values(by='Índice', ascending=False).reset_index(drop=True)

    pdf.set_font("Courier", "B", 10)
    pdf.cell(30, 6, "Posição", border=1, align='C')
    pdf.cell(20, 6, "ID", border=1, align='C')
    pdf.cell(40, 6, "Índice", border=1, align='C')
    pdf.cell(40, 6, "Grupo", border=1, align='C')
    pdf.ln()
    pdf.set_font("Courier", "", 10)

    top10 = tabela.head(10)
    for posicao, row in top10.iterrows():
        pdf.cell(30, 6, str(posicao + 1), border=1, align='C')
        pdf.cell(20, 6, str(int(row['ID'])), border=1, align='C')
        pdf.cell(40, 6, f"{row['Índice']:.0f}", border=1, align='C')
        pdf.cell(40, 6, str(int(row['Grupo'])), border=1, align='C')
        pdf.ln()

    # -- Seção 5: Estatísticas por Grupo Fast Greedy --
    pdf.add_page()
    pdf.chapter_title("5. Estatísticas por Grupo (Fast Greedy)")
    for _, linha in stats_por_grupo.iterrows():
        texto_estat = (
            f"Grupo {int(linha['Grupo'])}:\n"
            f"  - Média de Vulnerabilidade: {linha['Media Vulnerabilidade']:.2f}\n"
            f"  - Desvio Padrão: {linha['Desvio Padrão']:.2f}\n"
            f"  - Número de Estudantes: {int(linha['N de Estudantes'])}\n"
        )
        pdf.chapter_body(texto_estat)

    # Salvar PDF
    pdf.output(caminho_saida)

    # Remover arquivos temporários
    try:
        os.remove(fg['caminho_imagem'])
        os.remove(caminho_hist)
    except OSError:
        pass

    print(f"Relatório Fast Greedy gerado com sucesso em: {caminho_saida}")


# ==============================================
# 10. Fluxo principal
# ==============================================
if __name__ == "__main__":
    # 10.1. Carregar dados
    try:
        df_raw = carregar_dados(EXCEL_PATH, SHEET_NAME)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        exit(1)

    # 10.2. Calcular índice de vulnerabilidade
    df_vul = calcular_indice_vulnerabilidade(df_raw)

    # 10.3. Construir grafo de similaridade
    grafo, pesos = construir_grafo(df_vul)

    # 10.4. Rodar algoritmo Fast Greedy
    resultado_fg = rodar_fastgreedy(grafo, pesos)

    # 10.5. Calcular estatísticas por grupo (Fast Greedy)
    df_annotated, stats_fg = calcular_estatisticas_por_grupo(
        df_vul, resultado_fg['membership']
    )

    # 10.6. Exportar estatísticas para CSV
    stats_fg.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Arquivo CSV com estatísticas Fast Greedy salvo em: {OUTPUT_CSV}")

    # 10.7. Gerar histograma de médias por grupo (Fast Greedy)
    caminho_hist = gerar_histograma_medias_por_grupo(stats_fg)

    # 10.8. Gerar relatório em PDF (Fast Greedy)
    gerar_relatorio_pdf_fastgreedy(
        resultado_fg,
        df_annotated,
        stats_fg,
        caminho_hist,
        OUTPUT_PDF
    )
