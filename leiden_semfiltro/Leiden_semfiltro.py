# relatorio_leiden_unifei.py

import os
import tempfile
import pandas as pd
import gower
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
from fpdf import FPDF
from unidecode import unidecode

SALARIO_MINIMO = 1518.0
SIMILARIDADE_LIMIAR = 0.5
EXCEL_PATH = "./Solicitantes de Auxílio Estudantil - 2018.xlsx"
SHEET_NAME = 0
OUTPUT_PDF = "relatorio_leiden.pdf"
OUTPUT_CSV = "stats_leiden.csv"

# ========================================
# 1. Funções de pontuação
# ========================================

def pontos_renda(r):
    if pd.isna(r):
        return 0
    if r <= 0.5 * SALARIO_MINIMO:
        return 40
    elif r <= 1.0 * SALARIO_MINIMO:
        return 30
    elif r <= 1.5 * SALARIO_MINIMO:
        return 20
    return 0

def pontos_moradia(moradia_aluno, moradia_familia):
    pontos = 0
    if isinstance(moradia_aluno, str) and unidecode(moradia_aluno.strip().lower()) in ("aluguel", "financiado", "financiamento"):
        pontos += 8
    if isinstance(moradia_familia, str) and unidecode(moradia_familia.strip().lower()) in ("aluguel", "financiado", "financiamento"):
        pontos += 7
    return min(pontos, 15)

def pontos_procedencia(escola):
    if not isinstance(escola, str):
        return 0
    e = unidecode(escola.strip().lower())
    if "publica" in e:
        return 10
    elif "bolsa" in e or "filantropica" in e:
        return 5
    return 0

def pontos_bens(v):
    if pd.isna(v):
        return 0
    if v == 0:
        return 15
    elif v <= 20000:
        return 10
    elif v <= 50000:
        return 5
    return 0

def pontos_savs(row):
    pontos = 0
    if row.get('quantidade de individuos com doenca grave no grupo familiar', 0) > 0:
        pontos += 5
    if row.get('quantos filhos o solicitante possui?', 0) > 0:
        pontos += 5
    transporte = unidecode(str(row.get('qual o principal meio de transporte que voce utiliza para vir ate a universidade?', '')).lower())
    if any(x in transporte for x in ['zona rural', 'cidade vizinha', 'intermunicipal']):
        pontos += 5
    if row.get('familiares com superior completo ou pos', 0) == 0:
        pontos += 5
    return min(pontos, 20)

# =============================================
# 2. Carregamento e preparação dos dados
# =============================================

def carregar_dados(excel_path: str, sheet_name) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # Normalizar cabeçalhos: sem acentos, minúsculos, sem espaços extras
    df.columns = [unidecode(col.strip().lower()) for col in df.columns]

    # Lista de colunas esperadas (normalizadas)
    colunas_obrigatorias = [
        'id_discente',
        'qual sua procedencia escolar?',
        'qual a situacao da moradia do aluno?',
        'qual a situacao da moradia do grupo familiar?',
        'quantos filhos o solicitante possui?',
        'renda per capita',
        'valor total dos bens familiares',
        'quantidade de individuos com doenca grave no grupo familiar',
        'familiares com superior completo ou pos',
        'qual o principal meio de transporte que voce utiliza para vir ate a universidade?'
    ]
    for col in colunas_obrigatorias:
        if col not in df.columns:
            raise KeyError(f"Coluna esperada não encontrada: {col}")

    # Converter campos numéricos
    df['renda per capita'] = pd.to_numeric(df['renda per capita'], errors='coerce').fillna(0)
    df['valor total dos bens familiares'] = pd.to_numeric(df['valor total dos bens familiares'], errors='coerce').fillna(0)
    df['quantos filhos o solicitante possui?'] = pd.to_numeric(df['quantos filhos o solicitante possui?'], errors='coerce').fillna(0)
    df['quantidade de individuos com doenca grave no grupo familiar'] = pd.to_numeric(df['quantidade de individuos com doenca grave no grupo familiar'], errors='coerce').fillna(0)
    df['familiares com superior completo ou pos'] = pd.to_numeric(df['familiares com superior completo ou pos'], errors='coerce').fillna(0)
    # Não convertemos transporte e moradia, pois são strings

    # Filtrar estudantes com renda > 1.5 SM
    df = df[df['renda per capita'] <= 1.5 * SALARIO_MINIMO]

    return df

# ========================================
# 3. Cálculo do Índice de Vulnerabilidade
# ========================================

def calcular_indice_vulnerabilidade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def pontuar(row):
        return (
            pontos_renda(row['renda per capita']) +
            pontos_moradia(row['qual a situacao da moradia do aluno?'], row['qual a situacao da moradia do grupo familiar?']) +
            pontos_procedencia(row['qual sua procedencia escolar?']) +
            pontos_bens(row['valor total dos bens familiares']) +
            pontos_savs(row)
        )
    df['indice vulnerabilidade'] = df.apply(pontuar, axis=1).clip(upper=100)
    return df

# ====================================
# 4. Construir Grafo e Rodar Leiden
# ====================================

def construir_grafo(df: pd.DataFrame, limiar: float = SIMILARIDADE_LIMIAR):
    df_sim = df.drop(columns=['indice vulnerabilidade'])
    # Para Gower, é melhor converter todas colunas para arrays numpy
    matriz_sim = gower.gower_matrix(df_sim.values)
    n = len(df_sim)
    edges, weights = [], []
    for i in range(n):
        for j in range(i+1, n):
            sim = matriz_sim[i][j]
            if sim >= limiar:
                edges.append((i, j))
                weights.append(sim)
    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights
    g.vs['label'] = [str(i) for i in range(n)]
    return g, weights

def rodar_leiden(g: ig.Graph, weights: list) -> dict:
    layout = g.layout("fruchterman_reingold")
    g_leiden = g.copy()
    partition = la.find_partition(g_leiden, la.ModularityVertexPartition, weights='weight')
    g_leiden.vs['community'] = partition.membership
    palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
    g_leiden.vs['color'] = [palette[c] for c in partition.membership]
    caminho_img = tempfile.mktemp(suffix="_leiden.png")
    ig.plot(
        g_leiden,
        layout=layout,
        vertex_size=20,
        edge_width=[0.1 + 2*w for w in weights],
        bbox=(800, 800),
        margin=40,
        target=caminho_img
    )
    return {
        'num_comunidades': len(partition),
        'modularidade': partition.modularity,
        'caminho_imagem': caminho_img,
        'membership': partition.membership
    }

# =========================================
# 5. Estatísticas, Histograma e Relatório
# =========================================

def calcular_estatisticas_por_grupo(df: pd.DataFrame, membership: list) -> (pd.DataFrame, pd.DataFrame):
    df_annot = df.copy()
    df_annot['grupo leiden'] = membership
    stats = df_annot.groupby('grupo leiden')['indice vulnerabilidade'].agg(['mean', 'std', 'count']).reset_index()
    stats.columns = ['grupo', 'media vulnerabilidade', 'desvio padrao', 'n de estudantes']
    return df_annot, stats

def gerar_histograma(stats: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(stats['grupo'].astype(str), stats['media vulnerabilidade'])
    ax.set_xlabel("Grupo Leiden")
    ax.set_ylabel("Média de Vulnerabilidade")
    ax.set_title("Média de Vulnerabilidade por Grupo (Leiden)")
    plt.tight_layout()
    caminho_hist = tempfile.mktemp(suffix="_histogr.png")
    fig.savefig(caminho_hist)
    plt.close(fig)
    return caminho_hist

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Análise de Vulnerabilidade Estudantil - UNIFEI (Leiden)", ln=True, align='C')
        self.ln(4)
    def chapter_title(self, title: str):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
    def chapter_body(self, text: str):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)
    def add_image(self, path: str, title: str):
        self.chapter_title(title)
        if os.path.isfile(path):
            self.image(path, w=180)
            self.ln(4)
        else:
            self.chapter_body(f"Erro ao carregar imagem: {path}")

def gerar_relatorio_pdf_leiden(resultado: dict, df_annot: pd.DataFrame, stats: pd.DataFrame, caminho_hist: str, caminho_saida: str = OUTPUT_PDF):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.chapter_title("1. Resumo do Algoritmo Leiden")
    texto = f"Número de Comunidades: {resultado['num_comunidades']}\nModularidade: {resultado['modularidade']:.4f}"
    pdf.chapter_body(texto)
    pdf.add_image(resultado['caminho_imagem'], "2. Grafo - Leiden")
    pdf.add_image(caminho_hist, "3. Histograma de Médias por Grupo")
    pdf.chapter_title("4. Top 10 Estudantes Mais Vulneráveis")
    top10 = df_annot[['id_discente', 'indice vulnerabilidade', 'grupo leiden']].sort_values(by='indice vulnerabilidade', ascending=False).head(10)
    pdf.set_font("Courier", "B", 10)
    pdf.cell(30, 6, "ID", border=1, align='C')
    pdf.cell(30, 6, "Índice", border=1, align='C')
    pdf.cell(30, 6, "Grupo", border=1, align='C')
    pdf.ln()
    pdf.set_font("Courier", "", 10)
    for _, row in top10.iterrows():
        pdf.cell(30, 6, str(row['id_discente']), border=1, align='C')
        pdf.cell(30, 6, f"{row['indice vulnerabilidade']:.0f}", border=1, align='C')
        pdf.cell(30, 6, str(row['grupo leiden']), border=1, align='C')
        pdf.ln()
    pdf.add_page()
    pdf.chapter_title("5. Estatísticas por Grupo Leiden")
    for _, row in stats.iterrows():
        texto = f"Grupo {int(row['grupo'])}:\n  - Média: {row['media vulnerabilidade']:.2f}\n  - Desvio: {row['desvio padrao']:.2f}\n  - N: {int(row['n de estudantes'])}\n"
        pdf.chapter_body(texto)
    pdf.output(caminho_saida)
    try:
        os.remove(resultado['caminho_imagem'])
        os.remove(caminho_hist)
    except Exception:
        pass
    print(f"PDF salvo em: {caminho_saida}")

# ====================================
# Execução principal
# ====================================
if __name__ == "__main__":
    df_raw = carregar_dados(EXCEL_PATH, SHEET_NAME)
    df_vul = calcular_indice_vulnerabilidade(df_raw)
    grafo, pesos = construir_grafo(df_vul)
    resultado = rodar_leiden(grafo, pesos)
    df_annot, stats = calcular_estatisticas_por_grupo(df_vul, resultado['membership'])
    stats.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    caminho_hist = gerar_histograma(stats)
    gerar_relatorio_pdf_leiden(resultado, df_annot, stats, caminho_hist, OUTPUT_PDF)
