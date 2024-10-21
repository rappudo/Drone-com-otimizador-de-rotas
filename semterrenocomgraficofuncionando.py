import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from scipy.spatial.distance import euclidean

# Função para calcular a distância total com recarga de combustível e materiais
def calcular_distancia_com_recarga_materiais(rota, pontos, ponto_inicial, pontos_recarga, distancia_maxima, materiais_maximos):
    distancia_total = euclidean(ponto_inicial, pontos[rota[0]])  # Do ponto inicial ao primeiro ponto
    pontos_visitados = 1
    autonomia = distancia_maxima  # Inicialmente o drone tem a autonomia máxima
    materiais_restantes = materiais_maximos  # Inicialmente o drone tem o máximo de materiais
    
    for i in range(len(rota) - 1):
        proxima_distancia = euclidean(pontos[rota[i]], pontos[rota[i + 1]])
        
        # Verifica se o próximo ponto é uma estação de recarga
        if tuple(pontos[rota[i]]) in pontos_recarga:
            autonomia = distancia_maxima  # Recarrega combustível
            materiais_restantes = materiais_maximos  # Recarrega materiais

        if autonomia < proxima_distancia or materiais_restantes <= 0:
            # Se não for possível ir ao próximo ponto com a autonomia restante ou sem materiais, parar
            break
        
        # Deduz a distância da autonomia e consome um material
        autonomia -= proxima_distancia
        materiais_restantes -= 1
        distancia_total += proxima_distancia
        pontos_visitados += 1
    
    # Verificar se consegue voltar ao ponto inicial
    distancia_volta = euclidean(pontos[rota[pontos_visitados - 1]], ponto_inicial)
    if tuple(pontos[rota[pontos_visitados - 1]]) in pontos_recarga:
        autonomia = distancia_maxima  # Recarrega antes de voltar
        materiais_restantes = materiais_maximos  # Recarrega materiais também
    if autonomia >= distancia_volta:
        distancia_total += distancia_volta
    else:
        # Não foi possível voltar ao ponto inicial dentro do limite
        pontos_visitados -= 1
    
    return distancia_total, pontos_visitados

# Função para gerar a população inicial
def gerar_populacao_inicial(tamanho_populacao, n_pontos):
    populacao = []
    for _ in range(tamanho_populacao):
        rota = list(np.random.permutation(n_pontos))
        populacao.append(rota)
    return populacao

# Função de seleção - torneio
def selecao_torneio(populacao, aptidao, k=3):
    torneio = random.sample(list(zip(populacao, aptidao)), k)
    vencedor = max(torneio, key=lambda x: x[1])  # Maximizar o número de pontos visitados
    return vencedor[0]

# Função de crossover (ordem preservada)
def crossover(rota1, rota2):
    n = len(rota1)
    start, end = sorted(random.sample(range(n), 2))
    filho = [-1] * n
    filho[start:end+1] = rota1[start:end+1]
    
    ptr = 0
    for gene in rota2:
        if gene not in filho:
            while filho[ptr] != -1:
                ptr += 1
            filho[ptr] = gene
    return filho

# Função de mutação (swap de dois genes)
def mutacao(rota, taxa_mutacao=0.1):
    if random.random() < taxa_mutacao:
        i, j = random.sample(range(len(rota)), 2)
        rota[i], rota[j] = rota[j], rota[i]

# Função principal do algoritmo genético com recarga de combustível e materiais
def algoritmo_genetico_com_recarga_materiais(pontos, ponto_inicial, pontos_recarga, distancia_maxima, materiais_maximos, tamanho_populacao=100, n_geracoes=500, taxa_mutacao=0.1):
    n_pontos = len(pontos)
    
    # Gerar população inicial
    populacao = gerar_populacao_inicial(tamanho_populacao, n_pontos)
    
    # Loop das gerações
    for geracao in range(n_geracoes):
        # Avaliar a aptidão (quantidade de pontos visitados dentro da distância e materiais máximos) de cada rota
        aptidao = []
        for rota in populacao:
            _, pontos_visitados = calcular_distancia_com_recarga_materiais(rota, pontos, ponto_inicial, pontos_recarga, distancia_maxima, materiais_maximos)
            aptidao.append(pontos_visitados)
        
        # Nova população
        nova_populacao = []
        
        # Preservar o melhor indivíduo (elitismo)
        melhor_rota = populacao[np.argmax(aptidao)]
        nova_populacao.append(melhor_rota)
        
        # Gerar novos indivíduos por crossover e mutação
        for _ in range(tamanho_populacao - 1):
            # Seleção por torneio
            pai1 = selecao_torneio(populacao, aptidao)
            pai2 = selecao_torneio(populacao, aptidao)
            
            # Crossover
            filho = crossover(pai1, pai2)
            
            # Mutação
            mutacao(filho, taxa_mutacao)
            
            # Adicionar filho à nova população
            nova_populacao.append(filho)
        
        # Substituir a população pela nova
        populacao = nova_populacao
        
        # Melhor rota encontrada até o momento
        melhor_pontos_visitados = max(aptidao)
        print(f"Geração {geracao + 1}: Máximo de pontos visitados = {melhor_pontos_visitados}")
    
    # Avaliar a última geração e retornar a melhor rota
    aptidao_final = []
    for rota in populacao:
        distancia_total, pontos_visitados = calcular_distancia_com_recarga_materiais(rota, pontos, ponto_inicial, pontos_recarga, distancia_maxima, materiais_maximos)
        aptidao_final.append(pontos_visitados)
    
    melhor_rota_final = populacao[np.argmax(aptidao_final)]
    melhor_pontos_visitados = max(aptidao_final)
    melhor_distancia_final, _ = calcular_distancia_com_recarga_materiais(melhor_rota_final, pontos, ponto_inicial, pontos_recarga, distancia_maxima, materiais_maximos)
    
    return melhor_rota_final, melhor_pontos_visitados, melhor_distancia_final

# Exemplo de uso:
pontos = np.array([[25, 100,  41], [90,  97,  29], [31,  46,  53], [75,  93,  28], [26,  86,  20]])
ponto_inicial = np.array([28,68,57])  # Definir o ponto inicial
pontos_recarga = {(87,17,53), (75, 93, 28)}  # Definir os pontos de recarga
distancia_maxima = 1000  # Definir a distância máxima permitida
materiais_maximos = 10  # Definir o número máximo de entregas (materiais)

melhor_rota, pontos_visitados, distancia_total = algoritmo_genetico_com_recarga_materiais(pontos, ponto_inicial, pontos_recarga, distancia_maxima, materiais_maximos, tamanho_populacao=200, n_geracoes=100, taxa_mutacao=0.05)
rota = melhor_rota

print(f"Melhor rota encontrada: {melhor_rota}")
print(f"Quantidade de pontos visitados: {pontos_visitados}")
print(f"Distância total percorrida: {distancia_total:.2f}")


# Função para visualizar a rota do drone
def visualizar_rota(pontos, ponto_inicial, pontos_recarga, rota, titulo="Rota do Drone"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotar os pontos de entrega
    pontos_x = pontos[:, 0]
    pontos_y = pontos[:, 1]
    pontos_z = pontos[:, 2]
    ax.scatter(pontos_x, pontos_y, pontos_z, c='b', marker='o', label="Pontos de entrega")

    # Plotar o ponto inicial
    ax.scatter(ponto_inicial[0], ponto_inicial[1], ponto_inicial[2], c='g', marker='s', s=100, label="Ponto inicial")

    # Plotar os pontos de recarga
    recarga_x = [p[0] for p in pontos_recarga]
    recarga_y = [p[1] for p in pontos_recarga]
    recarga_z = [p[2] for p in pontos_recarga]
    ax.scatter(recarga_x, recarga_y, recarga_z, c='r', marker='^', s=100, label="Pontos de recarga")

    # Plotar a rota do drone
    rota_x = [ponto_inicial[0]] + [pontos[i][0] for i in rota] + [ponto_inicial[0]]
    rota_y = [ponto_inicial[1]] + [pontos[i][1] for i in rota] + [ponto_inicial[1]]
    rota_z = [ponto_inicial[2]] + [pontos[i][2] for i in rota] + [ponto_inicial[2]]
    ax.plot(rota_x, rota_y, rota_z, color='k', linestyle='-', label="Rota do drone")

    # Configurações do gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(titulo)
    ax.legend()

    plt.show()

# Chama a função para visualizar a rota
visualizar_rota(pontos, ponto_inicial, pontos_recarga, rota)
