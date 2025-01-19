import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Configurações para o drone
CAPACIDADE_COMBUSTIVEL = 500
CAPACIDADE_ESTOQUE = 4

# Função para criar pontos aleatórios em 3D
def gerar_pontos(n, limite):
    return np.random.rand(n, 3) * limite

# Função para calcular a distância entre dois pontos
def distancia(p1, p2):
    return np.linalg.norm(p1 - p2)

# Criação dos pontos de origem, entrega e recarga
ponto_origem = np.array([0, 0, 0])
pontos_entrega = gerar_pontos(30, 100)
pontos_recarga = gerar_pontos(3, 100)

# Função para o algoritmo genético completo
def algoritmo_genetico(pontos, ponto_inicial, tamanho_populacao, geracoes, taxa_mutacao):
    num_pontos = len(pontos)
    populacao = [random.sample(range(num_pontos), num_pontos) for _ in range(tamanho_populacao)]
    melhor_rota = None
    melhor_distancia = float('inf')

    for _ in range(geracoes):
        distancias = [distancia_total(rota, pontos, ponto_inicial) for rota in populacao]
        nova_populacao = selecao(populacao, distancias, tamanho_populacao // 2)
        
        while len(nova_populacao) < tamanho_populacao:
            parentes = random.sample(nova_populacao, 2)
            filho = mutacao(crossover(parentes[0], parentes[1]), taxa_mutacao)
            nova_populacao.append(filho)
        
        populacao = nova_populacao
        menor_distancia = min(distancias)
        
        if menor_distancia < melhor_distancia:
            melhor_distancia = menor_distancia
            melhor_rota = populacao[distancias.index(menor_distancia)]

    return melhor_rota

def distancia_total(rota, pontos, ponto_inicial):
    dist = distancia(ponto_inicial, pontos[rota[0]])
    for i in range(len(rota) - 1):
        dist += distancia(pontos[rota[i]], pontos[rota[i + 1]])
    dist += distancia(pontos[rota[-1]], ponto_inicial)
    return dist

def selecao(populacao, distancias, num_selecao):
    selecao = np.argsort(distancias)[:num_selecao]
    return [populacao[i] for i in selecao]

def crossover(rota1, rota2):
    tamanho = len(rota1)
    comeco, fim = sorted(random.sample(range(tamanho), 2))
    filho = [-1] * tamanho
    filho[comeco:fim] = rota1[comeco:fim]
    aux = fim
    for ponto in rota2:
        if ponto not in filho:
            if aux >= tamanho:
                aux = 0
            filho[aux] = ponto
            aux += 1
    return filho

def mutacao(rota, taxa_mutacao=0.1):
    if random.random() < taxa_mutacao:
        i, j = random.sample(range(len(rota)), 2)
        rota[i], rota[j] = rota[j], rota[i]
    return rota

# Função para aplicar limitações a cada sub-rota
def ajustar_subrotas_para_limitacoes(rota, n_drones):
    subrotas = np.array_split(rota, n_drones)
    rotas_com_limitacoes = []

    for subrota in subrotas:
        combustivel = CAPACIDADE_COMBUSTIVEL
        estoque = CAPACIDADE_ESTOQUE
        posicao_atual = ponto_origem
        subrota_limitada = [ponto_origem]
        pontos_recarga_visitados = set()

        for ponto_idx in subrota:
            proximo_ponto = pontos_entrega[ponto_idx]
            distancia_para_proximo = distancia(posicao_atual, proximo_ponto)
            distancia_para_origem = distancia(posicao_atual, ponto_origem)

            # Verificação se é possível retornar ao ponto de origem
            if combustivel < distancia_para_origem:
                print("Combustível insuficiente para garantir retorno ao ponto de origem.")
                subrota_limitada.append(ponto_origem)
                break

            # Verificar limitações de combustível e estoque
            if combustivel < distancia_para_proximo or estoque <= 0:
                pontos_recarga_disponiveis = [p for p in pontos_recarga if tuple(p) not in pontos_recarga_visitados]
                
                if pontos_recarga_disponiveis:
                    ponto_recarga_mais_proximo = min(pontos_recarga_disponiveis, key=lambda p: distancia(posicao_atual, p))
                    distancia_para_recarga = distancia(posicao_atual, ponto_recarga_mais_proximo)
                    
                    if combustivel >= distancia_para_recarga:
                        subrota_limitada.append(ponto_recarga_mais_proximo)
                        combustivel = CAPACIDADE_COMBUSTIVEL
                        estoque = CAPACIDADE_ESTOQUE
                        posicao_atual = ponto_recarga_mais_proximo
                        pontos_recarga_visitados.add(tuple(ponto_recarga_mais_proximo))
                        print(f"Reabastecendo no ponto de recarga em {ponto_recarga_mais_proximo}.")
                    else:
                        subrota_limitada.append(ponto_origem)
                        print("Retorno ao ponto de origem por falta de combustível.")
                        break
                else:
                    print("Todos os pontos de recarga já foram visitados. Retornando ao ponto de origem.")
                    subrota_limitada.append(ponto_origem)
                    break

            # Continuar para o próximo ponto de entrega
            combustivel -= distancia_para_proximo
            estoque -= 1
            subrota_limitada.append(proximo_ponto)
            posicao_atual = proximo_ponto

        subrota_limitada.append(ponto_origem)
        rotas_com_limitacoes.append(subrota_limitada)

    return rotas_com_limitacoes

# Plotar rotas no gráfico 3D
def plotar_rotas(rotas):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*ponto_origem, color='green', s=100, label='Origem', marker='o')
    ax.scatter(pontos_entrega[:, 0], pontos_entrega[:, 1], pontos_entrega[:, 2], color='blue', s=50, label='Pontos de Entrega', marker='o')
    ax.scatter(pontos_recarga[:, 0], pontos_recarga[:, 1], pontos_recarga[:, 2], color='red', s=100, label='Pontos de Recarga', marker='o')
    
    for rota in rotas:
        rota = np.array(rota)
        ax.plot(rota[:, 0], rota[:, 1], rota[:, 2], label='Sub-rota')
    
    ax.legend()
    plt.show()

# Executar o algoritmo genético e ajustar sub-rotas para limitações
rota_inicial = algoritmo_genetico(pontos_entrega, ponto_origem, 20, 100, 0.1)
rotas_com_limitacoes = ajustar_subrotas_para_limitacoes(rota_inicial, n_drones=3)
plotar_rotas(rotas_com_limitacoes)
