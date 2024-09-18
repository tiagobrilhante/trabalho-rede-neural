import pygame
import numpy as np
import random
import sys

# Inicialização do Pygame
pygame.init()

# Carregamento da imagem da pista sem converter
try:
    track_surface = pygame.image.load('pista.png')
except FileNotFoundError:
    print("Erro: Arquivo 'pista.png' não encontrado na pasta do jogo.")
    sys.exit()

track_width, track_height = track_surface.get_size()

# Definir a altura da área de informações no topo
INFO_AREA_HEIGHT = 100  # Altura da área de informações no topo

# Ajustar o tamanho da janela para incluir a área de informações
SCREEN_WIDTH = track_width
SCREEN_HEIGHT = track_height + INFO_AREA_HEIGHT + 250  # Espaço extra para a área inferior
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Agora podemos converter a imagem da pista
track_surface = track_surface.convert()

# Obter os pixels da pista
track_pixels = pygame.surfarray.array3d(track_surface)
track_pixels = np.transpose(track_pixels, (1, 0, 2))

# Atualizar o título da janela
pygame.display.set_caption('Simulação de Aprendizagem por Reforço')

# Variáveis de controle de velocidade e pausa
game_speed = 1.0
paused = False

# Cores
YELLOW = (255, 255, 0)
OLIVE_GREEN = (107, 142, 35)

# Encontrar a posição da linha de partida (linha azul)
blue_mask = np.all(track_pixels == [0, 0, 255], axis=2)
start_positions = np.argwhere(blue_mask)
if start_positions.size == 0:
    print("Erro: Não foi encontrada linha de partida (cor azul) na imagem.")
    sys.exit()
else:
    start_positions = [(int(pos[1]), int(pos[0])) for pos in start_positions]

# Encontrar a posição da linha de chegada (linha vermelha)
red_mask = np.all(track_pixels == [255, 0, 0], axis=2)
goal_positions = np.argwhere(red_mask)
if goal_positions.size == 0:
    print("Erro: Não foi encontrada linha de chegada (cor vermelha) na imagem.")
    sys.exit()
else:
    goal_x = np.mean(goal_positions[:, 1])
    goal_y = np.mean(goal_positions[:, 0])
    goal_position = (goal_x, goal_y)

# Classe do Carrinho
class Car:
    _id_counter = 0  # Classe variável para rastrear IDs únicos
    # Variáveis de classe para acesso aos pixels da pista e posições iniciais
    track_pixels = track_pixels
    track_width = track_width
    track_height = track_height
    start_positions = start_positions

    def __init__(self):
        self.id = Car._id_counter
        Car._id_counter += 1
        self.color = self.generate_color()
        self.create_brain()
        self.reset()

    def generate_color(self):
        random.seed(self.id)  # Semente baseada no ID para cores consistentes
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def reset(self):
        # Inicializar as propriedades do carrinho
        self.x, self.y = random.choice(self.start_positions)
        self.angle = 0
        self.speed = 2  # Velocidade inicial
        self.max_speed = 4  # Velocidade máxima
        self.min_speed = 1  # Velocidade mínima
        self.alive = True
        self.reached_goal = False
        self.distance_traveled = 0
        self.min_distance_to_goal = float('inf')
        self.sensors = []  # Valores dos sensores
        self.size = max(3, int(self.track_width * 0.01))  # Tamanho proporcional à largura da pista

        # Novos atributos para detectar falta de progresso
        self.steps_since_improvement = 0
        self.max_steps_since_improvement = 50  # Aumentado para permitir mais exploração
        self.steps = 0  # Contador de passos totais
        self.max_steps = 1000  # Aumentado para permitir mais exploração

        # Salvar a posição inicial
        self.start_x = self.x
        self.start_y = self.y

        # Novos atributos para rastrear acelerações e velocidade média
        self.accelerations = []
        self.speed_history = []
        self.average_speed = 0

        # Novo atributo para registrar o tempo de chegada
        self.arrival_time = None

    def create_brain(self):
        # Rede neural com pesos aleatórios
        self.input_size = 5  # Número de entradas (3 sensores de obstáculos, ângulo para objetivo, velocidade atual)
        self.hidden_size = 10  # Número de neurônios na camada oculta
        self.output_size = 2  # Número de saídas (steering e aceleração)

        # Inicializar pesos e biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.randn(self.hidden_size)
        self.bias_output = np.random.randn(self.output_size)

    def update(self, dt):
        if not self.alive or self.reached_goal:
            return
        self.read_sensors()
        outputs = self.think()
        self.move(outputs, dt)
        self.check_collision()
        self.update_min_distance_to_goal()
        self.steps += 1

        # Verificar se o carrinho está preso ou excedeu o número máximo de passos
        if self.steps_since_improvement > self.max_steps_since_improvement:
            self.alive = False
        if self.steps > self.max_steps:
            self.alive = False

    def read_sensors(self):
        # Sensor de direção relativa ao objetivo
        direction_to_goal = np.arctan2(goal_position[1] - self.y, goal_position[0] - self.x)
        angle_diff = self.angle_difference(direction_to_goal, self.angle)
        normalized_angle_diff = angle_diff / np.pi  # Normalizar entre -1 e 1

        # Sensores de obstáculos
        sensor_angles = [-np.pi / 4, 0, np.pi / 4]  # Ângulos dos sensores
        obstacle_sensors = []
        for angle_offset in sensor_angles:
            sensor_angle = self.angle + angle_offset
            distance = 0
            max_distance = 50
            # "Ray casting" para cada sensor
            while distance < max_distance:
                distance += 1
                sensor_x = int(self.x + distance * np.cos(sensor_angle))
                sensor_y = int(self.y + distance * np.sin(sensor_angle))
                if 0 <= sensor_x < self.track_width and 0 <= sensor_y < self.track_height:
                    r, g, b = self.track_pixels[sensor_y, sensor_x]
                    if (r, g, b) == (0, 0, 0):  # Preto - Obstáculo
                        break
                else:
                    break
            # Normalizar a distância do sensor
            obstacle_sensors.append(distance / max_distance)

        # Sensor de velocidade atual (normalizado entre 0 e 1)
        normalized_speed = (self.speed - self.min_speed) / (self.max_speed - self.min_speed)

        # Atualizar sensores
        self.sensors = obstacle_sensors + [normalized_angle_diff, normalized_speed]

    def angle_difference(self, angle1, angle2):
        diff = angle1 - angle2
        while diff < -np.pi:
            diff += 2 * np.pi
        while diff > np.pi:
            diff -= 2 * np.pi
        return diff

    def think(self):
        # Processamento da rede neural com função de ativação
        inputs = np.array(self.sensors)
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_activations = np.tanh(hidden)  # Função de ativação
        output = np.dot(hidden_activations, self.weights_hidden_output) + self.bias_output
        output_activations = np.tanh(output)  # Saídas entre -1 e 1
        return output_activations

    def move(self, outputs, dt):
        # Ajustar o ângulo com base na saída de direção (steering)
        steering = outputs[0]
        self.angle += steering * 0.1  # Ajustar a magnitude do steering
        # Limitar o ângulo para evitar rotações excessivas
        self.angle %= 2 * np.pi

        # Ajustar a velocidade com base na saída de aceleração
        acceleration = outputs[1]
        previous_speed = self.speed  # Salvar a velocidade anterior
        self.speed += acceleration * 0.05  # Ajustar a magnitude da aceleração
        # Limitar a velocidade dentro dos limites mínimo e máximo
        self.speed = max(self.min_speed, min(self.speed, self.max_speed))

        # Registrar a aceleração (positivo para acelerar, negativo para desacelerar)
        speed_change = self.speed - previous_speed
        self.accelerations.append(speed_change)

        # Registrar a velocidade atual no histórico
        self.speed_history.append(self.speed)

        # Atualizar a velocidade média
        self.average_speed = np.mean(self.speed_history)

        # Movimento com velocidade ajustada
        distance = self.speed * dt * 60  # Ajustar a velocidade com base no delta time
        self.x += distance * np.cos(self.angle)
        self.y += distance * np.sin(self.angle)
        self.distance_traveled += distance

    def check_collision(self):
        x, y = int(self.x), int(self.y)
        if x < 0 or x >= self.track_width or y < 0 or y >= self.track_height:
            self.alive = False
        else:
            r, g, b = self.track_pixels[y, x]
            if (r, g, b) == (0, 0, 0):  # Preto - Obstáculo
                self.alive = False
            elif (r, g, b) == (255, 0, 0):  # Vermelho - Linha de chegada
                self.reached_goal = True
                self.alive = False  # Considerar como finalizado
                self.arrival_time = pygame.time.get_ticks() - generation_start_time  # Tempo relativo ao início da geração

    def update_min_distance_to_goal(self):
        distance = np.hypot(goal_position[0] - self.x, goal_position[1] - self.y)
        if distance < self.min_distance_to_goal:
            self.min_distance_to_goal = distance
            self.steps_since_improvement = 0  # Resetar contador
        else:
            self.steps_since_improvement += 1  # Incrementar contador

    def draw(self, screen):
        color = self.color
        if self.reached_goal:
            color = (0, 255, 0)  # Verde para carrinhos que chegaram ao fim
        elif not self.alive:
            color = (255, 0, 0)  # Vermelho para carrinhos mortos
        # Desenhar o carrinho como um círculo
        pygame.draw.circle(screen, color, (int(self.x), int(self.y) + INFO_AREA_HEIGHT), self.size)
        # Desenhar uma seta indicando a direção
        end_x = int(self.x + self.size * 2 * np.cos(self.angle))
        end_y = int(self.y + self.size * 2 * np.sin(self.angle)) + INFO_AREA_HEIGHT
        pygame.draw.line(screen, YELLOW, (int(self.x), int(self.y) + INFO_AREA_HEIGHT), (end_x, end_y), 2)
        # Exibir o número do carrinho
        car_number = font_small.render(str(self.id), True, (0, 0, 0))
        number_rect = car_number.get_rect(center=(int(self.x), int(self.y) + INFO_AREA_HEIGHT))
        screen.blit(car_number, number_rect)

# Funções auxiliares para a evolução genética
def crossover(parent1, parent2):
    # Não criamos um novo objeto Car aqui
    child = parent1  # Inicialmente, usamos o primeiro pai como base
    # Cruzamento dos pesos e biases
    for attr in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
        parent1_attr = getattr(parent1, attr)
        parent2_attr = getattr(parent2, attr)
        # Implementação de cruzamento uniforme
        mask = np.random.rand(*parent1_attr.shape) > 0.5
        child_attr = np.where(mask, parent1_attr, parent2_attr)
        setattr(child, attr, child_attr)
    return child

def mutate(car):
    mutation_rate = 0.2  # Aumentar a taxa de mutação
    mutation_strength = 0.3  # Aumentar a intensidade da mutação
    for attr in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
        attr_value = getattr(car, attr)
        mutation_mask = np.random.rand(*attr_value.shape) < mutation_rate
        attr_value += mutation_mask * np.random.randn(*attr_value.shape) * mutation_strength
        setattr(car, attr, attr_value)

def calculate_fitness(car, arrival_order=None):
    # Fitness base inversamente proporcional à menor distância ao objetivo
    fitness = 1 / (car.min_distance_to_goal + 1)

    # Bônus se alcançou o objetivo
    if car.reached_goal:
        fitness += 5000

        # Recompensar por chegar mais rápido (menos passos)
        fitness += (car.max_steps - car.steps) * 10

        # Reforço adicional com base no tempo de chegada
        # Quanto menor o tempo de chegada, maior o bônus
        time_bonus = max(0, (generation_duration - car.arrival_time) / generation_duration) * 5000
        fitness += time_bonus

    # Incentivar afastar-se da linha de partida
    start_distance = np.hypot(car.x - car.start_x, car.y - car.start_y)
    fitness += start_distance * 0.1

    # Penalizar velocidade média baixa
    desired_speed = car.max_speed * 0.75  # Velocidade média desejada (75% da velocidade máxima)
    if car.average_speed < desired_speed:
        speed_penalty = (desired_speed - car.average_speed) * 200  # Ajuste o multiplicador conforme necessário
        fitness -= speed_penalty

    # Bônus por acelerações após desacelerações
    accelerations = [a for a in car.accelerations if a > 0]
    fitness += len(accelerations) * 10  # Recompensar cada aceleração

    # Penalizar colisão precoce de forma mais severa
    fitness -= car.steps * 0.05  # Aumentar a penalidade por colisão precoce

    return fitness

def evolve_population(cars):
    global generation_duration
    # Calcular fitness para cada carrinho
    # Primeiro, ordenar os carrinhos que chegaram ao fim por tempo de chegada
    arrived_cars = [car for car in cars if car.reached_goal]
    arrived_cars.sort(key=lambda c: c.arrival_time)
    # Calcular o tempo total da geração
    if arrived_cars:
        last_arrival_time = arrived_cars[-1].arrival_time
        generation_duration = last_arrival_time
    else:
        generation_duration = pygame.time.get_ticks() - generation_start_time

    # Calcular o fitness considerando o tempo de chegada
    for car in cars:
        car.fitness = calculate_fitness(car)

    # Ordenar os carrinhos pelo fitness
    cars.sort(key=lambda c: c.fitness, reverse=True)
    # Implementar elitismo
    elite_cars = cars[:max(1, population_size // 10)]  # Top 10% passam diretamente
    # Selecionar os melhores carrinhos para reprodução
    best_cars = cars[:max(1, population_size // 2)]
    # Criar nova população
    new_cars = []
    # Manter os carrinhos elite com o mesmo ID e cor
    for elite in elite_cars:
        elite.reset()
        new_cars.append(elite)
    # Gerar novos carrinhos para completar a população
    num_new_cars_needed = population_size - len(new_cars)
    # Usar os carros existentes para criar novos
    for i in range(num_new_cars_needed):
        parent1, parent2 = random.sample(best_cars, 2)
        # Escolher um carro existente para atualizar
        child = cars[len(elite_cars) + i]
        # Fazer crossover e mutação no carro existente
        for attr in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
            parent1_attr = getattr(parent1, attr)
            parent2_attr = getattr(parent2, attr)
            # Implementação de cruzamento uniforme
            mask = np.random.rand(*parent1_attr.shape) > 0.5
            child_attr = np.where(mask, parent1_attr, parent2_attr)
            setattr(child, attr, child_attr)
        # Mutação
        mutate(child)
        # Resetar o carro para a próxima geração
        child.reset()
        # Adicionar o carro à nova população
        new_cars.append(child)
    return new_cars

def display_scoreboard(screen, arrived_cars):
    # Limpar a tela
    screen.fill(OLIVE_GREEN)
    # Desenhar a pista
    screen.blit(track_surface, (0, INFO_AREA_HEIGHT))
    # Desenhar a borda amarela ao redor da pista
    pygame.draw.rect(screen, YELLOW, (0, INFO_AREA_HEIGHT, track_width, track_height), 5)
    # Desenhar o fundo da área superior de informações
    screen.fill(OLIVE_GREEN, rect=[0, 0, SCREEN_WIDTH, INFO_AREA_HEIGHT])
    # Desenhar o fundo da área inferior
    screen.fill(OLIVE_GREEN, rect=[0, INFO_AREA_HEIGHT + track_height, SCREEN_WIDTH, SCREEN_HEIGHT - (INFO_AREA_HEIGHT + track_height)])

    # Desenhar o placar
    scoreboard_rect = pygame.Rect(200, 200, 600, 400)
    pygame.draw.rect(screen, (50, 50, 50), scoreboard_rect)
    pygame.draw.rect(screen, (255, 255, 255), scoreboard_rect, 2)

    # Título do placar
    title_text = font.render("Placar de Chegada", True, (255, 255, 255))
    title_rect = title_text.get_rect(center=(scoreboard_rect.centerx, scoreboard_rect.top + 30))
    screen.blit(title_text, title_rect)

    # Listar os carrinhos em ordem de chegada
    start_y = title_rect.bottom + 20
    for idx, car in enumerate(arrived_cars):
        arrival_time_sec = car.arrival_time / 1000.0  # Converter para segundos
        line_text = font_small.render(f"{idx + 1}º Lugar: Carro {car.id} - Tempo: {arrival_time_sec:.2f}s", True, (255, 255, 255))
        line_rect = line_text.get_rect(x=scoreboard_rect.left + 50, y=start_y + idx * 30)
        screen.blit(line_text, line_rect)

    # Atualizar a tela
    pygame.display.flip()

# **Definição da função draw_brain**
def draw_brain(screen, car, position):
    # Verificar se o carrinho tem sensores válidos
    if not car.sensors:
        return

    # Fundo para a área da rede neural
    brain_area_rect = pygame.Rect(position[0] - 60, position[1] - 50, 620, 300)
    pygame.draw.rect(screen, (200, 200, 200), brain_area_rect)
    pygame.draw.rect(screen, (0, 0, 0), brain_area_rect, 3)

    # Processar a rede neural para obter as ativações
    inputs = car.sensors
    hidden = np.dot(inputs, car.weights_input_hidden) + car.bias_hidden
    hidden_activations = np.tanh(hidden)
    outputs = np.dot(hidden_activations, car.weights_hidden_output) + car.bias_output
    output_activations = np.tanh(outputs)

    # Posições dos neurônios
    input_neurons = [(position[0] + i * 100, position[1]) for i in range(len(inputs))]
    hidden_neurons = [(position[0] + i * 60, position[1] + 80) for i in range(len(hidden_activations))]
    output_neurons = [(position[0] + i * 100 + 100, position[1] + 160) for i in range(len(output_activations))]

    # Desenhar conexões entrada -> oculto
    for i, (x1, y1) in enumerate(input_neurons):
        for j, (x2, y2) in enumerate(hidden_neurons):
            weight = car.weights_input_hidden[i, j]
            color = (0, 255, 0) if weight > 0 else (255, 0, 0)
            width = int(abs(weight) * 2)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), max(1, width))

    # Desenhar conexões oculto -> saída
    for i, (x1, y1) in enumerate(hidden_neurons):
        for j, (x2, y2) in enumerate(output_neurons):
            weight = car.weights_hidden_output[i, j]
            color = (0, 255, 0) if weight > 0 else (255, 0, 0)
            width = int(abs(weight) * 2)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), max(1, width))

    # Desenhar neurônios de entrada
    input_labels = ['Sensor Esquerdo', 'Sensor Frontal', 'Sensor Direito', 'Ângulo para Objetivo', 'Velocidade Atual']
    for i, (x, y) in enumerate(input_neurons):
        intensity = int((inputs[i]) * 255)
        intensity = max(0, min(255, intensity))  # Garantir que esteja no intervalo [0, 255]
        pygame.draw.circle(screen, (intensity, intensity, intensity), (x, y), 15)
        # Rótulos dos neurônios de entrada
        label = font_small.render(input_labels[i], True, (0, 0, 0))
        label_rect = label.get_rect(center=(x, y - 25))
        screen.blit(label, label_rect)

    # Desenhar neurônios ocultos
    for i, (x, y) in enumerate(hidden_neurons):
        intensity = int((hidden_activations[i] + 1) / 2 * 255)
        intensity = max(0, min(255, intensity))
        pygame.draw.circle(screen, (intensity, intensity, intensity), (x, y), 10)

    # Desenhar neurônios de saída
    output_labels = ['Direção (Steering)', 'Aceleração']
    for i, (x, y) in enumerate(output_neurons):
        intensity = int((output_activations[i] + 1) / 2 * 255)
        intensity = max(0, min(255, intensity))
        pygame.draw.circle(screen, (intensity, intensity, intensity), (x, y), 15)
        # Rótulos dos neurônios de saída
        label = font_small.render(output_labels[i], True, (0, 0, 0))
        label_rect = label.get_rect(center=(x, y + 25))
        screen.blit(label, label_rect)

    # Rótulos das camadas
    input_label = font_small.render('Entradas', True, (0, 0, 0))
    input_label_rect = input_label.get_rect(center=(position[0] + 220, position[1] - 40))
    screen.blit(input_label, input_label_rect)

    hidden_label = font_small.render('Camada Oculta', True, (0, 0, 0))
    hidden_label_rect = hidden_label.get_rect(center=(position[0] + 270, position[1] + 80))
    screen.blit(hidden_label, hidden_label_rect)

    output_label = font_small.render('Saídas', True, (0, 0, 0))
    output_label_rect = output_label.get_rect(center=(position[0] + 200, position[1] + 220))
    screen.blit(output_label, output_label_rect)

# Configurações da simulação
population_size = 50
cars = [Car() for _ in range(population_size)]
generation = 1
best_cars_reached_goal = 0  # Recorde de carrinhos que chegaram ao fim
generation_all_reached_goal = None  # Geração em que todos chegaram ao fim

# Carregar fontes
font = pygame.font.SysFont('Arial', 20)
font_small = pygame.font.SysFont('Arial', 16)

# Função para desenhar botões
def draw_button(screen, rect, text):
    pygame.draw.rect(screen, (200, 200, 200), rect)
    pygame.draw.rect(screen, (0, 0, 0), rect, 2)
    text_surface = font_small.render(text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

# Definir botões
BUTTON_Y_OFFSET = INFO_AREA_HEIGHT + track_height + 30
button_reset = pygame.Rect(900, BUTTON_Y_OFFSET, 100, 30)
button_speed_up = pygame.Rect(900, BUTTON_Y_OFFSET + 40, 100, 30)
button_speed_down = pygame.Rect(900, BUTTON_Y_OFFSET + 80, 100, 30)
button_pause = pygame.Rect(900, BUTTON_Y_OFFSET + 120, 100, 30)

# Variáveis de controle do placar
simulation_state = 'running'
scoreboard_start_time = None
SCOREBOARD_DISPLAY_TIME = 5000  # milissegundos

# Variáveis para medir o tempo de geração
generation_start_time = pygame.time.get_ticks()
generation_duration = 0  # Duração total da geração

# Loop principal
clock = pygame.time.Clock()
running = True
while running:
    dt = clock.tick(60) / 1000.0  # Delta time em segundos
    dt *= game_speed  # Ajustar o delta time com base na velocidade do jogo

    if simulation_state == 'running':
        # Desenhar o fundo da área superior de informações
        screen.fill(OLIVE_GREEN, rect=[0, 0, SCREEN_WIDTH, INFO_AREA_HEIGHT])

        # Desenhar a pista
        screen.blit(track_surface, (0, INFO_AREA_HEIGHT))
        # Desenhar a borda amarela ao redor da pista
        pygame.draw.rect(screen, YELLOW, (0, INFO_AREA_HEIGHT, track_width, track_height), 5)

        # Desenhar o fundo da área inferior
        screen.fill(OLIVE_GREEN, rect=[0, INFO_AREA_HEIGHT + track_height, SCREEN_WIDTH, SCREEN_HEIGHT - (INFO_AREA_HEIGHT + track_height)])

        # Eventos do Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if button_reset.collidepoint(mouse_pos):
                    # Reiniciar o jogo
                    Car._id_counter = 0  # Resetar contador de IDs
                    cars = [Car() for _ in range(population_size)]
                    generation = 1
                    best_cars_reached_goal = 0
                    generation_all_reached_goal = None
                    scoreboard_start_time = None
                    simulation_state = 'running'
                    generation_start_time = pygame.time.get_ticks()  # Reiniciar o relógio da geração
                elif button_speed_up.collidepoint(mouse_pos):
                    game_speed *= 2
                    if game_speed > 16:
                        game_speed = 16  # Limitar a velocidade máxima
                elif button_speed_down.collidepoint(mouse_pos):
                    game_speed = max(0.25, game_speed / 2)
                elif button_pause.collidepoint(mouse_pos):
                    paused = not paused
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            # Atualizar e desenhar os carrinhos
            alive_cars = 0
            reached_goal_cars = 0
            for car in cars:
                car.update(dt)
                car.draw(screen)
                if car.alive:
                    alive_cars += 1
                if car.reached_goal:
                    reached_goal_cars += 1

            # Atualizar o recorde
            if reached_goal_cars > best_cars_reached_goal:
                best_cars_reached_goal = reached_goal_cars

            # Verificar se todos os carrinhos chegaram ao fim pela primeira vez
            if reached_goal_cars == population_size and generation_all_reached_goal is None:
                generation_all_reached_goal = generation

            # Exibir informações na tela (na área superior)
            info_y = 10
            if generation_all_reached_goal is not None:
                all_reached_text = font.render(f'Todos os carrinhos chegaram ao fim na geração: {generation_all_reached_goal}', True, (0, 255, 0))
                all_reached_rect = all_reached_text.get_rect(center=(SCREEN_WIDTH // 2, info_y))
                screen.blit(all_reached_text, all_reached_rect)
                info_y += 30
            record_text = font.render(f'Recorde de Carrinhos que Chegaram ao Final: {best_cars_reached_goal}', True, (255, 215, 0))
            record_rect = record_text.get_rect(center=(SCREEN_WIDTH // 2, info_y))
            screen.blit(record_text, record_rect)
            info_y += 30

            # Exibir informações da geração e carrinhos
            text = font.render(f'Geração: {generation} | Vivos: {alive_cars} | Mortos: {population_size - alive_cars - reached_goal_cars} | Chegaram ao Fim: {reached_goal_cars}', True, (255, 255, 255))
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, info_y))
            screen.blit(text, text_rect)

            # Desenhar os botões
            draw_button(screen, button_reset, 'Reset')
            draw_button(screen, button_speed_up, '+Velocidade')
            draw_button(screen, button_speed_down, '-Velocidade')
            draw_button(screen, button_pause, 'Pausar' if not paused else 'Continuar')

            # Desenhar a rede neural do primeiro carrinho
            if cars:
                draw_brain(screen, cars[0], (200, INFO_AREA_HEIGHT + track_height + 50))

            # Verificar se todos os carrinhos estão mortos ou chegaram ao fim
            if alive_cars == 0:
                arrived_cars = [car for car in cars if car.reached_goal]
                if arrived_cars:
                    # Iniciar exibição do placar
                    simulation_state = 'scoreboard'
                    scoreboard_start_time = pygame.time.get_ticks()
                    arrived_cars.sort(key=lambda c: c.arrival_time)
                else:
                    # Nenhum carro chegou ao fim, evoluir imediatamente
                    cars = evolve_population(cars)
                    generation += 1
                    generation_start_time = pygame.time.get_ticks()  # Reiniciar o relógio da geração

        else:
            # Pausado
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if button_pause.collidepoint(mouse_pos):
                        paused = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = False

            # Desenhar os botões com texto atualizado
            draw_button(screen, button_pause, 'Continuar')
            draw_button(screen, button_reset, 'Reset')
            draw_button(screen, button_speed_up, '+Velocidade')
            draw_button(screen, button_speed_down, '-Velocidade')

    elif simulation_state == 'scoreboard':
        # Eventos do Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if button_reset.collidepoint(mouse_pos):
                    # Reiniciar o jogo
                    Car._id_counter = 0  # Resetar contador de IDs
                    cars = [Car() for _ in range(population_size)]
                    generation = 1
                    best_cars_reached_goal = 0
                    generation_all_reached_goal = None
                    scoreboard_start_time = None
                    simulation_state = 'running'
                    generation_start_time = pygame.time.get_ticks()  # Reiniciar o relógio da geração
                elif button_pause.collidepoint(mouse_pos):
                    paused = not paused
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        # Exibir o placar
        display_scoreboard(screen, arrived_cars)
        current_time = pygame.time.get_ticks()
        if current_time - scoreboard_start_time >= SCOREBOARD_DISPLAY_TIME:
            # Tempo de exibição do placar terminou, evoluir população
            cars = evolve_population(cars)
            generation += 1
            generation_start_time = pygame.time.get_ticks()  # Reiniciar o relógio da geração
            simulation_state = 'running'

    pygame.display.flip()

pygame.quit()
sys.exit()
