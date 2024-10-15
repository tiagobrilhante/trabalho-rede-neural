import pygame
import numpy as np
import random
import sys

# Inicialização do Pygame
pygame.init()

# Configurações iniciais da janela temporária
TEMP_SCREEN_WIDTH = 800
TEMP_SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((TEMP_SCREEN_WIDTH, TEMP_SCREEN_HEIGHT))
pygame.display.set_caption('Simulação de Aprendizagem por Reforço')

# Carregamento da imagem da pista
try:
    track_surface = pygame.image.load('pista.png').convert()
except FileNotFoundError:
    print("Erro: Arquivo 'pista.png' não encontrado na pasta do jogo.")
    sys.exit()

track_width, track_height = track_surface.get_size()
track_pixels = pygame.surfarray.array3d(track_surface)

# Transpor track_pixels para [altura][largura][3]
track_pixels = np.transpose(track_pixels, (1, 0, 2))

# Ajustar o tamanho da janela para coincidir com o tamanho da pista
SCREEN_WIDTH = track_width
SCREEN_HEIGHT = track_height + 200  # Espaço extra para informações
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Atualizar o título da janela
pygame.display.set_caption('Simulação de Aprendizagem por Reforço')

# Variáveis de controle de velocidade e pausa
game_speed = 1.0
paused = False

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
    # Variáveis de classe para acesso aos pixels da pista e posições iniciais
    track_pixels = track_pixels
    track_width = track_width
    track_height = track_height
    start_positions = start_positions

    def __init__(self):
        self.create_brain()
        self.reset()

    def reset(self):
        self.x, self.y = random.choice(self.start_positions)
        self.angle = 0
        self.speed = 2  # Velocidade constante
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

    def create_brain(self):
        # Rede neural com pesos aleatórios
        self.weights_input_hidden = np.random.randn(4, 10)  # Aumentado para 10 neurônios ocultos
        self.weights_hidden_output = np.random.randn(10, 1)  # Ajustado para corresponder
        self.bias_hidden = np.random.randn(10)
        self.bias_output = np.random.randn(1)

    def update(self, dt):
        if not self.alive or self.reached_goal:
            return
        self.read_sensors()
        output = self.think()
        self.move(output, dt)
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
        self.sensors = obstacle_sensors + [normalized_angle_diff]

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
        hidden = np.tanh(hidden)  # Função de ativação
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        output = np.tanh(output)  # Saída entre -1 e 1
        return output

    def move(self, output, dt):
        # Ajustar o ângulo com base na saída
        steering = output[0]
        self.angle += steering * 0.1  # Ajustar a magnitude do steering
        # Limitar o ângulo para evitar rotações excessivas
        self.angle %= 2 * np.pi
        # Movimento com velocidade constante
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

    def update_min_distance_to_goal(self):
        distance = np.hypot(goal_position[0] - self.x, goal_position[1] - self.y)
        if distance < self.min_distance_to_goal:
            self.min_distance_to_goal = distance
            self.steps_since_improvement = 0  # Resetar contador
        else:
            self.steps_since_improvement += 1  # Incrementar contador

    def draw(self, screen):
        color = (128, 128, 128)
        if self.reached_goal:
            color = (0, 255, 0)  # Verde para carrinhos que chegaram ao fim
        elif not self.alive:
            color = (255, 0, 0)  # Vermelho para carrinhos mortos
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

# Funções auxiliares para a evolução genética
def crossover(parent1, parent2):
    child = Car()
    # Cruzamento dos pesos e bias
    for attr in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
        parent1_attr = getattr(parent1, attr)
        parent2_attr = getattr(parent2, attr)
        # Implementação de cruzamento uniforme
        mask = np.random.rand(*parent1_attr.shape) > 0.5
        child_attr = np.where(mask, parent1_attr, parent2_attr)
        setattr(child, attr, child_attr)
    return child

def mutate(car):
    mutation_rate = 0.05  # Reduzido para diminuir alterações abruptas
    mutation_strength = 0.1  # Reduzido para alterações mais sutis
    for attr in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
        attr_value = getattr(car, attr)
        mutation_mask = np.random.rand(*attr_value.shape) < mutation_rate
        attr_value += mutation_mask * np.random.randn(*attr_value.shape) * mutation_strength
        setattr(car, attr, attr_value)

# Função de avaliação (fitness)
def calculate_fitness(car):
    # A recompensa é inversamente proporcional à menor distância ao objetivo
    fitness = 1 / (car.min_distance_to_goal + 1)
    # Bônus maior se alcançou o objetivo
    if car.reached_goal:
        fitness += 5000
    # Penalidade se não se afastou da linha de partida
    start_distance = np.hypot(car.x - car.start_x, car.y - car.start_y)
    fitness += start_distance * 0.1  # Incentivar se afastar da partida
    # Penalidade adicional se colidiu cedo
    fitness -= car.steps * 0.01
    return fitness

# Configurações da simulação
population_size = 50
cars = [Car() for _ in range(population_size)]
generation = 1
best_cars_reached_goal = 0  # Recorde de carrinhos que chegaram ao fim
generation_all_reached_goal = None  # Geração em que todos chegaram ao fim

# Carregar fontes
font = pygame.font.SysFont('Arial', 20)

# Função para desenhar botões
def draw_button(screen, rect, text):
    pygame.draw.rect(screen, (200, 200, 200), rect)
    pygame.draw.rect(screen, (0, 0, 0), rect, 2)
    text_surface = font.render(text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

# Definir botões
button_reset = pygame.Rect(SCREEN_WIDTH - 110, 10, 100, 30)
button_speed_up = pygame.Rect(SCREEN_WIDTH - 330, 10, 60, 30)
button_speed_down = pygame.Rect(SCREEN_WIDTH - 260, 10, 60, 30)
button_pause = pygame.Rect(SCREEN_WIDTH - 190, 10, 60, 30)

# Variável para controlar o tempo entre gerações
next_generation_timer = None

# Loop principal
clock = pygame.time.Clock()
running = True
while running:
    dt = clock.tick(60) / 1000.0  # Delta time em segundos
    dt *= game_speed  # Ajustar o delta time com base na velocidade do jogo

    screen.fill((0, 0, 0))
    # Desenhar a pista
    screen.blit(track_surface, (0, 0))

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
                cars = [Car() for _ in range(population_size)]
                generation = 1
                best_cars_reached_goal = 0
                generation_all_reached_goal = None
                next_generation_timer = None
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

        # Exibir informações na tela
        text = font.render(f'Geração: {generation} | Vivos: {alive_cars} | Mortos: {population_size - alive_cars - reached_goal_cars} | Chegaram ao Fim: {reached_goal_cars}', True, (255, 255, 255))
        screen.blit(text, (10, track_height + 10))

        # Exibir o recorde
        record_text = font.render(f'Recorde de Carrinhos que Chegaram ao Final: {best_cars_reached_goal}', True, (255, 215, 0))
        record_rect = record_text.get_rect(center=(SCREEN_WIDTH // 2, track_height + 40))
        screen.blit(record_text, record_rect)

        # Exibir a geração em que todos chegaram ao fim
        if generation_all_reached_goal is not None:
            all_reached_text = font.render(f'Todos os carrinhos chegaram ao fim na geração: {generation_all_reached_goal}', True, (0, 255, 0))
            all_reached_rect = all_reached_text.get_rect(center=(SCREEN_WIDTH // 2, track_height + 70))
            screen.blit(all_reached_text, all_reached_rect)

        # Desenhar os botões
        draw_button(screen, button_reset, 'Reset')
        draw_button(screen, button_speed_up, '+Veloc')
        draw_button(screen, button_speed_down, '-Veloc')
        draw_button(screen, button_pause, 'Pausar' if not paused else 'Continuar')

        # Desenhar a rede neural do primeiro carrinho
        def draw_brain(screen, car, position):
            # Verificar se o carrinho tem sensores válidos
            if not car.sensors:
                return

            # Desenhar os neurônios e conexões da rede neural
            inputs = car.sensors
            hidden = np.dot(inputs, car.weights_input_hidden) + car.bias_hidden
            hidden_activations = np.tanh(hidden)
            output = np.dot(hidden_activations, car.weights_hidden_output) + car.bias_output
            output_activations = np.tanh(output)

            # Posições dos neurônios
            input_neurons = [(position[0] + i * 50, position[1]) for i in range(len(inputs))]
            hidden_neurons = [(position[0] + i * 50, position[1] + 50) for i in range(len(hidden_activations))]
            output_neurons = [(position[0], position[1] + 100)]

            # Desenhar conexões entrada -> oculto
            for i, (x1, y1) in enumerate(input_neurons):
                for j, (x2, y2) in enumerate(hidden_neurons):
                    weight = car.weights_input_hidden[i, j]
                    color = (0, 255, 0) if weight > 0 else (255, 0, 0)
                    width = int(abs(weight) * 2)
                    pygame.draw.line(screen, color, (x1, y1), (x2, y2), max(1, width))

            # Desenhar conexões oculto -> saída
            for i, (x1, y1) in enumerate(hidden_neurons):
                x2, y2 = output_neurons[0]
                weight = car.weights_hidden_output[i, 0]
                color = (0, 255, 0) if weight > 0 else (255, 0, 0)
                width = int(abs(weight) * 2)
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), max(1, width))

            # Desenhar neurônios de entrada
            for i, (x, y) in enumerate(input_neurons):
                intensity = int((inputs[i] + 1) / 2 * 255)
                intensity = max(0, min(255, intensity))  # Garantir que esteja no intervalo [0, 255])
                pygame.draw.circle(screen, (intensity, intensity, intensity), (x, y), 10)

            # Desenhar neurônios ocultos
            for i, (x, y) in enumerate(hidden_neurons):
                intensity = int((hidden_activations[i] + 1) / 2 * 255)
                intensity = max(0, min(255, intensity))
                pygame.draw.circle(screen, (intensity, intensity, intensity), (x, y), 10)

            # Desenhar neurônio de saída
            x, y = output_neurons[0]
            intensity = int((output_activations[0] + 1) / 2 * 255)
            intensity = max(0, min(255, intensity))
            pygame.draw.circle(screen, (intensity, intensity, intensity), (x, y), 10)

        if cars:
            draw_brain(screen, cars[0], (10, track_height + 100))

        # Verificar se todos os carrinhos estão mortos ou chegaram ao fim
        if alive_cars == 0:
            if next_generation_timer is None:
                next_generation_timer = pygame.time.get_ticks()
            else:
                if pygame.time.get_ticks() - next_generation_timer > 3000:  # 3 segundos
                    next_generation_timer = None
                    # Evolução genética
                    # Calcular fitness para cada carrinho
                    for car in cars:
                        car.fitness = calculate_fitness(car)
                    # Ordenar os carrinhos pelo fitness
                    cars.sort(key=lambda c: c.fitness, reverse=True)
                    # Implementar elitismo
                    elite_cars = cars[:max(1, population_size // 10)]  # Top 10% passam diretamente
                    # Selecionar os melhores carrinhos para reprodução
                    best_cars = cars[:max(1, population_size // 2)]
                    new_cars = elite_cars.copy()
                    while len(new_cars) < population_size:
                        parent1, parent2 = random.sample(best_cars, 2)
                        child = crossover(parent1, parent2)
                        mutate(child)
                        new_cars.append(child)
                    cars = new_cars
                    # Resetar todos os carrinhos para a próxima geração
                    for car in cars:
                        car.reset()
                    generation += 1
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
        draw_button(screen, button_speed_up, '+Veloc')
        draw_button(screen, button_speed_down, '-Veloc')

    pygame.display.flip()

pygame.quit()
sys.exit()
