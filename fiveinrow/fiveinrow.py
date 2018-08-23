#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# from pprint import pprint
# import numpy as np
import pygame
import os
import random
import time

USER, AI = 1, 0

WIDTH = 720
HEIGHT = 720
SIZE = 9
GRID_WIDTH = WIDTH // SIZE
FPS = 30

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class GobangState:

    def __init__(self):
        # Record moved position
        self.movements = []

        # Pygame Env init
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_caption("人机大战五子棋 V1.0")

        self.clock = pygame.time.Clock()
        self.all_sprites = pygame.sprite.Group()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        # self.font_name = pygame.font.match_font('华文黑体')
        self.font_name = pygame.font.get_default_font()

        # Images
        base_folder = os.path.dirname(__file__)
        img_folder = os.path.join(base_folder, 'images')
        background_img = pygame.image.load(os.path.join(img_folder, 'back.png')).convert()
        self.background = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
        self.back_rect = self.background.get_rect()

        # Sounds
        snd_folder = os.path.join(base_folder, 'music')
        hit_sound = pygame.mixer.Sound(os.path.join(snd_folder, 'buw.wav'))
        back_music = pygame.mixer.music.load(os.path.join(snd_folder, 'background.mp3'))
        pygame.mixer.music.set_volume(0.4)
        pygame.mixer.music.play(loops=-1)

    # draw background lines
    def draw_background(self):
        # 加载背景图片
        self.screen.blit(self.background, self.back_rect)

        # 1. 画出边框
        rect_lines = [
            ((GRID_WIDTH, GRID_WIDTH), (GRID_WIDTH, HEIGHT - GRID_WIDTH)),
            ((GRID_WIDTH, GRID_WIDTH), (WIDTH - GRID_WIDTH, GRID_WIDTH)),
            ((GRID_WIDTH, HEIGHT - GRID_WIDTH),
             (WIDTH - GRID_WIDTH, HEIGHT - GRID_WIDTH)),
            ((WIDTH - GRID_WIDTH, GRID_WIDTH),
             (WIDTH - GRID_WIDTH, HEIGHT - GRID_WIDTH)),
        ]
        # # [((36, 36), (36, 684)), ((36, 36), (684, 36)), ((36, 684), (684, 684)), ((684, 36), (684, 684))]
        # print(rect_lines)
        for line in rect_lines:
            pygame.draw.line(self.screen, BLACK, line[0], line[1], 2)

        # 2. 画网格线
        for i in range(2, SIZE - 1):
            pygame.draw.line(self.screen, BLACK,
                             (GRID_WIDTH * i, GRID_WIDTH),
                             (GRID_WIDTH * i, HEIGHT - GRID_WIDTH))
            pygame.draw.line(self.screen, BLACK,
                             (GRID_WIDTH, GRID_WIDTH * i),
                             (HEIGHT - GRID_WIDTH, GRID_WIDTH * i))

    def draw_text(self, text, size, x, y, color=WHITE):
        font = pygame.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        self.screen.blit(text_surface, text_rect)

    def move(self, pos):
        # print("pos", pos)
        grid = (int(round(pos[0] / (GRID_WIDTH + .0))),
                int(round(pos[1] / (GRID_WIDTH + .0))))
        # print("coord", grid)

        if grid[0] <= 0 or grid[0] > SIZE - 1:
            return
        if grid[1] <= 0 or grid[1] > SIZE - 1:
            return

        pos = (grid[0] * GRID_WIDTH, grid[1] * GRID_WIDTH)

        curr_move = (pos, BLACK)
        self.movements.append(curr_move)
        pygame.draw.circle(self.screen, BLACK, self.movements[-1][0], GRID_WIDTH // 2 - 2)
        pygame.display.flip()

        return curr_move

    def get_coordinate(self):
        while True:
            # 处理不同事件
            for event in pygame.event.get():
                # 检查是否关闭窗口
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # print("pos:", event.pos)
                    # # pos: (632, 88) = (X, Y) =  W, H --but we need--> H, W
                    grid = (int(round(event.pos[1] / (GRID_WIDTH + .0))),
                            int(round(event.pos[0] / (GRID_WIDTH + .0))))

                    # Throw out these invaild boundary points
                    if grid[0] <= 0 or grid[0] > SIZE - 1:
                        return
                    if grid[1] <= 0 or grid[1] > SIZE - 1:
                        return

                    grid = (grid[0] - 1, grid[1] - 1)  # H, W
                    # print("coordinate:", grid)
                    return grid

    def draw_movements(self):
        for move in self.movements[:-1]:
            pygame.draw.circle(self.screen, move[1], move[0], GRID_WIDTH // 2 - 2)
        if self.movements:
            # Assume AI use WHITE temporary
            if self.movements[-1][1] == WHITE:
                pygame.draw.circle(self.screen, GREEN, self.movements[-1][0], GRID_WIDTH // 2 - 2)
            else:
                pygame.draw.circle(self.screen, self.movements[-1][1], self.movements[-1][0], GRID_WIDTH // 2 - 2)

    def show_go_screen(self, winner=None):
        note_height = 10
        if winner is not None:
            self.draw_text('You {0} !'.format('win!' if winner == USER else 'lose!'),
                           64, WIDTH // 2, note_height, RED)
        else:
            self.screen.blit(self.background, self.back_rect)

        self.draw_text('Five in row', 64, WIDTH // 2, note_height + HEIGHT // 4, BLACK)
        self.draw_text('Press any key to start', 22, WIDTH // 2, note_height + HEIGHT // 2, BLUE)
        pygame.display.flip()
        waiting = True

        while waiting:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.KEYUP:
                    waiting = False
        self.frame_flash()

    def frame_flash(self):
        # 设置屏幕刷新频率
        self.clock.tick(FPS)

        # Update
        # self.all_sprites.update()
        # self.all_sprites.draw(self.screen)
        self.draw_background()
        self.draw_movements()

        # After drawing everything, flip the display
        pygame.display.flip()

    def frame_quit(self):
        pygame.quit()


if __name__ == '__main__':
    game_over = True
    running = True
    winner = None

    gobang = GobangState()
    gobang.frame_flash()

    while running:
        if game_over:
            gobang.show_go_screen(winner)
            game_over = False
            gobang.movements = []

        # 处理不同事件
        for event in pygame.event.get():
            # 检查是否关闭窗口
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                response = gobang.move(event.pos)
                # if response is not None and response[0] is False:
                #     game_over = True
                #     winner = response[1]
                #     continue

    gobang.frame_quit()
