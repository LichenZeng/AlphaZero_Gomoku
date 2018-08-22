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


# draw background lines
def draw_background(surf):
    # 加载背景图片
    screen.blit(background, back_rect)

    # 画网格线，棋盘为 19行 19列的
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
        pygame.draw.line(surf, BLACK, line[0], line[1], 2)

    for i in range(2, SIZE - 1):
        pygame.draw.line(surf, BLACK,
                         (GRID_WIDTH * i, GRID_WIDTH),
                         (GRID_WIDTH * i, HEIGHT - GRID_WIDTH))
        pygame.draw.line(surf, BLACK,
                         (GRID_WIDTH, GRID_WIDTH * i),
                         (HEIGHT - GRID_WIDTH, GRID_WIDTH * i))

    # circle_center = [
    #     (GRID_WIDTH * 4, GRID_WIDTH * 4),
    #     (WIDTH - GRID_WIDTH * 4, GRID_WIDTH * 4),
    #     (WIDTH - GRID_WIDTH * 4, HEIGHT - GRID_WIDTH * 4),
    #     (GRID_WIDTH * 4, HEIGHT - GRID_WIDTH * 4),
    #     (GRID_WIDTH * 10, GRID_WIDTH * 10)
    # ]
    # # # [(144, 144), (576, 144), (576, 576), (144, 576), (360, 360)]
    # # print(circle_center)
    #
    # for cc in circle_center:
    #     pygame.draw.circle(surf, BLACK, cc, 5)


def draw_text(surf, text, size, x, y, color=WHITE):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)


def move(surf, pos):
    '''
    Args:
        surf: 我们的屏幕
        pos: 用户落子的位置
    Returns a tuple or None:
        None: if move is invalid else return a
        tuple (bool, player):
            bool: True is game is not over else False
            player: winner (USER or AI)
    '''
    print("pos", pos)
    grid = (int(round(pos[0] / (GRID_WIDTH + .0))),
            int(round(pos[1] / (GRID_WIDTH + .0))))
    print("coord", grid)

    if grid[0] <= 0 or grid[0] > SIZE - 1:
        return
    if grid[1] <= 0 or grid[1] > SIZE - 1:
        return

    pos = (grid[0] * GRID_WIDTH, grid[1] * GRID_WIDTH)

    curr_move = (pos, BLACK)
    movements.append(curr_move)
    pygame.draw.circle(surf, BLACK, movements[-1][0], GRID_WIDTH // 2 - 2)

    return curr_move


def draw_movements(surf):
    for move in movements[:-1]:
        pygame.draw.circle(surf, move[1], move[0], GRID_WIDTH // 2 - 2)
        # print(movements)
        # print(type(movements))
        # print(movements[:-1])
        # print(movements[1])
        # print(move)
        # print(movements[-1][0])
        """
        [((144, 468), (0, 0, 0)), ((144, 504), (255, 255, 255)), ((180, 468), (0, 0, 0)), ((108, 468), (255, 255, 255)), ((324, 396), (0, 0, 0)), ((72, 432), (255, 255, 255)), ((252, 360), (0, 0, 0)), ((180, 540), (255, 255, 255))]
        <class 'list'>
        [((144, 468), (0, 0, 0)), ((144, 504), (255, 255, 255)), ((180, 468), (0, 0, 0)), ((108, 468), (255, 255, 255)), ((324, 396), (0, 0, 0)), ((72, 432), (255, 255, 255)), ((252, 360), (0, 0, 0))]
        ((144, 504), (255, 255, 255))
        ((252, 360), (0, 0, 0))
        (180, 540)
        """
    if movements:
        pygame.draw.circle(surf, GREEN, movements[-1][0], GRID_WIDTH // 2 - 2)


def show_go_screen(surf, winner=None):
    note_height = 10
    if winner is not None:
        draw_text(surf, 'You {0} !'.format('win!' if winner == USER else 'lose!'),
                  64, WIDTH // 2, note_height, RED)
    else:
        screen.blit(background, back_rect)

    draw_text(surf, 'Five in row', 64, WIDTH // 2, note_height + HEIGHT // 4, BLACK)
    draw_text(surf, 'Press any key to start', 22, WIDTH // 2, note_height + HEIGHT // 2,
              BLUE)
    pygame.display.flip()
    waiting = True

    while waiting:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYUP:
                waiting = False


game_over = True
running = True
winner = None
movements = []

# Pygame Env init
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("五子棋")
clock = pygame.time.Clock()
all_sprites = pygame.sprite.Group()
base_folder = os.path.dirname(__file__)
# font_name = pygame.font.match_font('华文黑体')
font_name = pygame.font.get_default_font()

# Pygame 加载各种资源
img_folder = os.path.join(base_folder, 'images')
background_img = pygame.image.load(os.path.join(img_folder, 'back.png')).convert()
snd_folder = os.path.join(base_folder, 'music')
hit_sound = pygame.mixer.Sound(os.path.join(snd_folder, 'buw.wav'))
back_music = pygame.mixer.music.load(os.path.join(snd_folder, 'background.mp3'))
pygame.mixer.music.set_volume(0.4)
background = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
back_rect = background.get_rect()
pygame.mixer.music.play(loops=-1)

if __name__ == '__main__':
    while running:
        if game_over:
            show_go_screen(screen, winner)
            game_over = False
            movements = []

        # 设置屏幕刷新频率
        clock.tick(FPS)

        # 处理不同事件
        for event in pygame.event.get():
            # 检查是否关闭窗口
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                response = move(screen, event.pos)
                # if response is not None and response[0] is False:
                #     game_over = True
                #     winner = response[1]
                #     continue

        # Update
        all_sprites.update()

        # Draw / render
        # screen.fill(BLACK)
        all_sprites.draw(screen)
        draw_background(screen)
        draw_movements(screen)

        # After drawing everything, flip the display
        pygame.display.flip()

    pygame.quit()
