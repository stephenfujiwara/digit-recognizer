from tkinter import messagebox
import pygame
import torch
import torchvision.transforms as transforms
from tkinter import *
from model import BobNet

import matplotlib.pyplot as plt

class Pixel():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height 
        self.color = (255, 255, 255)
        self.neighbors = []
    
    def draw(self, surface):
        pygame.draw.rect(surface=surface, color=self.color, rect=(self.x, self.y, self.x + self.width, self.y + self.height))

class Grid():
    def __init__(self, rows, cols, width, height):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.generate_pixels()
    
    def generate_pixels(self):
        self.pixels = []
        col_width = self.width // self.cols
        row_height = self.height // self.rows

        for row in range(self.rows):
            self.pixels.append([])
            for col in range(self.cols):
                self.pixels[row].append(Pixel(x=col_width * col, y=row_height * row, width=col_width, height=row_height))

        self.get_neighbors()       
    
    def get_neighbors(self):
        for row in range(self.rows):
            for col in range(self.cols):
                ## horizontal and vertical neighbors
                # if pixel not on right border
                if col < self.cols - 1:
                    self.pixels[row][col].neighbors.append(self.pixels[row][col + 1])
                # if pixel not on left border
                if col > 0:
                    self.pixels[row][col].neighbors.append(self.pixels[row][col - 1])
                # if pixel not on top border
                if row < self.rows - 1:
                    self.pixels[row][col].neighbors.append(self.pixels[row + 1][col])
                # if pixel not on bottom border
                if row > 0:
                    self.pixels[row][col].neighbors.append(self.pixels[row - 1][col])
                ## diagonal neighbors
                # if not on bottom left
                if col > 0 and row > 0:
                    self.pixels[row][col].neighbors.append(self.pixels[row - 1][col - 1])
                # if not on top left
                if col > 0 and row < self.rows - 1:
                    self.pixels[row][col].neighbors.append(self.pixels[row + 1][col - 1])
                # if not on bottom right
                if col < self.cols - 1 and row > 0:
                    self.pixels[row][col].neighbors.append(self.pixels[row - 1][col + 1])
                # if not on top right
                if col < self.cols - 1 and row < self.rows - 1:
                    self.pixels[row][col].neighbors.append(self.pixels[row + 1][col + 1])
                    
    
    def get_pixel(self, pos):
        try:
            x = pos[0]
            y = pos[1]
            g1 = int(x) // self.pixels[0][0].width
            g2 = int(y) // self.pixels[0][0].height
            return self.pixels[g2][g1]
        except:
            pass
    
    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)
    
    def convert_binary(self):
        # creating a nested list with list comprehension (pretty cool)
        binarized = torch.tensor([[0 if pixel.color == (255, 255, 255) else 1 for pixel in row] for row in self.pixels])
        return binarized

    def __len__(self):
        return self.rows * self.cols

def predict(x):
    # initialize model and load weights
    model = BobNet()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    pred = model(x)
    pred =  torch.argmax(pred).item()
    window = Tk()
    window.withdraw()
    messagebox.showinfo("Prediction", f"This number is a: {pred}")
    window.destroy()

if __name__ == '__main__':
    # nice multiple of 28 x 28
    WIDTH = 392
    HEIGHT = 392
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Digit Recognizer")

    grid = Grid(28, 28, WIDTH, HEIGHT)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # on key-press
            if event.type == pygame.KEYDOWN:
                binarized = grid.convert_binary()
                predict(binarized.unsqueeze(0).unsqueeze(0).float())
                grid.generate_pixels()
            # on left mouse click
            if pygame.mouse.get_pressed()[0]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked_pixel = grid.get_pixel(pos)
                    clicked_pixel.color = (0, 0, 0)
                    for neighbor in clicked_pixel.neighbors:
                        neighbor.color = (0, 0, 0)
                except:
                    pass
            # on right mouse click
            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked_pixel = grid.get_pixel(pos)
                    clicked_pixel.color = (255, 255, 255)
                except:
                    pass
        
        grid.draw(win)
        pygame.display.update()

    pygame.quit()