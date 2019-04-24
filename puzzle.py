from tkinter import *
from logic import *
from random import *
import pandas as pd
import time
import pickle
from sklearn.ensemble import RandomForestClassifier



'''
things to do:
1: add score to GUI
2: popup closes when you lose
3: change win to be much higher number


'''
# print tkinter.__file__

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")
colTitles = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'score','lastMove']

KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"

keyDict = {0:KEY_UP,1:KEY_LEFT,2:KEY_RIGHT,3:KEY_DOWN}
numberLetterDict = {0:'w',1:'a',2:'d',3:'s'}

AI_Dict = {'Random_Forest':'random_forest_model_simple.sav'}

# print help(mainloop)
class GameGrid(Frame):
    def __init__(self,AI_Class, AI_Model = None):
        Frame.__init__(self)

        #overarching type of model ex: 'SciKitModel'
        self.AI_Class = AI_Class
        #specific model to use ex: random forest
        self.AI_Model = AI_Model
        
        self.score = 0
        self.final_score = -1
        self.stuckScore = 0
        self.prevBoard = 0

        self.grid()
        self.master.title('2048 Sergey AI Version ')

        

        self.commands = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }

        # Dictionary to hold the different Model Types
        self.Function_Dict = {'SciKitModel':self.AI_Move, 'User_Input': self.User_Input,
                            'KerasModel':-1, 'Random_Move':self.Random_Move}
        
        # Framework AI will use to make a decision 
        self.decisionFunction = self.Function_Dict[self.AI_Class]

        # Loading model
        if (self.AI_Class == 'SciKitModel') or (self.AI_Class == 'Random_Move') :
            # If we're using SciKit model import the model
            if self.AI_Class == 'SciKitModel':
                self.model = pickle.load(open(AI_Dict[self.AI_Model], 'rb'))
                self.one_hot = pickle.load(open('one_hot.sav', 'rb'))      
            
            #FIXME: need to change so one hot encoding always the same
            self.master.bind("<Key>", self.key_down_AI)
            self.after(10,self.key_down_AI)


        if (self.AI_Class == 'User_Input'):
            self.master.bind("<Key>", self.key_down)

        if self.AI_Class == 'Keras':
            self.model = pickle.load(open(AI_Dict[self.AI_Model], 'rb'))
            self.one_hot = pickle.load(open('one_hot.sav', 'rb'))      
            
            #FIXME: need to change so one hot encoding always the same
            self.master.bind("<Key>", self.key_down_AI)
            self.after(10,self.key_down_AI)


        if self.AI_Class == 'Custom':
            # setting model weights
            #fixme, how do I just set a model
            self.one_hot = pickle.load(open('one_hot.sav', 'rb')) 

            self.master.bind("<Key>", self.key_down_AI)
            
            self.after(10,self.key_down_AI)








        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        self.gameStateDF = pd.DataFrame( columns = colTitles)

        self.mainloop()

        #exporting data
        # self.gameStateDF.to_csv('2048_Training_Data_1.csv')




    def init_grid(self):
        ###
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []

            for j in range(GRID_LEN):
                #displaying score
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)
        ###
        # self.myLabel = Label(background, text=self.score).grid(row=4,columnspan=4,ipady = 40,ipadx = 300)
        # myLabel.pack()
        x = 1



    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def update_grid_cells(self):
        ###
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
                    # myLabel.configure(text = 'x')
        ###
        # self.myLabel.configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
        
        self.update_idletasks()

    def User_Input(self,event):
        '''
        User has control over game
        '''
        return repr(event.char)[1]




    def AI_Move(self):
        '''
        Function to take in the game board and choose which move to play based on
        SciKit model.  Future functionality would specify which model as a parameter.
        '''

        #running classifyer
        gameState = unpacker(self.matrix)
        move = self.model.predict([gameState])
        AI_decision = self.one_hot.inverse_transform(move)
        AI_decision = str(AI_decision)[4]

        #if no progress made since last move ie game stuck
        if self.prevBoard == self.matrix:
            
            self.stuckScore += 1
            # print "No progress made on last move!"
            
            AI_decision = numberLetterDict[self.stuckScore]

            if self.stuckScore == 3:
                self.stuckScore = 0 

            return AI_decision

        #updating previous score
        self.prevBoard = self.matrix

        return AI_decision

    def Random_Move(self):
        '''
        Function to play a random move
        '''
        randomDigit = randint(0,3)
        AI_decision = numberLetterDict[randomDigit]
        return AI_decision


        
    def key_down_AI(self):

        # print self.score
        
        move = self.decisionFunction()

        key = repr(move)

        if key in self.commands:    
            # If we want bot to play
            self.matrix,done,scoreChange = self.commands[repr(move)](self.matrix) 
            
            self.score += scoreChange
            
            gameState = unpacker(self.matrix)

            gameState.append(self.score)
            gameState.append(key)

            #adding latest row
            if done == True:
                self.gameStateDF.loc[len(self.gameStateDF)] = gameState
  
            #automatically playing next move if we're in AI Mode, recursion
            if (game_state(self.matrix) !='lose'):
                self.after(10,self.key_down_AI)

            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()
                done=False

  
                # if game_state(self.matrix)=='win':
                #     self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                #     self.grid_cells[1][2].configure(text="Win!",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    
                if game_state(self.matrix) =='lose':
                    self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!",bg=BACKGROUND_COLOR_CELL_EMPTY)

                    self.final_score = int(self.score)
                    
                    self.destroy()
                    self.quit()

    def key_down(self,event):

        print self.score

        move = self.decisionFunction(event)

        key = repr(move)

        if key in self.commands:
            self.matrix,done,scoreChange = self.commands[repr(move)](self.matrix) 
            
            self.score += scoreChange
            
            gameState = unpacker(self.matrix)
            gameState.append(self.score)
            gameState.append(key)

            #adding latest row assuming there was a change in the game board
            if done == True:
                self.gameStateDF.loc[len(self.gameStateDF)] = gameState

            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()
                done=False
                    
                if game_state(self.matrix) =='lose':
                    self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!",bg=BACKGROUND_COLOR_CELL_EMPTY)

                    self.final_score = int(self.score)
                    print 'Final Score: ', self.final_score
                    
                    self.destroy()
                    self.quit()
        
        

        


    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2

def unpacker(list):
    '''
    Sergey code to unpack list
    '''
    unpackedList = []
    for i in list:
        for j in i:
            unpackedList.append(j)
    return unpackedList

def Model_Scorerer(columnName,saveFig = False):
    '''
    Runs a model a 100 times and then compiles the score into 
    a new column on the 2048_Final_Scores.csv spreadsheet

    Params:
    columnName -> title of column to score
    saveFig    -> T/F on saving to spreadshet
    '''
    fileName = '2048_Final_Scores.csv'
    
    df = pd.read_csv(fileName,index_col=[0])

    df[columnName] = 0

    for i in range(1):
        #intializing game
        gamegrid = GameGrid()

        #finding score of run
        final_score = gamegrid.final_score

        print "Score of Run: ", final_score

        #adding this to the data frame
        df.loc[i,columnName] = final_score

    print "Average Score for Run: ", df[columnName].mean()
    print "Standard Deviation: ", df[columnName].std()

    if saveFig == True:
        df.to_csv(fileName)
        print "--- Scores Saved! ---"
    

# gamegrid = GameGrid('Random_Move','Random_Move')

def AI_2048(AI_class = 'Custom' , AI_Model= None , display = True ):
    '''
    Function that takes a model and plays a game of 2048 with it
    
    Params:
    AI_class_init = set to 'Custom' if we're using our own model
    AI_model_init = actual model , not sure on dtype yet
    display       = display the game being played, need to add support for this 

    Returns:
    data_frame of the game that was played
    
    Future: 
    May want to extend the number of games to play

    '''
    gamegrid = GameGrid(AI_class , AI_Model)

    data_frame = gamegrid.gameStateDF

    return data_frame

# Model_Scorerer('Random_Forest',saveFig = False)


my_df = AI_2048('Random_Move','Random_Move')

print my_df.head(-1)


