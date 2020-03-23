import random
import os

class Furniture:
    '''contains all features of all funiture objects used on game map
       includes the ability to invert furniture'''
    
    __f_dict = {"chair": [[["|","_"," "], 
                          ["|"," ","|"]], "flip", True],
                "drawer": [[[" ","_","_","_","_","_","_"," "], 
                           ["|"," "," ","_","_"," "," ","|"],
                           ["|"," ","|","*","*","|"," ","|"],
                           ["|"," ","|","_","_","|"," ","|"],
                           ["|","_","_","_","_","_","_","|"]], "open", False],
                "cabinet": [[[" ","_","_","_"," "],
                            ["|"," ","|"," ","|"],
                            ["|"," ","|"," ","|"],
                            ["|","*","|","*","|"],
                            ["|"," ","|"," ","|"],
                            ["|","_","|","_","|"]], "open", False],
                "fireplace":[[[" ","_","_","_","_","_","_","_","_","_","_"," "],
                             ["|"," "," "," "," ","/","\\"," "," "," "," ","|"],
                             ["|"," "," "," ","/"," "," ","\\"," "," "," ","|"],
                             ["|","/","\\","/"," "," "," "," ","\\","/","\\","|"],
                             ["|","_","_","_","_","_","_","_","_","_","_","|"]], "open", False],
                "bookshelf":[[[" ","_","_","_","_"," "],
                             ["|"," "," "," "," ","|"],
                             ["|","_","_","_","_","|"],
                             ["|"," "," "," "," ","|"],
                             ["|","_","_","_","_","|"],
                             ["|"," "," "," "," ","|"],
                             ["|","_","_","_","_","|"],
                             ["|"," "," "," "," ","|"],
                             ["|","_","_","_","_","|"]], "flip", True],
                "carpet": [[[" ","_","_","_","_","_"," "],
                           ["|"," ","_","_","_"," ","|"],
                           ["|","|","@"," ","/","|","|"],
                           ["|","|"," ","/"," ","|","|"],
                           ["|","|","/"," ","@","|","|"],
                           ["|","|","_","_","_","|","|"],
                           ["|","_","_","_","_","_","|"]], "flip", True],
                "door": [[[" ","_","_","_"],
                         ["|"," "," "," "],
                         ["|"," "," ","*"],
                         ["|","_","_","_"]], "open", False]}
    
    def __init__(self, category):
        '''contains attributes of furniture, such as category, size, allowed actions (flip/open), and whether or not it can be flipped or opened'''
        
        self.category = category
        self.structure = Furniture.__f_dict[category][0]
        self.allowed_move = Furniture.__f_dict[category][1]
        self.position = (0,0)
        self.x = len(self.structure[0])
        self.y = len(self.structure)
        #NOTE - invert is an indication of whether or not a piece of furniture is default invertible.  
        #(flip) action furniture are default invertible, (open) action are not.
        #invert is also False for any furniture that has a unsolved puzzle attached.  Only once the puzzle is solved is the furniture able to invert.
        self.invert = Furniture.__f_dict[category][2]
        #is_open is actually the state of the furniture.  At the beginning all furniture are not open.  
        #if inverted, is_open is becomes True.  See inversion() method
        self.is_open = False
        self.message = ""
    
    def inversion(self):
        '''defines the furniture structure of a furniture object that has been inverted (flipped/opened)'''

        if self.invert == False:
            self.message = "You cannot " + self.allowed_move + " this " + self.category + " at this time.\nYou must solve the puzzle that is keeping this " + self.category + " locked."
            self.is_open = False
        else:
            if self.category == "chair":
                self.structure = [["|","_","|"], 
                                  ["|"," "," "]]
            if self.category == "door":
                for i in range(self.y):
                    for j in range(self.x):
                        self.structure[i][j] = " "
            else:
                for i in range(1,self.y-1):
                    for j in range(1,self.x-1):
                        self.structure[i][j] = " "
            self.is_open = True
            self.message = "The " + self.allowed_move + " move on this " + self.category + " is complete"
        return self.message
                    
        
    def __repr__(self):
        '''displays the full structure of the furniture, in original or inverted state'''

        strc_rows = []
        for i in range(len(self.structure)):
            strc_rows.append("".join(self.structure[i]))
        structure_display = "\n".join(strc_rows)
        return structure_display        


class Puzzle:
    '''generates the puzzle that the player needs to solve in order to move on in the game.  This is the parent class of a puzzle
       puzzles can only be enclosed by fireplace, cabinet, door, drawer'''    
    
    def __init__(self, furniture_position, furniture_category):
        '''defines attributes of every puzzle, like answer, enclosing furniture, placement position, default symbol, display'''

        if furniture_category not in ["fireplace", "cabinet", "door", "drawer"]:
            raise Exception("cannot create puzzle attached to furniture")
        else:
            #place the puzzle in the upper right hand corner of the furniture, enclosed by the furniture
            #enclosure means that the puzzle appears inside the area of the furniture, in the upper left corner.  All puzzles are visible at time of map instantiation.
            self.enclosing_furniture = furniture_category
            self.placement = (furniture_position[0]+1, furniture_position[1]+1)
            self.symbol = "?"
            self.hint = []
            self.num_tries = 0
            self.answer = ""
            self.solved = False
            self.display = ""
            self.output = ""
    
    def solve_attempt(self, player_input):
        '''verifies the puzzle is solved if the player input is correct and matches the answer
           notifies player if puzzle is not solved because input is incorrect
           displays the answer for the player if player's wrong attemps exceed 5'''

        if player_input.lower() != self.answer:
            self.num_tries += 1
            if self.num_tries >= 5:
                self.output = "Your have exceed the number of tries to solve this puzzle.\n The answer to the puzzle is: " + self.answer
                self.solved = True
                self.symbol = "~"
            else:
                self.solved = False
                self.output = "Your input is incorrect.  Please try again"
        else:
            self.solved = True
            self.symbol = "~"
            self.output = "Your input is correct.  This " + self.enclosing_furniture + " can now be opened"
        return self.output
    
    def generate_answer(self):
        '''inherited by every subclass, and will be rewritten specifically for each sublass'''

        pass

    def __repr__(self):
        '''displays a representation fo the puzzle.  Will have short instructions on how to solve, along with the number of characters needed to solve'''

        return self.display

class Cryptex(Puzzle):
    '''defines a simple puzzle where the correct input is all letters.  Number of letters required to solve is variable'''
    
    __letters = "abcdefghijklmnopqrstuvwxyz"
    
    def __init__(self, furniture_position, furniture_category, num_answer):
        '''defines attributes of every puzzle, like answer, enclosing furniture, placement position, default symbol, display'''

        super().__init__(furniture_position, furniture_category)
        self.puzzle_type = "Cryptex"
        self.size = num_answer
        self.generate_answer() 
        self.display = "This " + self.puzzle_type + " requires " + str(self.size) + " letters\n" + "?"*self.size
        
    def generate_answer(self):
        '''The answer letters are decided randomly at puzzle instantiation'''

        self.answer = "".join([random.choice(Cryptex.__letters) for i in range(self.size)])

class NumLock(Puzzle):
    '''defines a simple puzzle where the correct input is all numbers.  Number of digits required to solve is variable'''
    
    def __init__(self, furniture_position, furniture_category, num_answer):
        '''defines attributes of every puzzle, like answer, enclosing furniture, placement position, default symbol, display'''

        super().__init__(furniture_position, furniture_category)
        self.puzzle_type = "Number Lock"
        self.size = num_answer
        self.generate_answer()
        self.display = "This " + self.puzzle_type + " requires " + str(self.size) + " numbers\n" + "?"*self.size

    
    def generate_answer(self):
        '''The answer numbers are decided randomly at puzzle instantiation'''

        self.answer = "".join([str(random.randint(0,9)) for i in range(self.size)])
        
class Decoder(Puzzle):
    '''defines more complex puzzle where the correct input is all letters.  Number of letters required to solve is variable'''

    __letters = "abcdefghijklmnopqrstuvwxyz"
    
    def __init__(self, furniture_position, furniture_category, num_answer):
        '''defines attributes of every puzzle, like answer, enclosing furniture, placement position, default symbol, display'''

        super().__init__(furniture_position, furniture_category)
        self.puzzle_type = "cryptex"
        self.size = num_answer
        self.generate_answer()
        #this dictionary will be used as a way to display the key and resulting value for player:
        self.decoder_key = {}
        #this will display the actual puzzle string, which the player must decode
        self.puzzle_display_string = ""
        self.create_decoder()
        #once decoder is generated, decoder_display_string displays the key/value pair decoder for the player, and is the clue for 
        self.decoder_display_string = "key:   "+"|".join(self.decoder_key.keys())+"\n    value: "+"|".join(self.decoder_key.values())
        self.display = "Find the decoded string, given the following key:\n"+"|"+self.puzzle_display_string+"|\n|"+"?"*len(self.puzzle_display_string)+"|" 
        
    def generate_answer(self):
        '''The answer letters are decided randomly at puzzle instantiation'''

        self.answer = "".join([random.choice(Decoder.__letters) for i in range(self.size)])
        
    def create_decoder(self):
        '''defines the decoder that will be displayed as a clue to the player in order to solve the puzzle'''

        offset = random.randint(1,25)
        keys = list(Decoder.__letters[offset: ]+Decoder.__letters[0:offset])
        values = list(Decoder.__letters)
        decoder_generator = dict(zip(values, keys))
        self.decoder_key = dict(zip(keys, values))
        self.puzzle_display_string = "".join([decoder_generator[letter] for letter in self.answer])
        
    


class Room_Map:
    """defines room map appearance, position of all furniture, position of all puzzles, and position of all clues when they are uncovered"""
    
    __allowed_positions = [(1,1), (1,15), (11, 7), (22,1), (22,15)]
    __room_letters = "ABCD"
    
    def __init__(self, player, num_rooms):
        '''defines basic room attributes like size, player position, allowed furniture and allowed positions
           calls additional methods to add furniture and puzzles'''
        
        self.player_symbol = player
        
        #parameters must be fixed to limit scope:
        self.height = 30
        self.num_rooms = num_rooms
        self.width = 26
    
        #following generates the outline of the room, is dynamic and based on the number of rooms chosen:
        self.room = [list(" "+("_"*self.width+" ")*self.num_rooms)] \
                   +[list("|"+(" "*self.width+"|")*self.num_rooms) for i in range(self.height+1)] \
                   +[list("|"+("_"*self.width+"|")*self.num_rooms)]
        
        #defines the initial position of the player
        self.pposition = (self.height//2,1)
        self.room[self.pposition[0]][self.pposition[1]] = self.player_symbol
        
        #positions_dict holds all available positions player can move to, which will be indicated by a "^"
        #every available position is underneath a piece of furniture, in the left bottom corner
        #it will use a position coordinate string, that is displayed to the player, as the dict key.  For instance "A1" means the first Room A, furniture 1. 
        #for each key, it will store the list [^ position, furniture object, puzzle object OR clue string, puzzle object OR clue string placement]
        self.positions_dict = {}
        
        
        #defines the list of furniture in each room.  The exsiting furniture is predefined for up to 4 rooms, their positions will be fixed.
        self.furnitures = [["carpet", "drawer", "fireplace", "cabinet", "chair", "door"], 
                           ["chair", "cabinet", "carpet", "drawer", "bookshelf", "door"],
                           ["fireplace", "drawer", "drawer", "cabinet", "bookshelf", "door"],
                           ["drawer", "cabinet", "fireplace", "cabinet", "drawer", "door"]]
        #initialize the room with furniture
        self.add_furniture()
        self.add_puzzles_and_clues()
    
    def add_furniture(self):
        '''adds all furniture to the room'''

        #list is 2-D, will hold the values of positions_dict
        list_pos_display = []
        
        #the following creates a position for all furniture, including the door, then adds the furniture structure to map.
        for i in range(self.num_rooms):
            for j in range(len(self.furnitures[i])):
                f = Furniture(self.furnitures[i][j])
                if self.furnitures[i][j] != "door":
                    f.position = (Room_Map.__allowed_positions[j][0], Room_Map.__allowed_positions[j][1]+i*(self.width+1))
                else:
                    #doors must be placed walls
                    f.position = (self.height//2 - 1, (self.width)*(i+1)-2+i-1) 
                for a in range(f.y):
                    for b in range(f.x):
                        self.room[f.position[0]+a][f.position[1]+b] = f.structure[a][b]  
                #this adds the allowed position indicator ^ to the map and displays position indicator and position coordinate string to player:
                self.room[f.position[0]+f.y][f.position[1]] = "^"   
                #adds values of [^ position, furniture object] to list_pos_display, the list of values for positions_dict
                pos_display = Room_Map.__room_letters[i]+str((len(self.positions_dict))%6+1)
                self.positions_dict.update({pos_display: [(f.position[0]+f.y, f.position[1]), f]})
                self.room[f.position[0]+f.y][f.position[1]+2] = pos_display[0]
                self.room[f.position[0]+f.y][f.position[1]+3] = pos_display[1]  
  
    
    def add_puzzles_and_clues(self):
        '''adds all puzzles to the room, indicated by ?
           adds all positions of where clues will appear when they are uncovered'''
        
        ###General Notes###
        #every instantiation of a puzzle will generate at least one clue.  The clue is the answer of the puzzle in the case of NumLock() and Cryptex()
        #The clue is the decoder key/value pair in the case of the Decoder()
        #every room has a different puzzle/clue scheme.  
        #no clues in one room will be used in a different room.

        #the objective of room A will be to solve one puzzle at the door.  
        puzzle_A = NumLock(self.positions_dict["A6"][1].position, self.positions_dict["A6"][1].category, 5)
        self.positions_dict["A6"].append(puzzle_A)
        self.positions_dict["A6"].append(puzzle_A.placement)
        self.room[puzzle_A.placement[0]][puzzle_A.placement[1]] = puzzle_A.symbol
        clue_A = puzzle_A.answer
        for i in range(5):
            key = "A"+str(i+1)
            self.positions_dict[key][1].invert = True
            self.positions_dict[key].append(clue_A[i])
            clue_pos = (self.positions_dict[key][1].position[0]+1,self.positions_dict[key][1].position[1]+1)
            self.positions_dict[key].append(clue_pos)

        
        #objective of room B will be to solve one puzzle at the door.
        if self.num_rooms > 1:
            puzzle_B = Cryptex(self.positions_dict["B6"][1].position, self.positions_dict["B6"][1].category, 5)
            self.positions_dict["B6"].append(puzzle_B)
            self.positions_dict["B6"].append(puzzle_B.placement)
            self.room[puzzle_B.placement[0]][puzzle_B.placement[1]] = puzzle_B.symbol
            clue_B = puzzle_B.answer
            for i in range(5):
                key = "B"+str(i+1)
                self.positions_dict[key][1].invert = True
                self.positions_dict[key].append(clue_B[i])
                clue_pos = (self.positions_dict[key][1].position[0]+1,self.positions_dict[key][1].position[1]+1)
                self.positions_dict[key].append(clue_pos)

            
        #the objective of room C will be to solve 2 puzzles.  Only 2 pieces of furniture will be used, in addition to the door
        if self.num_rooms > 2:
            puzzle_C2 = NumLock(self.positions_dict["C2"][1].position, self.positions_dict["C2"][1].category, 7)
            self.positions_dict["C2"].append(puzzle_C2)
            self.positions_dict["C2"].append(puzzle_C2.placement)
            self.room[puzzle_C2.placement[0]][puzzle_C2.placement[1]] = puzzle_C2.symbol
            
            #unlike simple clue, which is 1 char, C3 drawer will include a complex clue (many char) denoted by symbol #.  
            #Complex clues are placed an extra position to the right
            clue_C3 = puzzle_C2.answer
            self.positions_dict["C3"].append(clue_C3)
            self.positions_dict["C3"].append((self.positions_dict["C3"][1].position[0]+1,self.positions_dict["C3"][1].position[1]+2))
            
            puzzle_C6 = Decoder(self.positions_dict["C6"][1].position, self.positions_dict["C6"][1].category, 5)
            self.positions_dict["C6"].append(puzzle_C6)
            self.positions_dict["C6"].append(puzzle_C6.placement)
            self.room[puzzle_C6.placement[0]][puzzle_C6.placement[1]] = puzzle_C6.symbol
            
            #C2 will also enclose a complex clue in addition to the puzzle, which is accessible one position to the right
            clue_C2 = puzzle_C6.decoder_display_string
            self.positions_dict["C2"].append(clue_C2)
            self.positions_dict["C2"].append((self.positions_dict["C2"][1].position[0]+1,self.positions_dict["C2"][1].position[1]+2))

            
            #allow all other furnitures to open in Room C, in case player wants to check there.  
            self.positions_dict["C1"][1].invert = True
            self.positions_dict["C3"][1].invert = True
            self.positions_dict["C4"][1].invert = True
            self.positions_dict["C5"][1].invert = True

        
        #In room 4, 5 number locks of varying size are created in succession.  
        #Player must solve in order of D2-D6, where the D1 provides the first clue and each subsequent furniture can be opened by previous furniture.
        if self.num_rooms>3:
            self.positions_dict["D1"][1].invert = True
            for i in range(1, 6):
                current_key = "D"+str(i+1)
                previous_key = "D"+str(i)
                puzzle_i = NumLock(self.positions_dict[current_key][1].position, self.positions_dict[current_key][1].category, i+3)
                self.positions_dict[current_key].append(puzzle_i)
                self.positions_dict[current_key].append(puzzle_i.placement)
                self.room[puzzle_i.placement[0]][puzzle_i.placement[1]] = puzzle_i.symbol
                clue_i = puzzle_i.answer
                self.positions_dict[previous_key].append(clue_i)
                #since all clues are complex, their # symbol will appear one more position to the right
                self.positions_dict[previous_key].append((self.positions_dict[previous_key][1].position[0]+1,self.positions_dict[previous_key][1].position[1]+2))

    def __repr__(self):
        '''displays the current struture of the entire rooms map'''

        room_rows = []
        for i in range(len(self.room)):
            room_rows.append("".join(self.room[i]))
        room_display = "\n".join(room_rows)
        return room_display


class Gameplay:
    '''The class controls the game play.  It defines all methods that can manipulate the map and furniture.
       Also defines additional features that the player can use to assist in the game play experience'''

    def __init__(self):
        '''creates welcome message
           defines menu header
           defines some attributes that are used repeatedly in gameplay
           error checks player inputs before map instantiation
           instantiates and displays the entire map'''

        #This is the welcome message body, displays at start of game
        print("\nWELCOME TO ESCAPE THE ROOM(S)!\n* You will be placed inside a series of rooms (4 max)")
        print("* You must escape through each door\n* The game is complete when you open the final door")
        print("*********** ENJOY ***********\n")
        
        #requests for number of rooms and player symbol is made.  Will repeately ask player for either, until a correct amount is entered
        number_of_rooms = int(input("Please enter the number of rooms you wish to play: "))
        while number_of_rooms > 4 or number_of_rooms < 0:
        	number_of_rooms = int(input("Number of rooms must be a number 1-4. \nPlease enter the number of rooms you wish to play: "))
        symbol_of_player = str(input("Please enter a ONE-character symbol to designate as your player icon: "))
        while symbol_of_player in "|_/ABCD@*^?~#\\" or len(symbol_of_player) != 1:
        	symbol_of_player = str(input("Invalid symbol! Please enter a symbol to designate as your player icon: "))
        
        #instantiates the room
        self.rm = Room_Map(symbol_of_player, number_of_rooms)

        #creates menu header used for when menu is called.  menu body is dynamic based on player position
        self.menu_header= "\n******* MENU *******\n" + \
                    "^ = movable position\n" + \
                    "? = unsolved puzzle location\n" + \
                    "~ = solved puzzle location\n" + \
                    "# = complex (multi-character) clue\n" + \
                    "To interact with furniture or puzzle, please move to the ^ underneath\n\n"+ \
                    "Please SELECT from the following current options:"
        
        #dictionary used to call other methods in this class inside of the run method, which uses while loop to repeatedly request player's next intended move
        self.all_options = {"move": self.move,
                            "open": self.open,
                            "flip": self.open,
                            "solve": self.solve,
                            "examine clue": self.examine_clue,
                            "examine puzzle": self.examine_puzzle,
                            "menu": self.menu, 
                            "clues": self.current_clues}

        self.player_coordinate = ""
        self.allowed_positions = ["A1", "A2", "A3", "A4", "A5", "A6"]
        self.furn_cat = ""
        self.furn_move = ""
        self.furn_pos = (0,0)
        self.current_clues = {}
        print(self.rm)

    def move(self):
        '''defines player's movement on map.  clears old map and redisplays new map version after move is complete
           also checks that the player moves to a valid position. For instance, moving to a position in a different room when the door is not open is considered invalid.'''

        #if door is open to next room, append next room's positions to allowed positions, so the player can move to a position in the next room
        if self.rm.num_rooms > 1 and self.rm.positions_dict["A6"][1].is_open == True:
            self.allowed_positions.extend(["B1", "B2", "B3", "B4", "B5", "B6"])
        if self.rm.num_rooms > 2 and self.rm.positions_dict["B6"][1].is_open == True:
            self.allowed_positions.extend(["C1", "C2", "C3", "C4", "C5", "C6"])
        if self.rm.num_rooms > 3 and self.rm.positions_dict["C6"][1].is_open == True:
            self.allowed_positions.extend(["D1", "D2", "D3", "D4", "D5", "D6"])

        move_to = str(input("Please enter the position coordinate next to the ^ that you wish to move to: ")).upper()
        while move_to not in self.allowed_positions:
            move_to = str(input("Not a valid position! Please enter the position coordinate next to the ^ that you wish to move to: ")).upper()
        if self.rm.pposition == (self.rm.height//2,1):
            self.rm.room[self.rm.pposition[0]][self.rm.pposition[1]] = " "
        else:
            self.rm.room[self.rm.pposition[0]][self.rm.pposition[1]] = "^"
        
        self.player_coordinate = move_to
        self.furn_cat = self.rm.positions_dict[self.player_coordinate][1].category
        self.furn_move = self.rm.positions_dict[self.player_coordinate][1].allowed_move
        self.furn_pos = self.rm.positions_dict[self.player_coordinate][1].position
        
        self.rm.pposition = self.rm.positions_dict[move_to][0]
        self.rm.room[self.rm.pposition[0]][self.rm.pposition[1]] = self.rm.player_symbol
        os.system("clear")
        print(self.rm)

    def menu(self, p=0):
        '''defines the player menu.  menu is dynamic based on player position'''

        #current options available at all positions
        current_options = {"move": "move to a new position ^",
                           "exit": "exit game, progress is NOT saved",
                           "menu": "see full gameplay menu",
                           "clues": "to see all clues you have acquired thus far"}
        
        #additional add on options if the player is in front of furniture that flips or opens, next to a clue, or next to a puzzle.
        if self.rm.pposition != (self.rm.height//2,1):
            if self.rm.positions_dict[self.player_coordinate][1].is_open == False:
                current_options.update({self.furn_move: self.furn_move+" the "+self.furn_cat})
            if self.rm.room[self.furn_pos[0]+1][self.furn_pos[1]+1] == "?":
                current_options.update({"solve": "attempt to solve the puzzle associated with this "+ self.furn_cat})
                current_options.update({"examine puzzle": "examine the puzzle associated with this "+ self.furn_cat})
            if self.rm.room[self.furn_pos[0]+1][self.furn_pos[1]+2] == "#":
                current_options.update({"examine clue": "examine the complex clue(#) associated with this "+ self.furn_cat})

        #p is a variable i used to define if the menu should print or not.  
        #if p is 0, or default, the menu will print for the player.  This happens when the player selects menu during run()
        #if p = 1 or any other value, as set explicitly by the code in the run() method, this means that 
            #I am using the menu method to output just the shortlist of available options
            #the shortlist of available options is displayed every time the player is then prompted on what to do next in the run() method
        if p != 0:
            return list(current_options.keys())
        else:
            print(self.menu_header)
            for o in current_options:
        	    print(o+": "+current_options[o])
            print("********************\n")


    def open(self):
        '''calls the inversion() method from the Furniture class
           then adds the outcome to the room map and redisplays the new verion of map
           since flip/open do the same thing, both choices from player will call this method'''
        
        f = self.rm.positions_dict[self.player_coordinate][1]
        message = f.inversion()
        if f.is_open:
        	for a in range(f.y):
        		for b in range(f.x):
        			self.rm.room[f.position[0]+a][f.position[1]+b] = f.structure[a][b]
        		if f.category == "door":
        			self.rm.room[f.position[0]+a][f.position[1]+f.x] = " "

        #retrive clue coordinates and clue string to show clue:
        #all uncovered clues are automatically added to self.current_clues for display in current_clues() method
	        coordinates = self.rm.positions_dict[self.player_coordinate]
	        if len(coordinates) == 4 and type(coordinates[2]) == str:
	        	clue = coordinates[2]
	        	self.current_clues.update({self.player_coordinate: clue})
	        	if len(clue) == 1:
	        		self.rm.room[coordinates[3][0]][coordinates[3][1]] = clue
	        	else:
	        		self.rm.room[coordinates[3][0]][coordinates[3][1]] = "#"
	        elif len(coordinates) == 6 and type(coordinates[4]) == str:
	        	clue = coordinates[4]
	        	self.current_clues.update({self.player_coordinate: clue})
	        	if len(clue) == 1:
	        		self.rm.room[coordinates[5][0]][coordinates[5][1]] = clue
	        	else:
	        		self.rm.room[coordinates[5][0]][coordinates[5][1]] = "#"

        os.system("clear")
        print(self.rm)
        print(message)

        #if final door is open
        last_door_check = {"A6":1, "B6":2, "C6":3, "D6":4}
        if f.category == "door" and f.is_open:
        	if last_door_check[self.player_coordinate] == self.rm.num_rooms:
        		print("\n********** CONGRATULATIONS!!!  YOU HAVE ESCAPED **********\n")

    def current_clues(self):
        '''displays a list of all clues the player has acquired thus far
           dictionary of clues and positions are actually generated inside the open() method and used here to print the list for player'''

        if len(self.current_clues) == 0:
            print("\nYou have acquired no clues\n")
        else:
            print("\nYou have acquired the following clues:")
            for c in self.current_clues:
                print(c+": "+self.current_clues[c])
            
            print("")

    def solve(self):
        '''checks to see that the player is standing next to furniture that contains a puzzle
           then prompts player for guess and calls the solve_attempt() method from the Puzzle class
           displays the guess result, adds changes to room map, and redisplays room map'''

        coordinates = self.rm.positions_dict[self.player_coordinate]
        if self.rm.room[coordinates[1].position[0]+1][coordinates[1].position[1]+1] == "?":
            coordinate_puzzle = coordinates[2]
            print("")
            print(coordinate_puzzle)
            guess = str(input("Please input your guess for this puzzle: ")).lower()
            message = (coordinate_puzzle).solve_attempt(guess)
            if coordinate_puzzle.solved:
                self.rm.room[coordinate_puzzle.placement[0]][coordinate_puzzle.placement[1]] = coordinate_puzzle.symbol
                #allow corresponding furniture to open:
                coordinates[1].invert = True
            os.system("clear")
            print(self.rm)
            print(message)
        else:
            print("There is no puzzle here for you to solve")

    def examine_clue(self):
        '''allows player to see details of a complex (multi-character) clue, denoted by #.  Player must be standing at the coordinate position underneath the clue location'''

        if self.player_coordinate in self.current_clues:
            print("\n" + self.player_coordinate+ ": "+self.current_clues[self.player_coordinate]+"\n")
        else:
            print("There is no visible clue to examine at this position")
    
    def examine_puzzle(self):
        '''allows player to see details of a puzzle, if the Player is standing at the coordinate position underneath the puzzle location.  
           requires the __repr__ method from Puzzle class to be well defined for printing'''

        coordinates = self.rm.positions_dict[self.player_coordinate]
        if self.rm.room[coordinates[1].position[0]+1][coordinates[1].position[1]+1] == "?":
            print("\nThe puzzle looks like this:\n")
            print(coordinates[2])
            print("")
        else:
            print("There is no puzzle here.")

    def run(self):
        '''method that runs the entire game by repeatedly prompting the player for their choice of action
           then calls a method from this class to execute the player's chosen action
           prompts to player display available action options for player based on position'''

        next_move = str(input("What would you like to do? (enter menu to see available options) ")).lower()

        while next_move != "exit":
            while next_move not in self.all_options:
                next_move = str(input("Not a valid move! What would you like to do next ("+string_available_options+")? ")).lower()
                if next_move == "exit":
                	break
            if next_move == "exit":
            	break
            
            #use the dictionary all_options to call the corresponding method from this class
            self.all_options[next_move]()

            #uses the output of menu() to create a short list of available actions inside the prompt.  
            #per method definition, menu() must have another input, not 0, to return list rather than print menu
            current_available_options = self.menu(1)
            string_available_options = ", ".join(current_available_options)
            next_move = str(input("What would you like to do next ("+string_available_options+")? ")).lower()
        
        print("\n************ THANK YOU FOR PLAYING! ************\n")

#these 2 lines are the only script needed to run the entire game, outside of Classes defined above.
g = Gameplay()
g.run()

