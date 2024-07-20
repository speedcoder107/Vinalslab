import math
import numpy as np

class vector ():
    def __init__ (self, r=None, theta=None, x=None , y=None):
        self.r = r
        self.theta = theta
        self.x = x
        self.y = y
        if self.r == None or self.theta == None:
            if self.x == None or self.y == None:
                raise Exception("Atleast provide (r, theta) or (x,y)")

        if self.x == None or self.y == None:
            if self.r == None or self.theta == None:
                raise Exception("Atleast provide (r, theta) or (x,y)")

        if self.x != None and self.y != None:
            self.x = float(self.x)
            self.y = float(self.y)
            self.r, self.theta = change_cartesian_to_polar(self.x,self.y)
        
        if self.r != None and self.theta != None:
            self.r = float(self.r)
            self.theta = float(self.theta)
            self.x, self.y = change_polar_to_cartesian(self.r,self.theta)
    
    def __add__(self,other):
        if not  isinstance(other, vector):
            raise TypeError("you cannot add a vector to a " + type(other))
        else:
            return (vector(x=(self.x + other.x), y=(self.y + other.y)))

    def __sub__(self, other):
        if not isinstance(other, vector):
            raise TypeError("You cannot subtract a vector from a " + type(other))
        else:
            return (vector(x=(self.x - other.x), y=(self.y - other.y)))            

    def __str__(self):
        return "({}, {}, {}, {})".format(self.x, self.y, self.r, self.theta)
    
    def __mul__(self, other):
        if not isinstance(other, vector):
            raise TypeError("you cannot muntiply a vector to " + type(other))
        else:
            dot_product = self.x*other.x + self.y*other.y
            return dot_product
        
def change_cartesian_to_polar(x,y):
    r = math.sqrt(x**2 + y**2)
    try:
        theta = math.atan(float(y)/float(x))
    except:
        if y > 0:
            theta = math.pi/2
        elif y < 0:
            theta = 3*math.pi/2
        else:
            theta = 0
    if theta < 0:
        theta = theta + 2*math.pi

    return r,theta

def change_polar_to_cartesian(r,theta):
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    return x, y

def split_vector(vec):
    # print(vec)
    x_component  = vector(r = None, theta = None, x = vec.x, y = 0)
    y_component = vector(None, None, x = 0, y = vec.y)
    return x_component, y_component

def print_numpy_vector(vec:np.ndarray):
    if isinstance(vec, vector):
        return str(vec)

def printc(vec):
    print("({}, {})".format(vec.x, vec.y))

def printp(vec):
    print("({}, {})".format(vec.r, vec.theta))


np.set_printoptions(formatter={'all': print_numpy_vector})
