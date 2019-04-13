import numpy as np 
import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 

X = [0.5, 2.5]
Y = [0.2, 0.9]

def f(w,b,x) :
    return 1.0/(1.0 + np.exp(-(w*x + b)))

def error(w,b) :
    err = 0.0
    for x,y in zip(X,Y) :
        fx = f(w,b,x)
        err = 0.5 * (fx - y) ** 2
    return err

def grad_b(w,b,x,y) :
    fx = f(w,b,x)
    return (fx - y) * fx * (1 - fx)

def grad_w(w,b,x,y) :
    fx = f(w,b,x)
    return (fx - y) * fx * (1 - fx) * x

def graph_plot():
    global ax
    ax = plt.axes(projection='3d')
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('error')
    zline = np.linspace(-1, 1, 1000)
    xline = np.linspace(-10, 10, 1000)
    yline = np.linspace(-10, 10, 1000)
    ax.plot3D(xline, yline, zline, 'gray')
    xdata = list(np.arange(-10,10,0.5))
    ydata = list(np.arange(-10,10,0.5))
    A, B = np.meshgrid(xdata,ydata)
    for a, b in zip(A,B):
        ax.scatter(a,b,error(a,b), cmap='viridis')


def do_gradient_descent(w, b, eta, max_epochs) :
    graph_plot()
    for i in range(max_epochs) :
        dw, db = 0, 0
        for x,y in zip(X,Y) :
            dw = dw + grad_w(w,b,x,y)
            db = db + grad_b(w,b,x,y)
        w = w - eta * dw
        b = b - eta * db
        ax.scatter(w,b,error(w,b),marker='.',color='k')
        ax.scatter(w,b,-1,marker='.',color='k')
    plt.show()

def do_stochastic_gradient_descent(w, b, eta, max_epochs) :
    graph_plot()
    for i in range(max_epochs) :
        dw, db = 0, 0
        for x,y in zip(X,Y) :
            dw = dw + grad_w(w,b,x,y)
            db = db + grad_b(w,b,x,y)
            w = w - eta * dw
            b = b - eta * db
        ax.scatter(w,b,error(w,b),marker='.',color='k')
        ax.scatter(w,b,-1,marker='.',color='k')
    plt.show()

def do_momentum_gradient_descent(w,b,eta,max_epochs):
    graph_plot()
    prev_v_w, prev_v_b, gamma = 0, 0, 0.2
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X,Y):
            dw = dw + grad_w(w,b,x,y)
            db = db + grad_b(w,b,x,y)

        v_w = gamma*prev_v_w + eta*dw
        v_b = gamma*prev_v_b + eta*db
        w = w - v_w
        b = b - v_b
        ax.scatter(w,b,error(w,b),marker='.',color='k')
        ax.scatter(w,b,-1,marker='.',color='k')
        prev_v_w = v_w
        prev_v_b = v_b 
    plt.show()

def do_nesterov_accelerated_gradient_descent(w,b,eta,max_epochs):
    graph_plot()
    prev_v_w, prev_v_b, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw, db = 0, 0
        v_w = gamma*prev_v_w 
        v_b = gamma*prev_v_b 
        for x, y in zip(X,Y):
            dw = dw + grad_w(w - v_w ,b - v_b,x,y)
            db = db + grad_b(w - v_w, b - v_b,x,y)

        v_w = gamma*prev_v_w + eta*dw
        v_b = gamma*prev_v_b + eta*db
        w = w - v_w
        b = b - v_b
        ax.scatter(w,b,error(w,b),marker='.',color='k')
        ax.scatter(w,b,-1,marker='.',color='k')
        prev_v_w = v_w
        prev_v_b = v_b 
    plt.show()

def do_mini_batch_gradient_descent(w,b,eta,max_epochs):
    graph_plot()
    w,b,eta =-2,-2,1.0
    mini_batch_size, num_points_seen = 2,0
    for i in range(max_epochs):
        dw, db, num_points_seen = 0,0,0
        for x,y in zip(X,Y):
            dw = dw + grad_w(w,b,x,y)
            db = db + grad_b(w,b,x,y)
            num_points_seen = num_points_seen + 1

            if num_points_seen % mini_batch_size == 0 :
                w = w - eta*dw
                b = b - eta*db
                ax.scatter(w,b,error(w,b),marker='.',color='k')
                ax.scatter(w,b,-1,marker='.',color='k')
                dw,db = 0, 0 
    plt.show()

def main():
    w, b, eta, max_epochs = -2, -2, 0.5, 1000
    do_mini_batch_gradient_descent(w, b, eta, max_epochs)
    


if __name__ == "__main__":
    main()
