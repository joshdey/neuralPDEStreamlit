import os
import sys

# Stop TF from showing unnecessary deprecation warnings
import warnings
warnings.filterwarnings('ignore')
from tensorflow import logging
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from pydens import Solver, NumpySampler, cart_prod, add_tokens
from pydens import plot_loss, plot_pair_1d, plot_2d, plot_sections_2d, plot_sections_3d

import streamlit as st

add_tokens()
#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title = "Neural Network Solutions to DiffEq's")
    #col1, col2 = st.beta_columns(2)
    #st.image("fatbrain-logo.png")
    #local_css("style.css")
    st.markdown("<img src = 'https://fb-logo-images.s3-us-west-2.amazonaws.com/fatbrain-logo-color-h.png' style='width:210px;height:50px;'>",unsafe_allow_html=True)
    st.title("Solving Differential Equations with Neural Networks")
    st.markdown("Version 1 | Author: Josh |  Build Date: 11/16/20")
    st.markdown("### **Link to PyDens Repo:** https://github.com/analysiscenter/pydens")
    
    st.header('Blog')
    st.subheader('What is a Differential Equation?')
    "A differential equation is an equation comprised of a function and one or multiple of its derivatives. One of the most famous physics equations of all time is a differential equation:"
    st.latex(r'''
             F(x)=ma=m\ddot{x}.
             ''')
    "Relating force as a function of position to acceleration, the second derivative of position, makes this a second-order ordinary differential equation; that is, no partial derivatives are involved in the differential equation with the max derivate being a double derivative of x. For example, the wave equation which models the behavior a classical wave:"
    
    st.latex(r'''
            \frac{\partial^2 u}{\partial t^2} = c^2 (\frac{\partial^2 u}{\partial x_1^2} + \frac{\partial^2 u}{\partial x_2^2} + ... + \frac{\partial^2 u}{\partial x_n^2}),
            ''')
    "is a second order partial differential equation - second order because again the highest order derivative is two and partial because it takes derivatives with respect to time and at least one other spatial dimension."
    
    st.subheader("Solving Partial Differential Equations")
    "Analytical solutions to PDE's are possible for rather simple linear PDE's - typically through the process of separation of variables. This technique involves reducing the PDE into a system of ODE's which are then simpler to solve given the dimensionality reduction from one equation of n variables to n equations of one variable. Alternatively, sometimes differential equations can be reduced to simpler differential equations to which the solution is known through a change of variables. Taking the Black-Scholes PDE from finance governing the price of different options calls or puts in the European market for example:"
    st.latex(r'''
        \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial ^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rv = 0,
    ''')
    "can be reduced to the heat equation from physics:"
    st.latex(r'''
        \frac{\partial u}{\partial \tau} = \frac{\partial^2 u}{\partial x^2},
    ''')
    
    "which has a known solution."
    "As the PDE gets more complex (higher order differential equations or those of the nonlinear nature) solving analytically becomes much harder and numerical methods can become more applicable in these contexts. Numerical techniques for solving such PDE's include finite difference or finite element methods typically taught in introductory undergraduate computational methods courses. In recent years, using neural networks to solve PDE's has generated a huge interest in the scientific community mostly due to the accuracy of the results by reducing the need for a step size -- even the smallest step size will generate some degree of inaccuracy, and as the step size shrinks or the equation becomes more difficult, more computational power is needed to generate the same accuracy."
    
    st.subheader("Using Neural Networks to Solve PDE's")
    "How exactly a neural network is used to solve unknown PDE's boils do to two main concepts: the Galerkin Method and using the neural network as a function approximator. "
    st.markdown("#### Galerkin Method")
    "The Galerkin method is a numerical method to transform continuous operator problems, like a differential equation where the differential operator:"
    st.latex(r'''
        \frac{\partial}{\partial x}
    ''')
    "is the continous operator, into discrete problems. Typically, the unknown function is approximated with a linear combination of basis functions. An error function is minimized by optimizing the coefficients of the linear combination over the entire domain of the function, or an integral over all possible values."
    st.markdown("#### NN's as Function Approximators")
    st.markdown("The * Universal Approximation Theorem * of artificial neural networks states that any continuous function can be approximated by a singular layered neural network. It should be noted that the number of units, or neurons, in the network scales exponentially with the complexity of the function it aims to approximate. ")
    
    st.subheader("Putting These Two Together")
    st.markdown("As mentioned before, the Galerkin Method approximates a function with a linear combination of basis functions. The _ Deep Galerkin Method _ (DGM) instead uses a neural network to approximate the function removing the need for the basis functions. Another key quality of the DGM algorithm is that its meshfree allowing it to solve higher order PDE's. Using the network to approximate the function, we just have to optimize the network parameters to reduce error. Particularly astute readers may notice that in order to reduce error, one must evaluate the actual function (which can contain derivatives up to the nth order) at every point in time and space which may be extremely computationally expensive. However, we can utilize backpropogation and computational graphs to calculate values of the differential form ** and ** compute the gradients of the network output to make the approximation better. We swap the integral mentioned above for a summation of a batch of points which we sample; i.e, we can sample more points in certain regions (potentially near any boundary/initial conditions passed through) and less or even zero in any uninteresting regions.")
    
    "This is exactly how the PyDEns package works in solving partial differential equations, at least up to order 2, of which examples can be found of below in the ** PDE's ** section below. Additionally, the PyDEns framework allows for an ansatz, or educated guess, to be entered as an input. We can supply what form we expect the solution to take as an input allowing the network to evaluate and solve the PDE much more easily."
        
    
    
    
    
    
    st.header("PDE's")
    st.subheader("Poisson")
    st.write("Let's look at the Poisson equation in 2 dimensions: ")
    st.latex(r'''
    \nabla^2 \Psi (x,y) = f(x,y),
    ''')
    st.write("with Dirichlet boundary conditions, that is the function takes on a value of 1 at all values along the boundaries. If we train our network on 1500 iterations on batches of 100 points, we can see the model loss as a function of step here: ")
    st.image("Images/poissonLoss.png")
    st.write("and can see that the loss essentially becomes 0 by the end of training. The equation with these boundary conditions are relatively simple and so 1500 iterations suffices for a very accurate result as shown below.")
    st.image("Images/poissonSolution.png")
    st.write("We can also track the solution for the sum of the second derivatives wrt x and y as shown below")
    st.image("Images/poisson2Derivative.png")
    
    st.subheader("Wave Equation")
    st.write("Now, let's look at a simple example of one of the most fundamental physics differential equations: the wave equation in one dimension. This equation, ")
    st.latex(r'''
        \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2},
    ''')
    st.write("models the evolution of a plucked string (say on a guitar) with fixed ends. This formula does eventually form the building blocks for more modern and relevant physics equations like Schrodinger's Equation, but let's consider this example for simplicity. As before, we'll plot the model loss over the iterations, but due to the somewhat more complex nature of this equation, we'll train it on 10000 iterations of 150 batch points.")
    st.image("Images/waveLoss.png")
    st.write('As we can see, we do need a lot more iterations to get to the "close to zero" point and even then we don\'t get as close, though as we can see below, our model is still a relatively good fit for what a plucked guitar string should look like.')
    st.image("Images/waveSolution.png")
    
    st.subheader("Black-Scholes Equation & Heat Equation")
    st.write("Though ordinary and partial differential equations are an * integral * part of physics, plenty of other fields model systems using them. Take for example mathematical finance and the Black-Scholes equation mentioned previously:")
    st.latex(r'''
        \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial ^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rv = 0,
    ''')
    st.write("and changing variables so we obtain the heat equation:")
    st.latex(r'''
        \frac{\partial u}{\partial \tau} = \frac{1}{2} \sigma^2 \frac{\partial^2 u}{\partial x^2},
    ''')
    st.write("where: ")
    st.latex(r'''
        \tau = T - t, u = C e^{r \tau}, x = ln(\frac{S}{K}) + (r - \frac{1}{2} \sigma^2) \tau.
    ''')
    st.write("The solution to the heat equation is given below with the added Black-Scholes heat equation solution below that.")
    st.image("Images/heatLoss.png")
    st.image("Images/heatSolution.png")
    
    st.header("Alternate Approaches and Recent Results")
    st.write("The PyDEns package works great for solving differential equations, whether they be ordinary or partial, on the simpler side, say up to second order. Problems arise when moving into higher dimensions and trying to solve more complex equations, take the Navier-Stokes equation from fluid dynamics for example. Recent research, specifically this paper: https://arxiv.org/abs/2010.08895, has somewhat solved this problem by taking an entirely different approach. Instead of creating the mappings in functional or Euclidean space (your normal x, y, z plane), their neural network creates mappings in Fourier space. What does this mean though? A Fourier transform essentially takes a function and breaks it down into a summation of frequencies or sine waves. Mapping the function into Fourier space eliminates our original x, y grid and instead plots over the Fourier transform as a function of frequency and this mapping is a lot easier to solve.")
    st.write("Imagine going to a fancy restaurant and ordering an amazing steak - eventually you'll want to have it again. This may not always be possible, say the steak was very expensive or a pandemic hits and the restaurant closes for the time being, but you really want it again. The next logical step would be to recreate or *approximate* this at home. Just going off of taste alone may get you close but it just isn't the same. Now, let's say you have a recipe for the steak: the outcome will be a lot closer to the dish you had the first time. This is exactly how the researchers at Cal-Tech solved the Navier-Stokes equation. If you recall, neural networks act as function approximators for the equation and a Fourier transform is nothing but the recipe for said equation; and approximating the recipe is *much* easier than approximating the dish as a whole. ")
    st.write("The Navier-Stokes equation may not ring a bell or be of much importance to anyone outside of physics, or even more specifically, fluid dynamics as a sub-field. But, the equation is used to model weather patterns in addition to air flow around a plane or a ball. Being able to solve the Navier-Stokes equation may lead to better and more fine-tuned weather predictions on a global scale, eliminating the need to check the weather app every hour or so to get a more accurate idea of what to wear out today. ")
    
    
    
    
    
    
    
    