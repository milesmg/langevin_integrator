# Langevin Integrator

## Background / Motivation

I've always been interested in computational methods for solving chemistry problems, and this is just about the most basic. Since Einstein, the question of 'how do molecules move' has had various answers, but the most important ones you run into today all revolve around computational methods of discretizing the equations of motion at the molecular scale. Such discretizations run into a few major challenges
- Ergodicity/Bias: You want your system to sample all the key microstates it should in the correct proportion, and you want this to happen quickly enough that you can get there in one simulation.
- Physical Accuracy: You need your system to be well represented, particularly when it comes to respecting conservation laws. 
- Efficiency: You need to run your code on a real computer.

Here, I will work with the BAOAB implementation of Langevin dynamics (one formualation of stochastic MD). This project is based on the landmark paper 'Robust and efficient configurational molecular sampling via Langevin dynamics' by Benedict Leimkuhler and Charles Matthews. 

## Notes

One of the key dilemmas the authors describe is choosing a step size for your numerical integrator. Too large a step size will make the integrator inaccurate, but too small a step size slows down your sampling. 