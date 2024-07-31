# Closure-for-Convection-Diffusion

## Summary

Given a 2D plane slice of a 3D scalar field generated using CFD at different flow times, the goal is to find a low-fidelity approach to perform transient 2D simulation that is similar to the extracted 2D plane slice.

![image](https://github.com/user-attachments/assets/89c9c0f0-9450-4c44-9dbf-f2b666ba221f)

The flow on the 2D plane-slice is modelled using as a convection-diffusion process with equation $\frac{dc}{dt} = \nabla \cdot (D\nabla c -\vec{v}c)+R$. Here $c, D, \vec{v}, R$ represents the concentration, diffusivity, velocity and inflow terms respectively. Starting from an initial concentration field, future concentration fields can be simulated numerically (refer to numerical_method) using the equation assuming all parameters are known. A suitable set of parameters need to be learnt such that this can be done.

## Training

The differential physics approach is adopted to train the parameters [cite]. The loss between the desired concentration field and the simulated concentration field using a set of parameters is used to update the parameters using the automatic-differentiation function in various numerical frameworks.

![image](https://github.com/user-attachments/assets/d68ee647-76b5-4811-b0d7-018f74b73252)

The training process is separated into three phases differing in terms of which parameters are being updated.

### Phase 1: Physical Parameters

Parameters with physical significance include the diffusivity and velocity. The physical parameters are expected to depend on the physical setup. For instance, a physical wall in the CFD simulation can be interpreted as a cell on the numerical grid with low diffusivity. There are four sets of physical parameters to be learnt: the two components of diffusivity and velocity respectively. The new convection-diffusion equation is as follows:

$$
\frac{dc}{dt} = \nabla \cdot (D_x \frac{\partial^2 c}{\partial x^2} + D_y \frac{\partial^2 c}{\partial y^2} -(\vec{v}-\vec{v}_{corr})c) + R
$$

### Phase 2: Neural Network Closure

Taking a 2D plane-slice of a 3D flow field will inevitably lead to loss of information which cannot be fully captured by the numerical simuation. To account for flow in the 3rd dimension as well as numerical erors, a neural network is used to account for the difference between the desired and simulated concentration field. 

As this term likely relies on concentration field on previous time steps, a recurrent neural network is used. 

![image](https://github.com/user-attachments/assets/dbb6b0c3-5549-4bed-8e43-bb14bf08b177)

$$\begin{aligned}
c^{t+1}&=c^t+A(c^t)+D(c^t)+R(c^t)+F_1(c^t, A(c^t), D(c^t), T, h^t)\\
h^{t+1}&=F_2(c^{t+1}, h^t)
\end{aligned}$$

$h$ contains the information of history of the flow, which is used to update the concentration field, which then used to update the history.

### Phase 3: Combined Training


