import jax
from jax import numpy as jnp

from jax import config
config.update("jax_enable_x64", True)

"""
Physics functions
- assumes the coordinates of all 2D grids are y,x
i.e. c[2][x] and c[1][x] have distance dy, c[y][2] and c[y][1] have distance dx
"""


"""
Upwind advection scheme using finite volume method
field: (ny, nx), field to be advected
velocity: (2,ny,nx), where velocity[0],velocity[1] represents the y,x velocity (v,u) respectively
dx, dy: lengths along x and y axes
"""
def advect_fvm(field: jnp.array, velocity: jnp.array, dx: float, dy: float):
    vel_u = velocity[1]
    vel_v = velocity[0]

    vel_u_stagger_x = stagger(field=vel_u, fn=mean)[1]                          #(ny, nx-1)
    vel_v_stagger_y = stagger(field=vel_v, fn=mean)[0]                          #(ny-1, nx)

    # where velocity is positive, take field value from lower coordinate
    vel_u_x_positive = (vel_u_stagger_x > 0)                                    #(ny, nx-1)
    c_x = vel_u_x_positive * field[:,:-1] + (1-vel_u_x_positive) * field[:,1:]

    vel_v_y_positive = (vel_v_stagger_y > 0)                                    #(ny-1, nx)
    c_y = vel_v_y_positive * field[:-1,:] + (1-vel_v_y_positive) * field[1:,:]

    x_comp = vel_u_stagger_x * c_x
    y_comp = vel_v_stagger_y * c_y
    advection_term = -stagger_divergence(x_comp, y_comp, dx, dy)
    return advection_term                                                       #(ny,nx)

"""
Upwind advection scheme using finite volume method with velocity correction
field: (ny, nx), field to be advected
velocity: (2,ny,nx), where velocity[0],velocity[1] represents the y,x velocity (v,u) respectively
u_corr: (ny, nx-1)
v_corr: (ny-1, nx)
dx, dy: lengths along x and y axes
"""
def advect_corr_fvm(field: jnp.array, velocity: jnp.array, u_corr: jnp.array, v_corr: jnp.array, dx: float, dy: float):
    vel_u = velocity[1]
    vel_v = velocity[0]

    vel_u_stagger_x = stagger(field=vel_u, fn=mean)[1]-u_corr                   #(ny, nx-1)
    vel_v_stagger_y = stagger(field=vel_v, fn=mean)[0]-v_corr                   #(ny-1, nx)

    # where velocity is positive, take field value from lower coordinate
    vel_u_x_positive = (vel_u_stagger_x > 0)                                    #(ny, nx-1)
    c_x = vel_u_x_positive * field[:,:-1] + (1-vel_u_x_positive) * field[:,1:]

    vel_v_y_positive = (vel_v_stagger_y > 0)                                    #(ny-1, nx)
    c_y = vel_v_y_positive * field[:-1,:] + (1-vel_v_y_positive) * field[1:,:]

    x_comp = vel_u_stagger_x * c_x
    y_comp = vel_v_stagger_y * c_y
    advection_term = -stagger_divergence(x_comp, y_comp, dx, dy)
    return advection_term   


"""
Diffusion scheme using finite volume method
field: (ny, nx), field to be advected
diffusivity: (ny,nx)
dx, dy: lengths along x and y axes
"""
def diffuse_fvm(field: jnp.array, diffusivity: jnp.array, dx: float, dy: float):
    diffusivity_y, diffusivity_x = stagger(field=diffusivity, fn=min) #(ny-1, nx), (ny, nx-1)

    dudy, dudx = stagger_gradient(field, dx, dy)                      #(ny-1, nx), (ny, nx-1)

    x_comp = diffusivity_x * dudx                                     #(ny, nx-1)
    y_comp = diffusivity_y * dudy                                     #(ny-1, nx)

    diffusion_term = stagger_divergence(x_comp=x_comp, y_comp=y_comp, dx=dx, dy=dy)
    return diffusion_term                                             #(ny,nx)

"""
Diffusion scheme using finite volume method
field: (ny, nx), field to be advected
diffusivity_x: (ny,nx-1)
diffusivity_y: (ny-1,nx)
dx, dy: lengths along x and y axes
"""
def diffuse_2d_fvm(field: jnp.array, diffusivity_x: jnp.array, diffusivity_y: jnp.array, dx: float, dy: float):
    dudy, dudx = stagger_gradient(field, dx, dy)                      #(ny-1, nx), (ny, nx-1)

    x_comp = diffusivity_x * dudx                                     #(ny, nx-1)
    y_comp = diffusivity_y * dudy                                     #(ny-1, nx)

    diffusion_term = stagger_divergence(x_comp=x_comp, y_comp=y_comp, dx=dx, dy=dy)
    return diffusion_term                                             #(ny,nx)
"""
Helper functions
"""

"""
Given field: (ny, nx), returns two staggered fields along two dimensions.

Face centered values is given by fn(x,y), where x,y is the value in the cell with lower,higher coordinate respectively

Returns staggered_y: (ny-1, nx), staggered_x: (ny,nx-1)
"""
def stagger(field: jnp.array, fn):
    lower_x = field[:, :-1]                           #(ny, nx-1)
    upper_x = field[:, 1:]                            #(ny, nx-1)

    lower_y = field[:-1, :]                           #(ny-1, nx)
    upper_y = field[1:, :]                            #(ny-1, nx)
    return fn(lower_y, upper_y), fn(lower_x, upper_x) #(ny-1, nx), (ny, nx-1)

def mean(x,y):
    return (x+y)/2

def min(x,y):
    return jnp.minimum(x,y)

"""
Computes divergence of staggered grid, by computing net outward flux per volume
y_comp: (ny-1, nx), x_comp: (ny,nx-1)
returns div: (ny, nx)
"""
def stagger_divergence(x_comp, y_comp, dx, dy):
    nx = y_comp.shape[1]
    ny = x_comp.shape[0]

    zero_col = jnp.zeros((ny,1))
    div_x = jnp.concatenate((x_comp, zero_col), axis=1) - jnp.concatenate((zero_col, x_comp), axis=1)

    zero_row = jnp.zeros((1,nx))
    div_y = jnp.concatenate((y_comp, zero_row), axis=0) - jnp.concatenate((zero_row, y_comp), axis=0)

    div = div_x/dx + div_y/dy
    return div

"""
Computes spatial gradient of field:(ny, nx) as staggered grid
returns gradient_y: (ny-1, nx), gradient_x: (ny, nx-1)
"""
def stagger_gradient(field, dx: float, dy: float):
    lower_x = field[:, :-1]                 #(ny, nx-1)
    upper_x = field[:, 1:]                  #(ny, nx-1)

    lower_y = field[:-1, :]                 #(ny-1, nx)
    upper_y = field[1:, :]                  #(ny-1, nx)

    gradient_x = (upper_x - lower_x)/dx     #(ny, nx-1)
    gradient_y = (upper_y - lower_y)/dy     #(ny-1, nx)

    return gradient_y, gradient_x           #(ny-1, nx), (ny, nx-1)

def centre_of_mass(grid: jnp.array):
    y_size, x_size = grid.shape
    x_coordinates = jnp.array([list(range(x_size)) for _ in range(y_size)])
    y_coordinates = jnp.array([list(range(y_size)) for _ in range(x_size)]).T
    grid_sum = jnp.sum(grid)

    x = jnp.sum(x_coordinates * grid) / grid_sum
    y = jnp.sum(y_coordinates * grid) / grid_sum

    return jnp.array([y,x])
