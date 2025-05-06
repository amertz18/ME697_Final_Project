import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap, grad
import matplotlib.pyplot as plt

PRESSURE_IN = .1
PRESSURE_OUT = 0.
RHO = 1
LENGTH = 1
DIAMETER = 0.1
EPOCHS = 10000
MAX_NU = 0.002
LEARNING_RATE = 5e-4

key = jr.PRNGKey(0)
key, subkey1, subkey2, subkey3 = jr.split(key, 4)

width_size = 128
depth = 5

N1 = eqx.nn.MLP(3, 1, width_size, depth, jnp.tanh, key=subkey1)
N2 = eqx.nn.MLP(3, 1, width_size, depth, jnp.tanh, key=subkey2)
N3 = eqx.nn.MLP(3, 1, width_size, depth, jnp.tanh, key=subkey3)

u = lambda x, y, nu, mlp: (DIAMETER**2 / 4 - y**2) * mlp(jnp.array([x, y, nu]))[0]
v = lambda x, y, nu, mlp: (DIAMETER**2 / 4 - y**2) * mlp(jnp.array([x, y, nu]))[0]
p = lambda x, y, nu, mlp: (x / LENGTH) * PRESSURE_OUT + ((LENGTH - x) / LENGTH) * PRESSURE_IN + x * (LENGTH - x) * mlp(jnp.array([x, y, nu]))[0]

u_x = grad(u, argnums = 0)
u_y = grad(u, argnums = 1)
u_xx = grad(u_x, argnums = 0)
u_yy = grad(u_y, argnums = 1)

v_x = grad(v, argnums = 0)
v_y = grad(v, argnums = 1)
v_xx = grad(v_x, argnums = 0)
v_yy = grad(v_y, argnums = 1)

p_x = grad(p, argnums = 0)
p_y = grad(p, argnums = 1)

def mass_res(mlps, x, y, nu): 
    N1, N2, _ = mlps
    return u_x(x, y, nu, N1) + v_y(x, y, nu, N2)

def mom_res_x(mlps, x, y, nu): 
    N1, N2, N3 = mlps
    term1 = u(x, y, nu, N1) * u_x(x, y, nu, N1)
    term2 = v(x, y, nu, N2) * u_y(x, y, nu, N1) 
    term3 = RHO**-1 * p_x(x, y, nu, N3)
    term4 = nu * MAX_NU * (u_xx(x, y, nu, N1) + u_yy(x, y, nu, N1))
    return term1 + term2 + term3 - term4

def mom_res_y(mlps, x, y, nu): 
    N1, N2, N3 = mlps
    term1 = u(x, y, nu, N1) * v_x(x, y, nu, N2)
    term2 = v(x, y, nu, N2) * v_y(x, y, nu, N2) 
    term3 = RHO**-1 * p_y(x, y, nu, N3)
    term4 = nu * MAX_NU * (v_xx(x, y, nu, N2) + v_yy(x, y, nu, N2))
    return term1 + term2 + term3 - term4
 
sum = lambda mlps, x, y, nu: mass_res(mlps, x, y, nu)**2 + mom_res_x(mlps, x, y, nu)**2 + mom_res_y(mlps, x, y, nu)**2
sumVec = vmap(sum, in_axes = (None, 0, 0, 0))

loss = lambda mlps, x, y, nu: jnp.mean(sumVec(mlps, x, y, nu))

def train(
        loss,
        mlps,
        key,
        optimizer,
        Lx=LENGTH,
        Ly=DIAMETER,
        #maxNu=MAX_NU,
        num_collocation_residual=32,
        num_iter=EPOCHS,
        freq=1
    ):

    @eqx.filter_jit
    def step(opt_state, mlps, xs, ys, nus):
        value, grads = eqx.filter_value_and_grad(loss)(mlps, xs, ys, nus)
        updates, opt_state = optimizer.update(grads, opt_state)
        mlps = eqx.apply_updates(mlps, updates)
        return mlps, opt_state, value

    opt_state = optimizer.init(eqx.filter(mlps, eqx.is_inexact_array))

    losses = []
    for i in range(num_iter):
        key, subkey1, subkey2, subkey3 = jr.split(key, 4)
        x = jr.uniform(subkey1, (num_collocation_residual,), maxval=Lx)
        y = jr.uniform(subkey2, (num_collocation_residual,), minval=-Ly/2., maxval=Ly/2.)
        nus = jr.uniform(subkey3, (num_collocation_residual,))
        mlps, opt_state, value = step(opt_state, mlps, x, y, nus)
        if i % freq == 0:
            losses.append(value)
            if i % 100 == 0:
                print(f'Reached epoch {i+1} of {num_iter+1}. Loss = {value:0.9f}')
    return mlps, losses

optimizer = optax.adam(LEARNING_RATE)

key, subkey = jr.split(key)
(trained_N1, trained_N2, trained_N3), losses = train(loss, (N1, N2, N3), subkey, optimizer)

x = jnp.linspace(0, LENGTH, 100)
y = jnp.linspace(-DIAMETER / 2, DIAMETER / 2, 100)
nu = jnp.array([MAX_NU / 2]) / MAX_NU
X, Y, NU = jnp.meshgrid(x, y, nu)

vec_u = eqx.filter_jit(vmap(u, in_axes=(0, 0, 0, None)))
vec_v = eqx.filter_jit(vmap(v, in_axes=(0, 0, 0, None)))
vec_p = eqx.filter_jit(vmap(p, in_axes=(0, 0, 0, None)))

u_pred = vec_u(X.flatten(), Y.flatten(), NU.flatten(), trained_N1).reshape(X.shape).squeeze()
v_pred = vec_v(X.flatten(), Y.flatten(), NU.flatten(), trained_N2).reshape(X.shape).squeeze()
p_pred = vec_p(X.flatten(), Y.flatten(), NU.flatten(), trained_N3).reshape(X.shape).squeeze()

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Residual loss')
plt.yscale('log')
plt.title('Training Loss')
plt.savefig('fig/TrainingLoss.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.contourf(X.squeeze(), Y.squeeze(), u_pred, levels = 100, cmap=plt.cm.jet)
plt.colorbar(label='Streamwise Velocity (u)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Streamwise Velocity')
plt.savefig('fig/CircularPipeFlowU.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.contourf(X.squeeze(), Y.squeeze(), v_pred, levels = 100, cmap=plt.cm.jet)
plt.colorbar(label='Spanwise Velocity (v)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spanwise Velocity')
plt.savefig('fig/CircularPipeFlowV.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.contourf(X.squeeze(), Y.squeeze(), p_pred, levels = 100, cmap=plt.cm.jet)
plt.colorbar(label='Centerline Pressure Profile (p)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Centerline Pressure Profile')
plt.savefig('fig/CircularPipeFlowP.png')
plt.close()

nu_test = [0.0019, 0.00061]
plt.figure(figsize=(10, 6))

for nu in nu_test:
    y_analytical = jnp.linspace(-DIAMETER / 2, DIAMETER / 2, 100)
    u_analytical = (PRESSURE_IN - PRESSURE_OUT) * (DIAMETER**2 / 4 - y_analytical**2) / (2 * nu * RHO * LENGTH)

    x = jnp.array([1.])
    y = jnp.linspace(-DIAMETER / 2, DIAMETER / 2, 100)
    nu = jnp.array([nu / MAX_NU])
    X, Y, NU = jnp.meshgrid(x, y, nu)

    u_pred_analytical = vec_u(X.flatten(), Y.flatten(), NU.flatten(), trained_N1).reshape(X.shape).squeeze()

    plt.plot(Y.squeeze(), u_analytical, color = 'blue')
    plt.plot(Y.squeeze(), u_pred_analytical, color = 'red', linestyle = '--')

plt.plot(0, 0, color = 'blue', label = 'Analytical Solutions')
plt.plot(0, 0, color = 'red', linestyle = '--', label = 'PINN Predictions')
plt.xlabel('y')
plt.ylabel('u(y)')
plt.title('Streamwise Velocity, nu = [0.0019, 0.00061]')
plt.legend()
plt.savefig('fig/CrossSectionVeloProfile.png')
plt.close()