import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import scipy.optimize


m, mu, k = 1.0, 0.5, 4.0
u0, v0 = 1.0, 0.0

delta = mu / (2*m)
omega0 = np.sqrt(k/m)
omega = np.sqrt(omega0**2 - delta**2)

def exact(t):
    t = np.array(t)
    A = u0
    B = (v0 + delta*u0)/omega
    return np.exp(-delta*t) *(A*np.cos(omega*t) + B*np.sin(omega*t))

t_plot = np.linspace(0, 10, 400)
plt.plot(t_plot, exact(t_plot)); plt.xlabel("t"); plt.ylabel("u(t)"); plt.show()

def build_model(width=32, depth=3):
    inp = keras.Input(shape=(1,))
    x = keras.layers.Rescaling(scale = 0.2, offset=-1.0)(inp)
    for _ in range(depth):
        x = keras.layers.Dense(width, activation="tanh")(x)

    out = keras.layers.Dense(1)(x)
    return keras.Model(inp, out)

model = build_model()
model.summary()

def derivatives(model, t):
    with tf.GradientTape() as tape2:
        tape2.watch(t)
        with tf.GradientTape() as tape1:
            tape1.watch(t)
            u = model(t)
        u_t = tape1.gradient(u, t)
    u_tt = tape2.gradient(u_t, t)
    return u, u_t, u_tt

t = tf.constant([[1.0], [2.0]])
u, u_t, u_tt = derivatives(model, t)
finite_difference = (model(t+1e-3) - model(t-1e-3)) / 2e-3
print("autodiff u_t:", u_t.numpy().flatten())
print(f"Finite difference approximation of u_t: {finite_difference.numpy().flatten()}")

t_phys = tf.constant(np.linspace(0, 10, 200)[:, None], dtype=tf.float32)
t_ic = tf.constant([[0.0]], dtype=tf.float32)

lambda_ic = 1.0

def compute_loss(model):
    u, u_t, u_tt = derivatives(model, t_phys)
    residual = m*u_tt + mu*u_t + k*u
    physics_loss = tf.reduce_mean(tf.square(residual))

    u_ic, u_t_ic, _ = derivatives(model, t_ic)
    ic_loss = tf.reduce_mean(tf.square(u_ic - u0)) + tf.reduce_mean(tf.square(u_t_ic - v0))
    total = physics_loss + lambda_ic *ic_loss
    return total, physics_loss, ic_loss

total_loss, physics_loss, ic_loss = compute_loss(model)
print(f"Total loss: {total_loss.numpy():.6f}, Physics loss: {physics_loss.numpy():.6f}, IC loss: {ic_loss.numpy():.6f}")

opt = keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        total, phys, ic = compute_loss(model)
    grads = tape.gradient(total, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return total, phys, ic

EPOCHS = 20000
for step in range(EPOCHS):
    total, phys, ic = train_step()
    if step % 1000 == 0:
        print(f"Step {step}, Total loss: {total.numpy():.6f}, Physics loss: {phys.numpy():.6f}, IC loss: {ic.numpy():.6f}")

t_test = np.linspace(0, 10, 400)[:, None].astype("float32")
u_pred = model(tf.constant(t_test)).numpy().flatten()
u_true = exact(t_test).flatten()

def make_lbfgs_objective(model, compute_loss):
    trainable = model.trainable_variables
    shapes = [v.shape for v in trainable]
    sizes = [int(np.prod(shape)) for shape in shapes]

    def assign_params(flat):
        flat = tf.constant(flat, dtype=tf.float32); i=0
        for v, s, sh in zip(trainable, sizes, shapes):
            v.assign(tf.reshape(flat[i:i+s], sh))
            i += s
        
    def objective(flat):
        assign_params(flat)
        with tf.GradientTape() as tape:
            L, _, _ = compute_loss(model)
        g = tape.gradient(L, trainable)
        gflat = tf.concat([tf.reshape(x, [-1]) for x in g], 0)
        return L.numpy().astype(np.float64), gflat.numpy().astype(np.float64)
    
    init = tf.concat([tf.reshape(v, [-1]) for v in trainable], 0).numpy().astype(np.float64)
    return objective, init, assign_params

objective, init, assign_params = make_lbfgs_objective(model, compute_loss)
res = scipy.optimize.minimize(objective, init, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-12})

assign_params(res.x)

rel_l2 = np.linalg.norm(u_pred - u_true)/np.linalg.norm(u_true)
print("relative L2 error:", rel_l2)

plt.plot(t_test, u_true, label="exact")
plt.plot(t_test, u_pred, "--", label="predicted")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.legend()
plt.show()

t_dense = tf.constant(np.linspace(0, 10, 500)[:, None], dtype=tf.float32)
u, u_t, u_tt = derivatives(model, t_dense)
r = m*u_tt + mu*u_t + k*u

plt.semilogy(t_dense.numpy().flatten(), np.abs(r.numpy()).flatten())
plt.xlabel("t"); plt.ylabel("|residual| (log scale)")
plt.title("Where the PINN violates the physics"); plt.show()

def train_and_eval(width, depth, epochs=20000, seed=0):
    keras.utils.set_random_seed(seed)
    model = build_model(width, depth)
    opt = keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            total, _, _ = compute_loss(model)
            grads = tape.gradient(total, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            return total
        
    for _ in range(epochs):
        step()

    u_pred = model(tf.constant(t_test)).numpy().flatten()
    rel_l2 = np.linalg.norm(u_pred - u_true)/np.linalg.norm(u_true)
    return rel_l2, model.count_params()

for (w, d) in [(16,2), (32, 3), (64, 4)]:
    err, n = train_and_eval(w, d)
    print(f"width {w:2d} | depth {d} | params {n:4d} | rel L2 {err:.2e}")
