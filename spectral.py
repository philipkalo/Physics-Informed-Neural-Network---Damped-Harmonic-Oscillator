import os
os.environ["KERAS_BACKEND"] = "tensorflow"
 
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

m, mu, k = 1.0, 0.1, 100.0 

u0, v0 = 1.0, 0.0

delta  = mu / (2 * m)
omega0 = np.sqrt(k / m)
omega  = np.sqrt(omega0**2 - delta**2)
print(f"omega = {omega:.2f}  (~{omega*10/(2*np.pi):.0f} cycles over [0,10])")
 
def exact(t):
    t = np.array(t)
    A = u0
    B = (v0 + delta * u0) / omega
    return np.exp(-delta * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))
 
def derivatives(model, t):
    with tf.GradientTape() as tape2:
        tape2.watch(t)
        with tf.GradientTape() as tape1:
            tape1.watch(t)
            u = model(t)
        u_t = tape1.gradient(u, t)
    u_tt = tape2.gradient(u_t, t)
    return u, u_t, u_tt

class FourierFeatures(keras.layers.Layer):
    def __init__(self, num_features=64, sigma = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.sigma = sigma

    def build(self, input_shape):
        self.B = self.add_weight(
            shape=(input_shape[-1], self.num_features),
            initializer = keras.initializers.RandomNormal(stddev=self.sigma),
            trainable = False,
            name = "B"
        )

    def call(self, x):
        proj = 2.0 * np.pi * tf.matmul(x, self.B)
        concat = tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)
        return concat

def build_model(width=64, depth=4, fourier=False, sigma=5.0, nfeat=64):
    inp = keras.Input(shape=(1,))
    x = keras.layers.Rescaling(scale = 0.2, offset = -1.0)(inp)
    if fourier:
        x = FourierFeatures(num_features=nfeat, sigma=sigma)(x)
    for _ in range(depth):
        x = keras.layers.Dense(width, activation="tanh")(x)
    return keras.Model(inp, keras.layers.Dense(1)(x))

t_test = np.linspace(0, 10, 2000)[:, None].astype("float32")
u_true = exact(t_test).flatten()

def train_pinn(fourier=False, sigma = 5.0, n_col=500, epochs = 15000,
                lambda_ic = 100.0, seed=0, lr=1e-3):
    keras.utils.set_random_seed(seed)
    t_phys = tf.constant(np.linspace(0, 10, n_col)[:, None], dtype=tf.float32)
    t_ic = tf.constant([[0.0]], dtype=tf.float32)
    model = build_model(fourier=fourier, sigma=sigma)
    opt = keras.optimizers.Adam(lr)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            u, u_t, u_tt = derivatives(model, t_phys)
            physics = tf.reduce_mean(tf.square(m*u_tt + mu*u_t + k*u))
            u_ic, u_t_ic, _ = derivatives(model, t_ic)
            ic = tf.reduce_mean(tf.square(u_ic - u0)) \
               + tf.reduce_mean(tf.square(u_t_ic - v0))
            loss = physics + lambda_ic * ic
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                model.trainable_variables))
        return loss
 
    for i in range(epochs):
        L = step()
        if i % 3000 == 0:
            print(f"  step {i:5d} | loss {float(L):.3e}")
 
    u_pred = model(tf.constant(t_test)).numpy().flatten()
    rel_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    return rel_l2, u_pred

print("\n=== Part 1: plain MLP at high frequency ===")
err_plain, pred_plain = train_pinn(fourier=False)
print(f"plain MLP  : relative L2 = {err_plain:.3e}   (expect ~0.9 -> FAILS)")
 
plt.figure()
plt.plot(t_test, u_true, label="exact")
plt.plot(t_test, pred_plain, "--", label="plain PINN")
plt.xlabel("t"); plt.ylabel("u(t)"); plt.legend()
plt.title(f"Spectral bias resulting in MLP failure (rel L2 = {err_plain:.2f})")
plt.savefig("figs/spectral_failure.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n=== Part 3: Fourier features at high frequency ===")
err_fourier, pred_fourier = train_pinn(fourier=True, sigma=5.0)
print(f"Fourier    : relative L2 = {err_fourier:.3e}   (expect ~1e-2 -> WORKS)")
print(f"improvement factor: {err_plain / err_fourier:.0f}x")
 
plt.figure()
plt.plot(t_test, u_true, label="exact")
plt.plot(t_test, pred_fourier, "--", label="Fourier PINN")
plt.xlabel("t"); plt.ylabel("u(t)"); plt.legend()
plt.title(f"Fourier features fix it (rel L2 = {err_fourier:.1e})")
plt.savefig("figs/spectral_fixed.png", dpi=150, bbox_inches="tight")
plt.show()