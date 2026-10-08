import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import scipy.optimize

# ============================================================================
# Foundations (reused from Module 1)
# mu = 0.5 is the GROUND TRUTH. It is used ONLY by exact() to fabricate the
# synthetic measurements. The PINN never sees it — it only sees the noisy data.
# ============================================================================
m, mu, k = 1.0, 0.5, 4.0
u0, v0 = 1.0, 0.0

delta  = mu / (2 * m)
omega0 = np.sqrt(k / m)
omega  = np.sqrt(omega0**2 - delta**2)

def exact(t):
    t = np.array(t)
    A = u0
    B = (v0 + delta * u0) / omega
    return np.exp(-delta * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))

def build_model(width=32, depth=3):
    inp = keras.Input(shape=(1,))
    x = keras.layers.Rescaling(scale=0.2, offset=-1.0)(inp)
    for _ in range(depth):
        x = keras.layers.Dense(width, activation="tanh")(x)
    out = keras.layers.Dense(1)(x)
    return keras.Model(inp, out)

def derivatives(model, t):
    with tf.GradientTape() as tape2:
        tape2.watch(t)
        with tf.GradientTape() as tape1:
            tape1.watch(t)
            u = model(t)
        u_t = tape1.gradient(u, t)
    u_tt = tape2.gradient(u_t, t)
    return u, u_t, u_tt

t_phys      = tf.constant(np.linspace(0, 10, 200)[:, None], dtype=tf.float32)
lambda_phys = 1.0

# ============================================================================
# The whole inverse experiment, wrapped so we can run it at different settings.
# Fresh model AND fresh mu_var every call -> each call is an independent,
# fair experiment. Returns the recovered mu.
# ============================================================================
def run_inverse(n_data, noise_std, mu0=2.0, epochs=15000, seed=0):
    """ONE experiment, ONE initial guess. Returns (recovered_mu, mu_history)."""
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    t_data_np = np.sort(rng.uniform(0, 10, n_data))[:, None]
    u_data_np = exact(t_data_np) + noise_std * rng.standard_normal((n_data, 1))
    t_data = tf.constant(t_data_np, dtype=tf.float32)
    u_data = tf.constant(u_data_np, dtype=tf.float32)

    model  = build_model()                        # FRESH every call
    mu_var = tf.Variable(mu0, dtype=tf.float32)   # FRESH, at the requested guess
    variables = model.trainable_variables + [mu_var]
    opt = keras.optimizers.Adam(1e-3)

    def compute_loss():
        data_loss = tf.reduce_mean(tf.square(model(t_data) - u_data))
        u, u_t, u_tt = derivatives(model, t_phys)
        residual = m*u_tt + mu_var*u_t + k*u
        return data_loss + lambda_phys * tf.reduce_mean(tf.square(residual))

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            L = compute_loss()
        opt.apply_gradients(zip(tape.gradient(L, variables), variables))

    mu_history = []
    for i in range(epochs):
        step()
        if i % 300 == 0:
            mu_history.append(float(mu_var.numpy()))

    
    trainable = model.trainable_variables + [mu_var]
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
            L = compute_loss()
        g = tape.gradient(L, trainable)
        gflat = tf.concat([tf.reshape(x, [-1]) for x in g], 0)
        return L.numpy().astype(np.float64), gflat.numpy().astype(np.float64)

    mu_adam = float(mu_var.numpy())
    init = tf.concat([tf.reshape(v, [-1]) for v in trainable], 0).numpy().astype(np.float64)
    res = scipy.optimize.minimize(objective, init, jac=True, method="L-BFGS-B",
                                  options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-14})
    assign_params(res.x)                     # <-- load the optimum back into the model

    print(f"  mu: Adam {mu_adam:.5f} -> +L-BFGS {float(mu_var.numpy()):.5f}")
    return float(mu_var.numpy()), mu_history          # <-- OUTSIDE the loop

# ---- Stretch 3: the sweep lives OUT here, at module level ----
plt.figure()
for mu0 in [0.0, 1.0, 5.0]:
    mu_rec, hist = run_inverse(20, 0.03, mu0=mu0)
    print(f"mu0 = {mu0:>4} -> recovered mu = {mu_rec:.4f}")
    plt.plot(np.arange(len(hist))*300, hist, label=f"μ₀ = {mu0}")
plt.axhline(0.5, ls="--", color="k", label="true μ")
plt.xlabel("training step"); plt.ylabel("μ estimate"); plt.legend(); plt.show()

    # ============================================================================
    # Task 5 — headline run (with plots)
    # ============================================================================
mu_rec, hist = run_inverse(n_data=20, noise_std=0.03)
print(f"\nrecovered mu = {mu_rec:.4f}   (true 0.5)\n")

    # ============================================================================
    # Task 6 — robustness sweeps
    # (each call is a fresh experiment. ~15k epochs x 8 runs takes a few minutes;
    #  drop `epochs=` lower if you just want the trend quickly.)
    # ============================================================================
print("noise sweep (20 points):")
for ns in [0.0, 0.03, 0.1, 0.2]:
    mu, _ = run_inverse(20, ns)
    print(f"  noise {ns:>4} -> mu {run_inverse(20, ns):.4f}")

print("data-count sweep (noise 0.05):")
for nd in [5, 10, 20, 40]:
    mu_r, _ = run_inverse(nd, 0.05)
    print(f"  {nd:>2} pts -> mu {run_inverse(nd, 0.05):.4f}")