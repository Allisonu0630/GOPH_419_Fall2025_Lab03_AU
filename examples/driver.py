import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate   # <-- new import

from my_python_package.freefall import (
    ode_freefall_euler,
    ode_freefall_rk4,
    compute_cd_star
)

def main():
    #  parameters
    g0 = 9.811636          #  calgary grav (m/s^2)
    dg_dz = 3.086e-6       # FA gradient ((m/s^2)/m)
    cd_star = compute_cd_star()  # drag coefficient normalized

    
    heights = [10.0, 20.0, 40.0]   
    timesteps = np.logspace(-3, -1, 10)  # dt values from 0.001 to 0.1 s

    # Storage for results
    results = {"Euler": {}, "RK4": {}}

    for method_name, solver in [("Euler", ode_freefall_euler),
                                ("RK4", ode_freefall_rk4)]:
        for H in heights:
            drop_times = []
            runtimes = []
            errors = []

            # ref solution 
            t_ref, _, _ = solver(g0, dg_dz, cd_star, H, dt=1e-4)
            t_true = t_ref[-1]

            for dt in timesteps:
                start = time.perf_counter()
                t, z, v = solver(g0, dg_dz, cd_star, H, dt)
                end = time.perf_counter()

                drop_time = t[-1]
                runtime = end - start
                rel_error = abs(drop_time - t_true) / t_true

                drop_times.append(drop_time)
                runtimes.append(runtime)
                errors.append(rel_error)

            # storing results
            results[method_name][H] = {
                "dt": timesteps,
                "drop_times": drop_times,
                "errors": errors,
                "runtimes": runtimes
            }
             

          ##### summary table ##### (COMMENTED OUT BECUASE I DEEMED IT UNNECESSARY)
             

          #  print(f"\n{method_name} | H={H} m")
           # table = zip(timesteps, drop_times, errors, runtimes)
          #  headers = ["dt (s)", "drop time (s)", "rel error", "runtime (s)"]
          #  print(tabulate(table, headers=headers,
                        #   floatfmt=[".4f", ".6f", ".2e", ".4e"]))

    # plotting figures

    # checking figures directory exists
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("___plots saved to GOPH_419_Fall2025_Lab03_AU/figures___")
    for H in heights:
        plt.figure(figsize=(12, 4))

        # Drop time vs dt
        plt.subplot(1, 3, 1)
        plt.plot(results["Euler"][H]["dt"], results["Euler"][H]["drop_times"], 'o-', label="Euler")
        plt.plot(results["RK4"][H]["dt"], results["RK4"][H]["drop_times"], 's-', label="RK4")
        plt.xscale("log")
        plt.xlabel("dt (s)")
        plt.ylabel("Drop time (s)")
        plt.title(f"Drop time vs dt (H={H} m)")
        plt.legend()

        # relative error vs dt
        plt.subplot(1, 3, 2)
        plt.plot(results["Euler"][H]["dt"], results["Euler"][H]["errors"], 'o-', label="Euler")
        plt.plot(results["RK4"][H]["dt"], results["RK4"][H]["errors"], 's-', label="RK4")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("dt (s)")
        plt.ylabel("Relative error")
        plt.title(f"Error vs dt (H={H} m)")
        plt.legend()

        # runtime vs dt
        plt.subplot(1, 3, 3)
        plt.plot(results["Euler"][H]["dt"], results["Euler"][H]["runtimes"], 'o-', label="Euler")
        plt.plot(results["RK4"][H]["dt"], results["RK4"][H]["runtimes"], 's-', label="RK4")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("dt (s)")
        plt.ylabel("Runtime (s)")
        plt.title(f"Runtime vs dt (H={H} m)")
        plt.legend()

        plt.tight_layout()

        # save figure to figures folder
        fig_name = f"H{int(H)}.png"
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=300)

        plt.close()
        

    # Sensitivity (question 3)
    alpha = 1e-2  # 1% perturbation
    for H in heights:
        print(f"\nSensitivity at H={H} m")

        # Reference drop time with RK4 (small dt for accuracy)
        t_ref, _, _ = ode_freefall_rk4(g0, dg_dz, cd_star, H, dt=1e-4)
        t_base = t_ref[-1]

        # perturb g0
        t_g0, _, _ = ode_freefall_rk4(g0 * (1 + alpha), dg_dz, cd_star, H, dt=1e-4)
        delta_t_g0 = t_g0[-1] - t_base

        # dg_dz
        t_grad, _, _ = ode_freefall_rk4(g0, dg_dz * (1 + alpha), cd_star, H, dt=1e-4)
        delta_t_grad = t_grad[-1] - t_base

        # cd_star
        t_cd, _, _ = ode_freefall_rk4(g0, dg_dz, cd_star * (1 + alpha), H, dt=1e-4)
        delta_t_cd = t_cd[-1] - t_base

        # Print results
        print(f"Δt* from 1% change in g0     = {delta_t_g0:.6f} s")
        print(f"Δt* from 1% change in g′     = {delta_t_grad:.6f} s")
        print(f"Δt* from 1% change in cD*    = {delta_t_cd:.6f} s")

if __name__ == "__main__":
    main()

