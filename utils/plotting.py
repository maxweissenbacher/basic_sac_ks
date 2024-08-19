import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from solver.ks_solver import KS
import torch


def contourplot_KS(uu, dt=0.5, plot_frame=True, frameskip=1, filename=''):
    # Make contour plot of solution
    fig, ax = plt.subplots()
    tt = np.arange(uu.shape[0]) * dt
    N = uu.shape[1]
    x = np.arange(0, 2 * np.pi, 2 * np.pi / N)
    ct = ax.contourf(x, tt[::frameskip], uu[::frameskip],
                     61,
                     #extend='both',
                     cmap=cm.RdBu,
                     vmin=-2.5,
                     vmax=2.5)
    if plot_frame:
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        # ax.colorbar()
        ax.set_title('Solution of the KS equation')
    else:
        ax.axis('off')
    fig.colorbar(ct)
    # plt.savefig(f'ks_plot_{filename}.png', bbox_inches = 'tight', dpi=3000)
    plt.show()


if __name__ == '__main__':

    N = 256  # number of collocation points
    dt = 0.01 # timestep size
    nu = 0.08156697852139966
    actuator_locs = torch.tensor(np.linspace(0.0, 2*np.pi, num=3, endpoint=False))
    ks = KS(nu=nu, N=N, dt=dt, actuator_locs=actuator_locs, actuator_scale=0.2)

    # Random initial data
    u = 1e-2 * np.random.normal(size=N)  # noisy intial data
    u = u - u.mean()
    u = torch.tensor(u)

    action = torch.zeros(ks.num_actuators)
    action1 = torch.ones(ks.num_actuators)
    action2 = torch.tensor(np.random.uniform(size=ks.num_actuators), dtype=torch.float32)

    # Burn-in
    burnin = 0
    for _ in range(burnin):
        u = ks.advance(u, action)
    # Advance solver
    uu = []
    for _ in range(1000):
        u = ks.advance(u, action)
        uu.append(u.detach().numpy())
    uu = np.array(uu)

    contourplot_KS(uu, dt=dt, plot_frame=True, frameskip=1, filename=f'{nu:.3f}')

