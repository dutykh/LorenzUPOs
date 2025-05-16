#!/usr/bin/env python3
"""
Periodic orbit finder for the Lorenz system using automatic differentiation.

This script locates periodic orbits of the Lorenz system by minimizing a differentiable
objective function with an automatic-differentiation-based optimizer.

Author: Dr. Denys Dutykh
        Khalifa University of Science and Technology
        Abu Dhabi, UAE

Date: May 16, 2025

License: GNU Lesser General Public License version 3.0 (LGPL-3.0)
         See the LGPL-3.0 file for details.
"""

import argparse
import time
import torch
import numpy as np
from torchdiffeq import odeint
from scipy.spatial import cKDTree

# Set default dtype for numerical accuracy
torch.set_default_dtype(torch.float64)

# Ensure deterministic results
torch.manual_seed(1)

# Global constants for the Lorenz system
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

# Optimization constants (defaults)
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-9
DEFAULT_LAMBDA = 1e-3
DEFAULT_EPS = 1e-12
DEFAULT_MAX_ITER = 500

def lorenz_rhs(t, state):
    """
    Right-hand side of the Lorenz system.
    
    Args:
        t: Time (unused but required by odeint interface)
        state: State vector [x, y, z]
    
    Returns:
        Tensor of shape (3,) containing [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z
    return torch.stack([dx, dy, dz])

def flow_map(x0_vec, logT, t_min=1.0):
    """
    Compute the time-T flow map of the Lorenz system.
    
    Args:
        x0_vec: Initial state vector [x0, y0, z0]
        logT: Logarithm of integration time ratio (T = t_min * exp(logT))
        t_min: Minimum allowed period (default: 1.0)
    
    Returns:
        Final state after integrating from 0 to T
    """
    # Ensure T > t_min using T = t_min * exp(logT)
    T = t_min * torch.exp(logT)
    t_span = torch.stack([torch.zeros_like(T), T])
    
    # Integrate the Lorenz system
    sol = odeint(lorenz_rhs, x0_vec, t_span, method='dopri5', 
                 rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    
    # Return only the final state
    return sol[-1]

def cost(u_vec, lambda_=DEFAULT_LAMBDA, eps=DEFAULT_EPS, t_min=1.0):
    """
    Cost function for periodic orbit optimization.
    
    The cost consists of:
    1. Periodicity residual: ||flow(T) - initial_state||²
    2. Penalty term to avoid equilibria: λ/(||f(x0)||² + ε)
    
    Args:
        u_vec: Vector [x0, y0, z0, logT] with gradient tracking
        lambda_: Penalty weight
        eps: Small constant to avoid division by zero
        t_min: Minimum allowed period (default: 1.0)
    
    Returns:
        Total cost value
    """
    # Split the unknown vector
    pos = u_vec[:3]  # [x0, y0, z0]
    logT = u_vec[3]  # log(T/t_min)
    
    # Compute periodicity residual
    final_state = flow_map(pos, logT, t_min=t_min)
    residual = final_state - pos
    residual_cost = 0.5 * residual.pow(2).sum()
    
    # Compute penalty to avoid equilibria
    f_at_pos = lorenz_rhs(0.0, pos)
    penalty = lambda_ / (f_at_pos.pow(2).sum() + eps)
    
    return residual_cost + penalty

def initial_guess_po(
    T_final=100.0,
    n_steps=10000,
    t_min=5.0,  # Minimum time separation
    t_max=10.0,  # Maximum time separation
    x_bounds=(-20, 20),
    y_bounds=(-30, 30),
    z_bounds=(0, 50),
    seed=None
):
    """
    Generate a heuristic initial guess for a periodic orbit (PO) of the Lorenz system.
    Uses k-d tree for efficient nearest neighbor search.
    Returns a dict with:
      'T0': time lag between twin states,
      'x0','y0','z0': components of the earlier state,
      'min_dist': the minimum distance between twin states found.
    """
    if seed is not None:
        np.random.seed(seed)

    x0_rand = np.random.uniform(*x_bounds)
    y0_rand = np.random.uniform(*y_bounds)
    z0_rand = np.random.uniform(*z_bounds)

    t_eval = torch.linspace(0.0, T_final, n_steps)
    state0 = torch.tensor([x0_rand, y0_rand, z0_rand], dtype=torch.float64)
    traj_torch = odeint(lorenz_rhs, state0, t_eval, method='dopri5', rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    traj = traj_torch.detach().cpu().numpy()  # shape (n_steps, 3)
    t_numpy = t_eval.numpy()

    # Convert time bounds to index bounds
    dt = t_numpy[1] - t_numpy[0]
    min_steps = int(t_min / dt)
    max_steps = int(t_max / dt)
    
    min_dist = float('inf')
    best = None
    
    # Process trajectory in windows for better performance
    window_size = max_steps - min_steps + 100  # Add some padding
    
    for i_start in range(0, n_steps, window_size // 2):
        i_end = min(i_start + window_size, n_steps)
        
        # Build k-d tree for this window
        window_traj = traj[i_start:i_end]
        tree = cKDTree(window_traj)
        
        # For each point in the first half of the window
        for i_local in range(min(window_size // 2, i_end - i_start)):
            i_global = i_start + i_local
            
            # Find k nearest neighbors 
            k = min(50, i_end - i_start - i_local)
            if k < 2:
                continue
                
            distances, indices = tree.query(window_traj[i_local], k=k)
            
            for dist, j_local in zip(distances[1:], indices[1:]):  # Skip self (index 0)
                j_global = i_start + j_local
                
                # Calculate time separation
                dt = t_numpy[j_global] - t_numpy[i_global]
                
                # Skip if outside time bounds
                if dt < t_min or dt > t_max:
                    continue
                
                if dist < min_dist:
                    min_dist = dist
                    best = (i_global, j_global)

    if best is None:
        raise RuntimeError(f"No sufficiently similar state pairs found with t_min={t_min}, t_max={t_max}")

    i0, j0 = best
    T0 = t_numpy[j0] - t_numpy[i0]
    x0, y0, z0 = traj[i0]
    
    print(f"Best twin states found: distance={min_dist:.6f}, T={T0:.4f}")
    
    return {'T0': T0, 'x0': x0, 'y0': y0, 'z0': z0, 'min_dist': min_dist}

def main(args):
    """
    Main optimization routine for finding periodic orbits.
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    
    # Get initial guess
    guess = initial_guess_po()
    # Convert T to log(T/t_min) parameterization
    t_min = 1.0
    logT_init = torch.log(torch.tensor(guess['T0'] / t_min))
    u_init = torch.tensor([guess['x0'], guess['y0'], guess['z0'], logT_init], 
                          dtype=torch.float64, requires_grad=True)
    
    print(f"Initial guess: x0={u_init[0]:.6f}, y0={u_init[1]:.6f}, "
          f"z0={u_init[2]:.6f}, T={guess['T0']:.6f}")
    
    # Note about constraint
    print(f"Note: T = {t_min} * exp(logT) ensures T > {t_min}")
    
    # Optional: Pre-optimization with Adam
    if not args.skip_adam:
        print("\nPre-optimization with Adam...")
        u_vec = u_init.clone().detach().requires_grad_(True)
        adam = torch.optim.Adam([u_vec], lr=1e-2)
        
        for i in range(2000):
            adam.zero_grad()
            loss = cost(u_vec, lambda_=args.lambda_, t_min=1.0)
            loss.backward()
            adam.step()
            
            if i % 100 == 0:  # Reduced frequency for cleaner output
                with torch.no_grad():
                    pos = u_vec[:3]
                    logT = u_vec[3]
                    # Compute actual T value
                    T_actual = t_min * torch.exp(logT)
                    residual = flow_map(pos, logT, t_min=t_min) - pos
                    residual_norm = residual.norm()
                print(f"Adam step {i}: cost={loss:.6e}, residual_norm={residual_norm:.6e}, "
                      f"T={T_actual:.6f}, x0={pos[0]:.6f}, y0={pos[1]:.6f}, z0={pos[2]:.6f}")
    else:
        print("\nSkipping Adam pre-optimization (--skip-adam flag set)")
        u_vec = u_init.clone().detach().requires_grad_(True)
    
    # Main optimization with L-BFGS
    print("\nMain optimization with L-BFGS...")
    u_vec = u_vec.detach().requires_grad_(True)
    
    lbfgs = torch.optim.LBFGS([u_vec], 
                             max_iter=args.max_iter,
                             tolerance_grad=1e-12,
                             tolerance_change=1e-12,
                             history_size=25,
                             line_search_fn='strong_wolfe')
    
    iteration = 0
    prev_cost = float('inf')
    converged = False
    
    def closure():
        nonlocal iteration, prev_cost, converged
        
        lbfgs.zero_grad()
        
        # Split the unknown vector and compute residual first
        pos = u_vec[:3]
        logT = u_vec[3]
        
        # Compute residual and its norm (without gradients for efficiency)
        with torch.no_grad():
            T_actual = t_min * torch.exp(logT)
            residual = flow_map(pos, logT, t_min=t_min) - pos
            residual_norm = residual.norm().item()
        
        # Now compute loss with gradients
        loss = cost(u_vec, lambda_=args.lambda_, t_min=1.0)
        loss.backward()
        
        # Display iteration info
        with torch.no_grad():
            print(f"L-BFGS iteration {iteration}:")
            print(f"  Cost: {loss:.12e}")
            print(f"  Residual norm: {residual_norm:.12e}")
            print(f"  Position: x0={pos[0]:.6f}, y0={pos[1]:.6f}, z0={pos[2]:.6f}")
            print(f"  Period T: {T_actual:.6f}")
            
            # Check convergence using multiple criteria
            loss_value = loss.item()
            cost_change = abs(loss_value - prev_cost)
            
            # Primary convergence check: residual norm should be very small for good periodicity
            if residual_norm < 1e-10 or (residual_norm < 1e-8 and cost_change < 1e-12):
                converged = True
                print(f"Converged! Residual norm: {residual_norm:.3e}")
                if residual_norm > 1e-10:
                    print(f"  (Cost change: {cost_change:.3e} < 1e-12)")
            
            prev_cost = loss_value
        
        iteration += 1
        
        return loss
    
    # Run L-BFGS optimization
    lbfgs.step(closure)
    
    # Extract final solution
    with torch.no_grad():
        pos_final = u_vec[:3]
        logT_final = u_vec[3]
        T_final = t_min * torch.exp(logT_final)
        
        # Verify periodicity over 3T
        print("\nVerifying periodicity over 3T...")
        t_verify = torch.linspace(0, 3*T_final.item(), 4)
        traj = odeint(lorenz_rhs, pos_final, t_verify, method='dopri5',
                     rtol=args.rtol, atol=args.atol)
        
        max_deviation = 0.0
        for i in range(1, 4):
            deviation = (traj[i] - pos_final).norm()
            max_deviation = max(max_deviation, deviation)
            print(f"Deviation at {i}T: {deviation:.12e}")
        
        print(f"Maximum deviation over 3T: {max_deviation:.12e}")
    
    # Plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Lorenz attractor (in grey)
        print("\nGenerating Lorenz attractor for visualization...")
        attractor_time = 50.0  # Shorter time for faster generation
        attractor_initial = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        t_attractor = torch.linspace(0, attractor_time, 10000)
        attractor_traj = odeint(lorenz_rhs, attractor_initial, t_attractor, method='dopri5',
                                rtol=args.rtol, atol=args.atol).detach().numpy()
        ax.plot(attractor_traj[1000:, 0], attractor_traj[1000:, 1], attractor_traj[1000:, 2], 
                'gray', alpha=0.4, linewidth=0.8, label='Lorenz attractor')
        
        # Plot periodic orbit (in red, one period only)
        t_po = torch.linspace(0, T_final.item(), 500)
        po_traj = odeint(lorenz_rhs, pos_final, t_po, method='dopri5',
                         rtol=args.rtol, atol=args.atol).detach().numpy()
        ax.plot(po_traj[:, 0], po_traj[:, 1], po_traj[:, 2], 'red', 
                linewidth=3, label=f'Periodic orbit (T={T_final:.6f})')
        
        # Mark the initial point
        ax.scatter(*pos_final.detach().numpy(), color='red', s=100, marker='o', 
                   edgecolor='black', label='Initial point')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('Lorenz System: Periodic Orbit on Strange Attractor', fontsize=14)
        ax.legend(fontsize=11)
        
        # Adjust viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig('lorenz_periodic_orbit.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'lorenz_periodic_orbit.png'")
    except ImportError:
        print("\nMatplotlib not available; skipping plot")
    
    # Final summary
    elapsed_time = time.time() - start_time
    residual_final = flow_map(pos_final, logT_final, t_min=t_min) - pos_final
    residual_norm_final = residual_final.norm()
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Final state: [{pos_final[0]:.12f}, {pos_final[1]:.12f}, {pos_final[2]:.12f}]")
    print(f"Period T: {T_final:.12f}")
    print(f"Residual norm: {residual_norm_final:.12e}")
    print(f"CPU time: {elapsed_time:.2f} seconds")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find periodic orbits of the Lorenz system')
    parser.add_argument('--rtol', type=float, default=DEFAULT_RTOL,
                        help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=DEFAULT_ATOL,
                        help='Absolute tolerance for ODE solver')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=DEFAULT_LAMBDA,
                        help='Penalty weight for avoiding equilibria')
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER,
                        help='Maximum number of L-BFGS iterations')
    parser.add_argument('--skip-adam', action='store_true',
                        help='Skip Adam pre-optimization and go directly to L-BFGS')
    
    args = parser.parse_args()
    
    # Update global tolerances
    DEFAULT_RTOL = args.rtol
    DEFAULT_ATOL = args.atol
    
    main(args)

# README
"""
Usage: python lorenz_periodic_orbit.py [options]

This script finds periodic orbits of the Lorenz system by minimizing a differentiable
cost function using automatic differentiation.

Options:
  --rtol        Relative tolerance for ODE integration (default: 1e-9)
  --atol        Absolute tolerance for ODE integration (default: 1e-9)
  --lambda      Penalty weight for avoiding equilibria (default: 1e-3)
  --max-iter    Maximum L-BFGS iterations (default: 300)

Example:
  python lorenz_periodic_orbit.py --lambda 0.001 --max-iter 500

The initial_guess_po() function uses k-d tree nearest neighbor search to find
good initial guesses for periodic orbits.
"""