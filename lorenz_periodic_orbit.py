#!/usr/bin/env python3
"""
Unstable Periodic Orbit (UPO) finder for the Lorenz system using automatic differentiation.

This script locates Unstable Periodic Orbits (UPOs) of the Lorenz system by combining
numerical optimization and automatic differentiation. The key methods employed are:

1. Initial Guess Generation:
   - Uses k-d tree nearest neighbor search to find close returns in the Lorenz attractor
   - Identifies candidate periodic orbits by analyzing trajectory recurrence

2. Numerical Optimization:
   - Formulates periodicity as a boundary value problem
   - Minimizes a differentiable objective function that measures deviation from periodicity
   - Uses L-BFGS optimizer for efficient convergence

3. Automatic Differentiation:
   - Leverages PyTorch's autograd for exact gradient computation
   - Enables efficient optimization by automatically computing gradients of the ODE solution
   with respect to initial conditions and period

4. Numerical Integration:
   - Employs DOPRI5 (Dormand-Prince) adaptive step-size ODE solver
   - Uses strict tolerances (RTOL=1e-9, ATOL=1e-9) for accurate trajectory computation

UPOs are fundamental to understanding the chaotic dynamics of the Lorenz attractor,
serving as the 'skeleton' that organizes the chaotic flow in phase space. This
implementation provides a robust and efficient method for their numerical detection.

Author: Dr. Denys Dutykh
        Khalifa University of Science and Technology
        Abu Dhabi, UAE

Date: May 16, 2025

License: GNU Lesser General Public License version 3.0 (LGPL-3.0)
         See the LGPL-3.0 file for details.
"""

import argparse
import os
import time
import torch
import numpy as np
from torchdiffeq import odeint
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
# Set REPRODUCIBLE = True for deterministic results, False for random behavior
REPRODUCIBLE = False
SEED = 1  # Only used if REPRODUCIBLE is True

# Set random seeds for reproducibility
if REPRODUCIBLE:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    # Use system entropy for random number generation
    torch.seed()
    np.random.seed()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Set default dtype for numerical accuracy
torch.set_default_dtype(torch.float64)

# =============================================================================
# Optimization Parameters
# =============================================================================
# Default convergence tolerance for optimization
DEFAULT_TOL = 1e-6  # Default convergence tolerance for optimization

# =============================================================================
# Lorenz System Parameters
# =============================================================================
# Standard parameter values for the Lorenz system that produce chaotic behavior:
# dx/dt = σ(y - x)
# dy/dt = x(ρ - z) - y
# dz/dt = xy - βz
SIGMA = 10.0    # Prandtl number (ratio of momentum diffusivity to thermal diffusivity)
RHO = 28.0      # Rayleigh number (controls the intensity of convection)
BETA = 8.0 / 3.0  # Geometric factor (aspect ratio of the convection rolls)

# =============================================================================
# Optimization Parameters
# =============================================================================
# Numerical tolerances for the ODE solver (DOPRI5 method):
DEFAULT_RTOL = 1e-9      # Relative tolerance for ODE solver
DEFAULT_ATOL = 1e-9      # Absolute tolerance for ODE solver

# Weight for the penalty term that prevents convergence to trivial solutions (equilibria):
# Higher values make the optimizer avoid equilibrium points more aggressively
DEFAULT_LAMBDA = 1e-3    # Penalty weight for avoiding equilibria

# Small constant to prevent division by zero in the cost function:
DEFAULT_EPS = 1e-12     # Numerical stability constant

# Maximum number of L-BFGS optimization iterations:
DEFAULT_MAX_ITER = 500  # Maximum iterations for L-BFGS optimizer

def lorenz_rhs(t, state):
    """
    Right-hand side of the Lorenz system.
    
    This function defines the system of ordinary differential equations (ODEs)
    that describe the Lorenz attractor, a simplified mathematical model for 
    atmospheric convection.
    
    The system is defined by three coupled nonlinear ODEs:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
    
    Where (x, y, z) represent:
    - x: Rate of convective motion (proportional to the intensity of convection)
    - y: Temperature difference between ascending and descending currents
    - z: Distortion of vertical temperature profile from linearity
    
    Args:
        t: Time (unused in autonomous system but required by odeint interface)
        state: State vector [x, y, z]
    
    Returns:
        Tensor of shape (3,) containing [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state  # Unpack state variables
    
    # x-equation: Represents the rate of convective motion
    # σ(y - x) describes the transfer of heat between the fluid layers
    dx = SIGMA * (y - x)
    
    # y-equation: Represents the temperature difference
    # x(ρ - z) - y includes the effect of temperature gradient and damping
    dy = x * (RHO - z) - y
    
    # z-equation: Represents the distortion of the vertical temperature profile
    # xy - βz includes the nonlinear coupling and damping
    dz = x * y - BETA * z
    
    return torch.stack([dx, dy, dz])  # Return derivatives as a PyTorch tensor

def flow_map(x0_vec, T):
    """
    Compute the time-T flow map of the Lorenz system.
    
    This function integrates the Lorenz system from initial state x0_vec
    for time T using the DOPRI5 adaptive step-size ODE solver.
    
    Args:
        x0_vec: Initial state vector [x0, y0, z0], can be batched
        T: Integration time (can be a scalar or tensor of same batch size as x0_vec)
    
    Returns:
        Final state after integrating from 0 to T
    """
    # Create time span from 0 to T for each batch element
    t_span = torch.stack([torch.zeros_like(T), T])
    
    # Integrate the Lorenz system
    sol = odeint(lorenz_rhs, x0_vec, t_span, method='dopri5', 
                 rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    
    # Return only the final state (at t=T)
    return sol[-1]

def cost(u_vec, lambda_=DEFAULT_LAMBDA, eps=DEFAULT_EPS, t_min=0.0):
    """
    Cost function for unstable periodic orbit (UPO) optimization.
    
    The cost consists of:
    1. Periodicity residual: ||flow(T) - initial_state||²
    2. Penalty term to avoid equilibria: λ/(||f(x0)||² + ε)
    
    The period T is parameterized as T = t_min + exp(logT) to ensure T > t_min.
    This transforms the constrained optimization problem into an unconstrained one.
    
    Args:
        u_vec: Vector [x0, y0, z0, logT] with gradient tracking
        lambda_: Weight for the penalty term (default: 1e-3)
        eps: Small constant for numerical stability (default: 1e-12)
        t_min: Minimum allowed period (default: 0.0, effectively no constraint)
    
    Returns:
        Scalar cost value with gradient tracking
    """
    # Unpack the optimization variables
    # u_vec = [x0, y0, z0, logT], where T = t_min + exp(logT)
    x0_vec = u_vec[:-1]  # Initial state vector [x0, y0, z0]
    logT = u_vec[-1]     # Log of (T - t_min) for unconstrained optimization
    
    # Convert from unconstrained parameter logT to actual period T
    # T = t_min + exp(logT) ensures T > t_min for all real logT
    T = t_min + torch.exp(logT)
    
    # Compute the flow map: integrate the system from x0 for time T
    xT = flow_map(x0_vec, T)  # Final state after time T
    
    # Periodicity condition: minimize ||x(T) - x(0)||²
    # This measures how close the trajectory comes back to its starting point
    periodicity_residual = torch.sum((xT - x0_vec) ** 2)
    
    # Compute the vector field at the initial point: f(x0) = [dx/dt, dy/dt, dz/dt] at t=0
    # This is used to avoid convergence to equilibrium points
    f_x0 = lorenz_rhs(None, x0_vec)  # None for time since system is autonomous
    f_norm_sq = torch.sum(f_x0 ** 2)  # Squared L2 norm of the vector field
    
    # Penalty term to avoid trivial solutions (equilibrium points)
    # As f(x0) → 0 (approaching equilibrium), penalty → λ/ε
    # For non-equilibrium points, penalty ≈ λ/||f(x0)||²
    penalty = lambda_ / (f_norm_sq + eps)
    
    # Total cost combines periodicity condition and equilibrium avoidance
    # The optimizer will try to minimize this total cost
    total_cost = periodicity_residual + penalty
    
    return total_cost  # Return the total cost to be minimized

def initial_guess_po(
    T_final=400.0,
    n_steps=40000,
    t_min=3.0,  # Minimum time separation
    t_max=6.0,  # Maximum time separation
    x_bounds=(-20, 20),
    y_bounds=(-30, 30),
    z_bounds=(0, 50),
    seed=None
):
    """
    Generate a heuristic initial guess for an Unstable Periodic Orbit (UPO) of the Lorenz system.
    Uses k-d tree for efficient nearest neighbor search to find points in phase space that
    return close to their starting position after some time (potential periodic orbits).
    
    Args:
        T_final: Total integration time for generating the trajectory
        n_steps: Number of time steps for integration
        t_min: Minimum time separation between similar states (avoids nearby points)
        t_max: Maximum time separation between similar states (looks for periodicity within this range)
        x_bounds: Tuple of (min, max) for initial x-coordinate
        y_bounds: Tuple of (min, max) for initial y-coordinate
        z_bounds: Tuple of (min, max) for initial z-coordinate
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'T0': Estimated period (time lag between similar states)
        - 'x0','y0','z0': Initial state vector of the potential periodic orbit
        - 'min_dist': Minimum distance between similar states found
    """
    print("\n" + "="*50)
    print("GENERATING INITIAL GUESS FOR PERIODIC ORBIT")
    print("="*50)
    
    # Set random seed if provided
    if seed is not None:
        print(f"Using random seed: {seed}")
        np.random.seed(seed)
    else:
        print("No random seed provided - using system entropy")

    # Generate random initial condition within specified bounds
    print(f"\nGenerating trajectory with random initial condition in bounds:")
    print(f"  x ∈ [{x_bounds[0]}, {x_bounds[1]}], y ∈ [{y_bounds[0]}, {y_bounds[1]}], z ∈ [{z_bounds[0]}, {z_bounds[1]}]")
    x0_rand = np.random.uniform(*x_bounds)
    y0_rand = np.random.uniform(*y_bounds)
    z0_rand = np.random.uniform(*z_bounds)
    print(f"Initial condition: x0 = {x0_rand:.4f}, y0 = {y0_rand:.4f}, z0 = {z0_rand:.4f}")

    # Generate time points and integrate the Lorenz system
    print(f"\nIntegrating Lorenz system for T = {T_final:.2f} with {n_steps} steps...")
    t_eval = torch.linspace(0.0, T_final, n_steps)
    state0 = torch.tensor([x0_rand, y0_rand, z0_rand], dtype=torch.float64)
    
    # Use adaptive step-size ODE solver with specified tolerances
    print(f"Using DOPRI5 ODE solver with rtol = {DEFAULT_RTOL:.1e}, atol = {DEFAULT_ATOL:.1e}")
    
    # Add progress bar for ODE integration
    print("Integrating...")
    with tqdm(total=len(t_eval), desc="ODE Integration", unit="step", ncols=100) as pbar:
        def ode_progress(t, y):
            # Update progress bar to current time step
            current_step = int(t / T_final * (len(t_eval) - 1))
            pbar.update(current_step - pbar.n)
            return lorenz_rhs(t, y)
            
        # Run the integration with progress callback
        traj_torch = odeint(ode_progress, state0, t_eval, method='dopri5',
                           rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
        
        # Ensure progress bar completes
        pbar.update(len(t_eval) - pbar.n)
    
    # Convert to numpy for k-d tree operations
    traj = traj_torch.detach().cpu().numpy()  # shape (n_steps, 3)
    t_numpy = t_eval.numpy()
    print(f"Trajectory generated with {len(traj)} points")

    # Convert time bounds to index bounds for efficient searching
    dt = t_numpy[1] - t_numpy[0]
    min_steps = int(t_min / dt)
    max_steps = int(t_max / dt)
    print(f"\nSearching for similar states with time separation: {t_min:.2f} < T < {t_max:.2f}")
    print(f"Corresponds to {min_steps} - {max_steps} time steps (dt = {dt:.4f})")
    
    # Initialize tracking variables
    min_dist = float('inf')
    best = None
    
    # Process trajectory in overlapping windows for better performance with k-d trees
    window_size = max_steps - min_steps + 100  # Add padding to catch all possible pairs
    n_windows = (n_steps + window_size // 2 - 1) // (window_size // 2)  # Ceiling division
    print(f"\nProcessing trajectory in {n_windows} overlapping windows...")
    
    for window_idx, i_start in enumerate(range(0, n_steps, window_size // 2)):
        i_end = min(i_start + window_size, n_steps)
        window_traj = traj[i_start:i_end]
        
        # Build k-d tree for this window (enables fast nearest-neighbor searches)
        tree = cKDTree(window_traj)
        
        # Only search first half of window to avoid duplicate checks
        search_limit = min(window_size // 2, i_end - i_start)
        
        for i_local in range(search_limit):
            i_global = i_start + i_local
            
            # Find k nearest neighbors (excluding the point itself)
            k = min(50, i_end - i_start - i_local)
            if k < 2:
                continue  # Need at least one neighbor (excluding self)
                
            # Query k nearest neighbors (returns distances and indices)
            distances, indices = tree.query(window_traj[i_local], k=k)
            
            # Check each neighbor (skip index 0 which is the point itself)
            for dist, j_local in zip(distances[1:], indices[1:]):
                j_global = i_start + j_local
                
                # Calculate actual time separation between points
                dt_sep = t_numpy[j_global] - t_numpy[i_global]
                
                # Only consider pairs within the specified time range
                if dt_sep < t_min or dt_sep > t_max:
                    continue
                
                # Update best candidate if this pair is closer than previous best
                if dist < min_dist:
                    min_dist = dist
                    best = (i_global, j_global)
                    # Print progress for each improvement
                    print(f"  New best: dist = {dist:.6f}, T = {dt_sep:.4f} "
                          f"at t = {t_numpy[i_global]:.2f} → {t_numpy[j_global]:.2f}")

    # Check if any valid pairs were found
    if best is None:
        raise RuntimeError(
            f"No sufficiently similar state pairs found with {t_min=} and {t_max=}.\n"
            f"Try increasing T_final or adjusting the time separation bounds."
        )

    # Extract the best candidate
    i0, j0 = best
    T0 = t_numpy[j0] - t_numpy[i0]
    x0, y0, z0 = traj[i0]
    
    print("\n" + "="*50)
    print("INITIAL GUESS GENERATION COMPLETE")
    print("="*50)
    print(f"Best candidate found with distance = {min_dist:.6f}")
    print(f"Initial state: x0 = {x0:.8f}, y0 = {y0:.8f}, z0 = {z0:.8f}")
    print(f"Estimated period: T = {T0:.8f} (found at t = {t_numpy[i0]:.2f} → {t_numpy[j0]:.2f})")
    print("="*50 + "\n")
    
    return {
        'T0': T0, 
        'x0': x0, 
        'y0': y0, 
        'z0': z0, 
        'min_dist': min_dist,
        't0': t_numpy[i0],
        't1': t_numpy[j0]
    }

def main(args):
    """
    Main optimization routine for finding Unstable Periodic Orbits (UPOs).
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    
    # Get initial guess
    guess = initial_guess_po()
    
    # Convert initial period to log(T - t_min) parameterization
    t_min = 1.0  # Minimum period constraint
    T_initial = guess['T0']
    logT_init = torch.log(torch.tensor(T_initial - t_min + 1e-12))  # Add small offset for numerical stability
    
    # Initial guess vector [x0, y0, z0, log(T - t_min)]
    u_init = torch.tensor([guess['x0'], guess['y0'], guess['z0'], logT_init], 
                         dtype=torch.float64, requires_grad=True)
    
    print(f"Initial guess: x0 = {u_init[0]:.6f}, y0 = {u_init[1]:.6f}, "
          f"z0 = {u_init[2]:.6f}, T = {T_initial:.6f}")
    print(f"Using parameterization: T = {t_min} + exp(logT) ensures T > {t_min}")
    
    # Main optimization with L-BFGS
    print("\nStarting L-BFGS optimization...")
    u_vec = u_init.clone().detach().requires_grad_(True)
    
    # Set up L-BFGS optimizer
    lbfgs = torch.optim.LBFGS([u_vec], 
                            max_iter=args.max_iter,
                            tolerance_grad=1e-12,
                            tolerance_change=1e-12,
                            history_size=25,
                            line_search_fn='strong_wolfe')
    
    # Track optimization progress
    iteration = 0
    prev_cost = float('inf')
    converged = False
    best_iter = 0
    best_cost = float('inf')
    best_state = u_vec.detach().clone()
    
    # Initialize history
    history = {
        'iter': [],
        'cost': [],
        'period': [],
        'residual_norm': [],
        'grad_norm': []
    }
    
    def closure():
        nonlocal iteration, prev_cost, converged, best_iter, best_cost, best_state
        
        lbfgs.zero_grad()
        
        # Compute cost with current parameters
        current_cost = cost(u_vec, lambda_=args.lambda_, t_min=t_min)
        
        # Backpropagate gradients
        current_cost.backward()
        
        with torch.no_grad():
            # Get current state and cost
            current_pos = u_vec[:3]
            current_T = t_min + torch.exp(u_vec[3]).item()
            cost_value = current_cost.item()
            
            # Compute residual norm (periodicity condition)
            xT = flow_map(current_pos, torch.tensor(current_T))
            residual = (xT - current_pos).norm().item()
            
            # Compute gradient norm
            grad_norm = u_vec.grad.norm().item() if u_vec.grad is not None else 0.0
            
            # Track best solution
            if cost_value < best_cost:
                best_cost = cost_value
                best_state = u_vec.detach().clone()
                best_iter = iteration
            
            # Save history
            history['iter'].append(iteration)
            history['cost'].append(cost_value)
            history['period'].append(current_T)
            history['residual_norm'].append(residual)
            history['grad_norm'].append(grad_norm)
            
            # Print progress every 5 iterations, first iteration, or on convergence
            if iteration % 5 == 0 or iteration == 0 or converged or iteration == args.max_iter - 1:
                if iteration % 5 == 0 or iteration == 0:
                    print(f"\n--- Iteration {iteration} ---")
                    print(f"Cost: {cost_value:.6e}")
                    print(f"Position: x = {current_pos[0]:.8f}, y = {current_pos[1]:.8f}, z = {current_pos[2]:.8f}")
                    print(f"Period: T = {current_T:.8f}")
                    print(f"Residual norm: {residual:.6e}")
                    print(f"Gradient norm: {grad_norm:.6e}")
            
            # Check for convergence (relative cost change)
            if iteration > 0 and abs(prev_cost - cost_value) < args.tol * (1 + abs(prev_cost)):
                if not converged:  # Only print convergence message once
                    print(f"\nConverged after {iteration} iterations (relative cost change < {args.tol})")
                    print(f"Final cost: {cost_value:.6e}, Final period: {current_T:.8f}")
                converged = True
            
            # Early stopping if no improvement for 50 iterations
            if iteration - best_iter > 50 and iteration > 100 and not converged:
                print(f"\nEarly stopping: No improvement for 50 iterations")
                converged = True
            
            prev_cost = cost_value
            iteration += 1
            
            if cost_value < 1e-10:  # Convergence threshold
                converged = True
        
        return current_cost
    
    # Run optimization with enhanced monitoring
    print("\n" + "="*50)
    print("STARTING OPTIMIZATION")
    print("="*50)
    print(f"Max iterations: {args.max_iter}")
    print(f"Convergence tolerance: {args.tol:.1e}")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    # Initial evaluation
    with torch.no_grad():
        initial_cost = cost(u_vec, lambda_=args.lambda_, t_min=t_min).item()
        initial_T = t_min + torch.exp(u_vec[3]).item()
        print(f"Initial cost: {initial_cost:.6e}, Initial period: {initial_T:.6f}")
        
        # Save initial state to history
        history['iter'].append(0)
        history['cost'].append(initial_cost)
        history['period'].append(initial_T)
        history['residual_norm'].append(float('nan'))
        history['grad_norm'].append(float('nan'))
    
    try:
        # Run the optimization
        lbfgs.step(closure)
        
        # Final evaluation if not already converged
        if not converged:
            with torch.no_grad():
                current_pos = u_vec[:3]
                current_T = t_min + torch.exp(u_vec[3]).item()
                current_cost = cost(u_vec, lambda_=args.lambda_, t_min=t_min).item()
                
                # Save final state if it's better
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = u_vec.detach().clone()
                    
                print("\nOptimization finished without explicit convergence.")
    
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        print("Returning best solution found so far...")
    
    # Use the best state found during optimization
    final_pos = best_state[:3]
    final_T = t_min + torch.exp(best_state[3]).item()
    final_cost = best_cost
    
    # Compute final residual
    with torch.no_grad():
        xT = flow_map(final_pos, torch.tensor(final_T))
        final_residual = (xT - final_pos).norm().item()
    
    # Print final results
    print("\n" + "="*50)
    print("OPTIMIZATION FINISHED")
    print("="*50)
    print(f"Status: {'Converged' if converged else 'Max iterations reached'}")
    print(f"Final cost: {final_cost:.6e}")
    print(f"Final residual: {final_residual:.6e}")
    print(f"Final position: x = {final_pos[0]:.8f}, y = {final_pos[1]:.8f}, z = {final_pos[2]:.8f}")
    print(f"Final period: T = {final_T:.8f}")
    print(f"Total iterations: {iteration}")
    print(f"Best iteration: {best_iter}")
    print("="*50 + "\n")
    
    # Verify periodicity of final solution
    with torch.no_grad():
        xT = flow_map(final_pos, torch.tensor(final_T))
        residual = (xT - final_pos).norm().item()
        print(f"Final residual norm: {final_residual:.6e}")
    
    # Verify the solution by integrating for 3 periods before plotting
    print("\n" + "="*50)
    print("VERIFYING PERIODIC SOLUTION")
    print("="*50)
    
    # Create time points for 3 periods with high resolution
    t_verify = torch.linspace(0, 3 * final_T, 3000)
    
    # Integrate the system
    with torch.no_grad():
        traj = odeint(lorenz_rhs, final_pos, t_verify, 
                     method='dopri5', rtol=args.rtol, atol=args.atol)
    
    # Convert to numpy for easier manipulation
    traj_np = traj.numpy()
    t_verify_np = t_verify.numpy()
    
    # Find the state at T, 2T, and 3T using interpolation
    T = final_T
    times = [T, 2*T, 3*T]
    states = []
    
    for t in times:
        # Find the index where we should insert t in t_verify_np
        idx = np.searchsorted(t_verify_np, t)
        if idx == 0 or idx == len(t_verify_np):
            states.append(traj_np[idx-1])
            continue
            
        # Linear interpolation
        alpha = (t - t_verify_np[idx-1]) / (t_verify_np[idx] - t_verify_np[idx-1])
        state = traj_np[idx-1] + alpha * (traj_np[idx] - traj_np[idx-1])
        states.append(state)
    
    # Calculate errors at each period
    print("\nVerification of periodicity:")
    print("-" * 90)
    print(f"{'Cycle':<10} | {'Time':<12} | {'Error (norm)':<15} | {'x-Error':<15} | {'y-Error':<15} | {'z-Error':<15}")
    print("-" * 90)
    
    # Get the optimized periodic orbit's initial state
    x0 = final_pos.numpy()  # This is the periodic orbit found by the optimizer
    
    for i, (t, state) in enumerate(zip(times, states)):
        # Calculate error between state after i+1 periods and the initial state
        # For a perfect periodic orbit, this should be close to zero
        error = state - x0
        error_norm = np.linalg.norm(error)
        
        print(f"{i+1}T{' '*(9-len(str(i+1)))} | {t:<12.6f} | {error_norm:<15.6e} | "
              f"{error[0]:<15.6e} | {error[1]:<15.6e} | {error[2]:<15.6e}")
    
    # Final verification after 3 periods
    final_state = states[-1]  # State after 3 periods
    final_error = final_state - x0  # Compare with initial state of periodic orbit
    final_error_norm = np.linalg.norm(final_error)
    
    print("\nFinal verification after 3 periods:")
    print(f"Initial state: {final_pos.numpy()}")
    print(f"Final state:   {final_state}")
    print(f"Error norm:    {final_error_norm:.6e}")
    
    # Store the trajectory for plotting
    traj_3T = traj_np
    t_3T = t_verify_np
    
    # Final timing
    elapsed_time = time.time() - start_time
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Final state: [{final_pos[0]:.12f}, {final_pos[1]:.12f}, {final_pos[2]:.12f}]")
    print(f"Period T: {final_T:.12f}")
    print(f"Final cost: {final_cost:.12e}")
    print(f"CPU time: {elapsed_time:.2f} seconds")
    print("="*50)
    
    # Now generate the plots after verification is complete
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create a 3D plot of the attractor and the periodic orbit
        print("\nGenerating visualization...")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot a portion of the Lorenz attractor for context (in gray)
        print("  - Plotting attractor...")
        attractor_traj = odeint(lorenz_rhs, final_pos, torch.linspace(0, 100, 10000), 
                              method='dopri5', rtol=args.rtol, atol=args.atol).detach().numpy()
        ax.plot(attractor_traj[1000:, 0], attractor_traj[1000:, 1], attractor_traj[1000:, 2], 
               'gray', alpha=0.4, linewidth=0.8, label='Lorenz attractor')
        
        # Plot unstable periodic orbit (in red, one period only)
        print("  - Plotting periodic orbit...")
        t_po = torch.linspace(0, final_T, 1000)
        po_traj = odeint(lorenz_rhs, final_pos, t_po, 
                        method='dopri5', rtol=args.rtol, atol=args.atol).detach().numpy()
        ax.plot(po_traj[:, 0], po_traj[:, 1], po_traj[:, 2], 'red', 
               linewidth=3, label=f'Unstable Periodic Orbit (T={final_T:.6f})')
            
        # Mark the initial point
        ax.scatter(*final_pos.detach().numpy(), color='red', s=100, 
                 marker='o', edgecolor='black', label='Initial point')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('Lorenz System: Unstable Periodic Orbit (UPO) on Strange Attractor', 
                    fontsize=14)
        ax.legend(fontsize=11)
        ax.view_init(elev=20, azim=135)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('images', exist_ok=True)
        base_name = 'lorenz_periodic_orbit'
        file_ext = '.png'
        counter = 1
        output_file = f'images/{base_name}{file_ext}'
        
        while os.path.exists(output_file):
            output_file = f'images/{base_name}_{counter}{file_ext}'
            counter += 1
            
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as '{output_file}'")
        
        # Plot optimization history
        print("  - Plotting optimization history...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot cost and residual
        ax1.semilogy(history['iter'], history['cost'], 'b-', label='Cost')
        ax1.semilogy(history['iter'], history['residual_norm'], 'r--', label='Residual norm')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Value (log scale)')
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)
        
        # Plot period and gradient norm
        ax2_twin = ax2.twinx()
        ax2.plot(history['iter'], history['period'], 'g-', label='Period')
        ax2_twin.semilogy(history['iter'], history['grad_norm'], 'm--', label='Gradient norm')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Period', color='g')
        ax2_twin.set_ylabel('Gradient norm', color='m')
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        
        # Save the optimization history plot
        history_file = f'images/{base_name}_history.png'
        plt.savefig(history_file, dpi=150, bbox_inches='tight')
        print(f"Optimization history plot saved as '{history_file}'")
        
        # Show the plots
        plt.show()
        
    except ImportError as e:
        print(f"\nWarning: Could not generate plots - {str(e)}")
        print("Make sure you have matplotlib installed: pip install matplotlib")
    except Exception as e:
        print(f"\nWarning: Error generating plots - {str(e)}")
    
    return final_pos, final_T, final_cost, history, traj_3T, t_3T

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find Unstable Periodic Orbits (UPOs) of the Lorenz system')
    parser.add_argument('--rtol', type=float, default=DEFAULT_RTOL,
                        help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=DEFAULT_ATOL,
                        help='Absolute tolerance for ODE solver')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=DEFAULT_LAMBDA,
                        help='Penalty weight for avoiding equilibria')
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER,
                        help='Maximum number of L-BFGS iterations')
    parser.add_argument('--tol', type=float, default=DEFAULT_TOL,
                        help=f'Convergence tolerance for optimization (default: {DEFAULT_TOL})')
    args = parser.parse_args()
    
    # Update global tolerances
    DEFAULT_RTOL = args.rtol
    DEFAULT_ATOL = args.atol
    
    main(args)

# README
"""
Usage: python lorenz_periodic_orbit.py [options]

This script finds Unstable Periodic Orbits (UPOs) of the Lorenz system by minimizing a differentiable
cost function using automatic differentiation.

Options:
  --rtol        Relative tolerance for ODE integration (default: DEFAULT_RTOL=1e-9)
  --atol        Absolute tolerance for ODE integration (default: DEFAULT_ATOL=1e-9)
  --lambda      Penalty weight for avoiding equilibria (default: DEFAULT_LAMBDA=1e-3)
  --max-iter    Maximum L-BFGS iterations (default: DEFAULT_MAX_ITER=300)

Example:
  python lorenz_periodic_orbit.py --lambda 0.001 --max-iter 500

The initial_guess_po() function uses k-d tree nearest neighbor search to find
good initial guesses for Unstable Periodic Orbits (UPOs).
"""