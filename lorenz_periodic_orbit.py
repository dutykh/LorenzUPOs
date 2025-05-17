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

# --- torch.compile for PyTorch 2.0+ ---
import torch
USE_COMPILE = hasattr(torch, 'compile')
if USE_COMPILE:
    lorenz_rhs = torch.compile(lorenz_rhs, mode='max-autotune')

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
    
    # Penalty term to avoid equilibria: lambda_ / (||f(x0)||^2 + eps)
    f_x0 = lorenz_rhs(0.0, x0_vec)
    penalty = lambda_ / (torch.sum(f_x0 ** 2) + eps)
    
    # Total cost
    total_cost = periodicity_residual + penalty
    return total_cost  # Return the total cost to be minimized

# Do NOT compile cost with torch.compile, as it may call ODE solvers or code with dynamic shapes.
# Only compile lorenz_rhs (already done above).

def initial_guess_po(
    T_final=2000.0,
    n_steps=400000,
    t_min=4.0,  # Minimum time separation
    t_max=8.0,  # Maximum time separation
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
    Main optimization routine for finding Unstable Periodic Orbits (UPOs) in the Lorenz system.
    
    This function sets up and runs the optimization process to find periodic solutions to the
    Lorenz system using the L-BFGS optimization algorithm. It handles the parameterization of
    the period, initializes the optimization variables, and sets up tracking for the optimization
    progress.
    
    Args:
        args: Command-line arguments containing optimization parameters such as
              max_iter (maximum iterations), rtol (relative tolerance), and atol (absolute tolerance).
    """
    # Record start time to measure total execution time
    start_time = time.time()
    
    # Step 1: Generate initial guess for the periodic orbit
    # This includes initial position (x0, y0, z0) and an estimated period T0
    guess = initial_guess_po()
    
    # Step 2: Parameterize the period T to ensure it remains positive during optimization
    # We use the transformation T = t_min + exp(logT) where t_min is the minimum allowed period
    # This ensures T > t_min for any real value of logT
    t_min = 1.0  # Minimum period constraint (avoids numerical issues with very small periods)
    T_initial = guess['T0']  # Initial period estimate from the guess
    
    # Convert initial period to log-space with numerical stability offset
    # The small offset (1e-12) prevents log(0) if T_initial = t_min
    logT_init = torch.log(torch.tensor(T_initial - t_min + 1e-12))
    
    # Create the initial parameter vector: [x0, y0, z0, log(T - t_min)]
    # We use requires_grad=True to enable automatic differentiation
    u_init = torch.tensor([guess['x0'], guess['y0'], guess['z0'], logT_init], 
                         dtype=torch.float64, requires_grad=True)
    
    # Print initial guess information for user feedback
    print(f"Initial guess: x0 = {u_init[0]:.6f}, y0 = {u_init[1]:.6f}, "
          f"z0 = {u_init[2]:.6f}, T = {T_initial:.6f}")
    print(f"Using parameterization: T = {t_min} + exp(logT) ensures T > {t_min}")
    
    # Step 3: Set up the L-BFGS optimization
    print("\nStarting L-BFGS optimization...")
    
    # Create a copy of initial parameters that will be updated during optimization
    u_vec = u_init.clone().detach().requires_grad_(True)
    
    # Configure the L-BFGS optimizer with the following parameters:
    # - max_iter: Maximum number of iterations per optimization step
    # - tolerance_grad: Termination tolerance on the gradient norm
    # - tolerance_change: Termination tolerance on function value/parameter changes
    # - history_size: Update history size (affects memory usage and convergence)
    # - line_search_fn: Line search algorithm ('strong_wolfe' is robust for this problem)
    lbfgs = torch.optim.LBFGS([u_vec], 
                            max_iter=args.max_iter,
                            tolerance_grad=1e-12,     # Very tight tolerance for precise solutions
                            tolerance_change=1e-12,   # Very tight tolerance for precise solutions
                            history_size=25,          # Number of past updates to store
                            line_search_fn='strong_wolfe')  # Robust line search method
    
    # Step 4: Initialize variables to track optimization progress
    iteration = 0               # Current iteration counter
    prev_cost = float('inf')    # Cost from previous iteration (for convergence check)
    converged = False          # Flag to track if convergence was achieved
    best_iter = 0              # Iteration number with the best solution found
    best_cost = float('inf')   # Best (lowest) cost found so far
    best_state = u_vec.detach().clone()  # Best parameter values found so far
    
    # Initialize history dictionary to store optimization metrics at each iteration
    # This is useful for analysis and visualization after optimization completes
    history = {
        'iter': [],         # Iteration numbers
        'cost': [],          # Cost function values
        'period': [],        # Period values
        'residual_norm': [], # Norm of the periodicity residual
        'grad_norm': []      # Norm of the gradient
    }
    
    def closure():
        """
        Closure function for L-BFGS optimization that computes the loss, performs backpropagation,
        and tracks optimization progress.
        
        This function is called multiple times by the L-BFGS optimizer. It:
        1. Computes the loss (cost) for the current parameters
        2. Performs backpropagation to compute gradients
        3. Tracks optimization metrics and convergence
        4. Implements early stopping criteria
        5. Maintains the best solution found
        
        Returns:
            torch.Tensor: The computed cost value
        """
        # Access and potentially modify these variables from the outer scope
        nonlocal iteration, prev_cost, converged, best_iter, best_cost, best_state
        
        # Clear any previously computed gradients
        lbfgs.zero_grad()
        
        # Compute the cost (objective function) with current parameters
        # The cost measures how well the current solution satisfies the periodicity condition
        current_cost = cost(u_vec, lambda_=args.lambda_, t_min=t_min)
        
        # Backpropagate to compute gradients of the cost w.r.t. parameters
        # This populates the .grad attribute of u_vec
        current_cost.backward()
        
        # The following operations don't require gradient computation
        with torch.no_grad():
            # Extract current position (first 3 elements) and period (4th element)
            current_pos = u_vec[:3]  # [x, y, z] coordinates
            # Convert parameterized period back to actual period: T = t_min + exp(logT)
            current_T = t_min + torch.exp(u_vec[3]).item()
            cost_value = current_cost.item()  # Get Python scalar from tensor
            
            # Compute how well the periodicity condition is satisfied
            # This is the norm of the difference between initial and final state after one period
            xT = flow_map(current_pos, torch.tensor(current_T))
            residual = (xT - current_pos).norm().item()
            
            # Compute the norm of the gradient (useful for monitoring convergence)
            grad_norm = u_vec.grad.norm().item() if u_vec.grad is not None else 0.0
            
            # Track the best solution found so far based on cost value
            if cost_value < best_cost:
                best_cost = cost_value
                best_state = u_vec.detach().clone()  # Store a copy of the best parameters
                best_iter = iteration
            
            # Save optimization history for analysis and visualization
            history['iter'].append(iteration)               # Current iteration number
            history['cost'].append(cost_value)              # Current cost value
            history['period'].append(current_T)             # Current period estimate
            history['residual_norm'].append(residual)       # Periodicity condition residual
            history['grad_norm'].append(grad_norm)          # Gradient magnitude
            
            # Print progress at regular intervals and on important events
            if iteration % 5 == 0 or iteration == 0 or converged or iteration == args.max_iter - 1:
                if iteration % 5 == 0 or iteration == 0:
                    print(f"\n--- Iteration {iteration} ---")
                    print(f"Cost: {cost_value:.6e}")
                    print(f"Position: x = {current_pos[0]:.8f}, y = {current_pos[1]:.8f}, z = {current_pos[2]:.8f}")
                    print(f"Period: T = {current_T:.8f}")
                    print(f"Residual norm: {residual:.6e}")
                    print(f"Gradient norm: {grad_norm:.6e}")
            
            # Check for convergence based on relative change in cost
            # Stop if the relative change in cost is smaller than the tolerance
            if iteration > 0 and abs(prev_cost - cost_value) < args.tol * (1 + abs(prev_cost)):
                if not converged:  # Only print the convergence message once
                    print(f"\nConverged after {iteration} iterations (relative cost change < {args.tol})")
                    print(f"Final cost: {cost_value:.6e}, Final period: {current_T:.8f}")
                converged = True
            
            # Early stopping if no improvement for 50 consecutive iterations
            # This prevents wasting computation when progress has stalled
            if iteration - best_iter > 50 and iteration > 100 and not converged:
                print(f"\nEarly stopping: No improvement for 50 iterations")
                converged = True
            
            # Update tracking variables for next iteration
            prev_cost = cost_value
            iteration += 1
            
            # Additional convergence check for very small cost values
            if cost_value < 1e-10:  # Hard threshold for numerical precision
                converged = True
        
        # Return the computed cost value to the optimizer
        return current_cost
    
    # ======================================================================
    # OPTIMIZATION EXECUTION AND MONITORING
    # ======================================================================
    
    # Print header with optimization parameters
    print("\n" + "="*50)
    print("STARTING OPTIMIZATION")
    print("="*50)
    print(f"Max iterations: {args.max_iter}")
    print(f"Convergence tolerance: {args.tol:.1e}")
    print("="*50 + "\n")
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # ----------------------------------------------------------------------
    # Initial evaluation before starting optimization
    # ----------------------------------------------------------------------
    with torch.no_grad():  # Disable gradient calculation for evaluation
        # Compute initial cost and period estimate
        # The cost function evaluates how well the current state satisfies the periodicity condition
        initial_cost = cost(u_vec, lambda_=args.lambda_, t_min=t_min).item()
        
        # Convert from log-space to actual period (u_vec[3] stores log(T - t_min))
        initial_T = t_min + torch.exp(u_vec[3]).item()
        
        # Display initial conditions
        print(f"Initial cost: {initial_cost:.6e}, Initial period: {initial_T:.6f}")
        
        # Initialize history tracking with starting point
        # This will be used for monitoring convergence and generating plots
        history['iter'].append(0)                       # Iteration counter
        history['cost'].append(initial_cost)            # Initial cost value
        history['period'].append(initial_T)             # Initial period estimate
        history['residual_norm'].append(float('nan'))   # Will be updated during optimization
        history['grad_norm'].append(float('nan'))       # Will be updated during optimization
    
    # ----------------------------------------------------------------------
    # Main optimization loop (handled by L-BFGS)
    # ----------------------------------------------------------------------
    try:
        # Execute the L-BFGS optimization algorithm
        # The closure() function will be called multiple times by the optimizer
        # to compute the objective function and its gradients
        lbfgs.step(closure)
        
        # If we reach here without convergence, perform final evaluation
        if not converged:
            with torch.no_grad():
                # Get current state and period
                current_pos = u_vec[:3]  # Current position in state space [x, y, z]
                current_T = t_min + torch.exp(u_vec[3]).item()  # Current period estimate
                
                # Compute current cost
                current_cost = cost(u_vec, lambda_=args.lambda_, t_min=t_min).item()
                
                # Update best solution if current is better
                # This ensures we always keep track of the best solution found
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = u_vec.detach().clone()  # Create a detached copy
                    
                print("\nOptimization finished without explicit convergence.")
    
    # Handle user interruption (Ctrl+C)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    # Handle any other exceptions during optimization
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        print("Returning best solution found so far...")
    
    # ======================================================================
    # POST-OPTIMIZATION PROCESSING
    # ======================================================================
    
    # Extract the best solution found during optimization
    final_pos = best_state[:3]  # Final position [x, y, z] in state space
    final_T = t_min + torch.exp(best_state[3]).item()  # Final period estimate
    final_cost = best_cost  # Final cost value
    
    # Compute the final residual (how well the periodicity condition is satisfied)
    with torch.no_grad():
        # Integrate the system for one period starting from final_pos
        xT = flow_map(final_pos, torch.tensor(final_T))
        # Calculate the norm of the difference between initial and final states
        # This measures how close we are to a perfect periodic orbit
        final_residual = (xT - final_pos).norm().item()
    
    # ======================================================================
    # FINAL RESULTS REPORTING
    # ======================================================================
    
    # Display comprehensive optimization results
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
    
    # ======================================================================
    # VERIFICATION OF PERIODIC SOLUTION
    # ======================================================================
    
    # Verify periodicity by checking if the solution returns to its starting point after one period
    with torch.no_grad():
        # Integrate the system for one period starting from final_pos
        xT = flow_map(final_pos, torch.tensor(final_T))
        # Calculate the residual (difference between initial and final states)
        residual = (xT - final_pos).norm().item()
        print(f"Final residual norm: {final_residual:.6e}")
    
    # ======================================================================
    # LONG-TERM INTEGRATION FOR VERIFICATION
    # ======================================================================
    
    # Print section header
    print("\n" + "="*50)
    print("VERIFYING PERIODIC SOLUTION")
    print("="*50)
    
    # Create a high-resolution time grid for 3 periods of the orbit
    # This will be used to verify the periodicity over multiple cycles
    t_verify = torch.linspace(0, 3 * final_T, 3000)  # 3000 points for smooth plotting
    
    # Integrate the system for 3 periods to verify periodicity
    with torch.no_grad():
        # Use the Dormand-Prince (dopri5) adaptive step-size integrator
        traj = odeint(lorenz_rhs, final_pos, t_verify, 
                     method='dopri5', rtol=args.rtol, atol=args.atol)
    
    # Convert results to NumPy arrays for easier manipulation
    traj_np = traj.numpy()       # Trajectory points [time, xyz]
    t_verify_np = t_verify.numpy()  # Time points
    
    # ======================================================================
    # PERIODICITY CHECK AT MULTIPLE PERIODS
    # ======================================================================
    
    # We'll check how well the solution repeats at T, 2T, and 3T
    T = final_T
    times = [T, 2*T, 3*T]  # Times to check periodicity
    states = []  # Will store the state at each multiple of T
    
    # For each time point of interest, find the corresponding state using interpolation
    for t in times:
        # Find where t would be inserted in the time array to maintain order
        idx = np.searchsorted(t_verify_np, t)
        
        # Handle edge cases (shouldn't normally happen with our time grid)
        if idx == 0 or idx == len(t_verify_np):
            states.append(traj_np[idx-1])
            continue
            
        # Perform linear interpolation between the two nearest time points
        # alpha is the fractional distance between idx-1 and idx
        alpha = (t - t_verify_np[idx-1]) / (t_verify_np[idx] - t_verify_np[idx-1])
        # Linearly interpolate the state at time t
        state = traj_np[idx-1] + alpha * (traj_np[idx] - traj_np[idx-1])
        states.append(state)
    
    # ======================================================================
    # PERIODICITY ERROR ANALYSIS
    # ======================================================================
    
    # Print header for the periodicity verification table
    print("\nVerification of periodicity:")
    print("-" * 90)
    print(f"{'Cycle':<10} | {'Time':<12} | {'Error (norm)':<15} | "
          f"{'x-Error':<15} | {'y-Error':<15} | {'z-Error':<15}")
    print("-" * 90)
    
    # Get the initial state for comparison (the optimized periodic orbit)
    x0 = final_pos.numpy()
    
    # Calculate and display errors at each period multiple
    for i, (t, state) in enumerate(zip(times, states)):
        # Calculate the difference between current state and initial state
        # For a perfect periodic orbit, this difference would be zero
        error = state - x0
        error_norm = np.linalg.norm(error)  # Overall error magnitude
        
        # Print a formatted row with error information
        print(f"{i+1}T{' '*(9-len(str(i+1)))} | {t:<12.6f} | {error_norm:<15.6e} | "
              f"{error[0]:<15.6e} | {error[1]:<15.6e} | {error[2]:<15.6e}")
    
    # ======================================================================
    # FINAL VERIFICATION SUMMARY
    # ======================================================================
    
    # Check the final state after 3 periods
    final_state = states[-1]  # State after 3 periods
    final_error = final_state - x0  # Difference from initial state
    final_error_norm = np.linalg.norm(final_error)  # Magnitude of final error
    
    # Print summary of final verification
    print("\nFinal verification after 3 periods:")
    print(f"Initial state: {final_pos.numpy()}")
    print(f"Final state:   {final_state}")
    print(f"Error norm:    {final_error_norm:.6e}")
    
    # Store the full trajectory for plotting
    traj_3T = traj_np      # Full trajectory points
    t_3T = t_verify_np     # Corresponding time points
    
    # ======================================================================
    # FINAL RESULTS SUMMARY
    # ======================================================================
    
    # Calculate total execution time
    elapsed_time = time.time() - start_time
    
    # Print comprehensive final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Final state: [{final_pos[0]:.12f}, {final_pos[1]:.12f}, {final_pos[2]:.12f}]")
    print(f"Period T: {final_T:.12f}")
    print(f"Final cost: {final_cost:.12e}")
    print(f"CPU time: {elapsed_time:.2f} seconds")
    print("="*50)
    
    # ======================================================================
    # VISUALIZATION OF RESULTS
    # ======================================================================
    # This section handles the generation of visualizations to help understand
    # the found periodic orbit and the optimization process.
    
    try:
        # Import required plotting libraries
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
        
        # ==================================================================
        # 3D PLOT: LORENZ ATTRACTOR AND PERIODIC ORBIT
        # ==================================================================
        print("\nGenerating visualization...")
        
        # Create a new figure with specified size
        fig = plt.figure(figsize=(12, 8))
        # Add a 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # ------------------------------------------------------------------
        # Plot the background Lorenz attractor for context
        # ------------------------------------------------------------------
        print("  - Plotting attractor...")
        # Generate a long trajectory to show the full attractor
        # We skip the first 1000 points to remove transient behavior
        attractor_traj = odeint(lorenz_rhs, final_pos, torch.linspace(0, 100, 10000), 
                              method='dopri5', rtol=args.rtol, atol=args.atol).detach().numpy()
        # Plot the attractor in gray with transparency
        ax.plot(attractor_traj[1000:, 0], attractor_traj[1000:, 1], attractor_traj[1000:, 2], 
               'gray', alpha=0.4, linewidth=0.8, label='Lorenz attractor')
        
        # ------------------------------------------------------------------
        # Plot the found unstable periodic orbit (UPO)
        # ------------------------------------------------------------------
        print("  - Plotting periodic orbit...")
        # Generate points for exactly one period of the orbit
        t_po = torch.linspace(0, final_T, 1000)  # High resolution for smooth curve
        po_traj = odeint(lorenz_rhs, final_pos, t_po, 
                        method='dopri5', rtol=args.rtol, atol=args.atol).detach().numpy()
        # Plot the periodic orbit in red with a thicker line
        ax.plot(po_traj[:, 0], po_traj[:, 1], po_traj[:, 2], 'red', 
               linewidth=3, label=f'Unstable Periodic Orbit (T={final_T:.6f})')
        
        # Mark the initial point of the orbit
        ax.scatter(*final_pos.detach().numpy(), color='red', s=100, 
                 marker='o', edgecolor='black', label='Initial point')
        
        # ------------------------------------------------------------------
        # Configure plot appearance
        # ------------------------------------------------------------------
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('Lorenz System: Unstable Periodic Orbit (UPO) on Strange Attractor', 
                    fontsize=14)
        ax.legend(fontsize=11)
        ax.view_init(elev=20, azim=135)  # Set 3D viewing angle
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        
        # ==================================================================
        # SAVE THE 3D PLOT
        # ==================================================================
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        # Generate a unique filename to prevent overwriting
        base_name = 'lorenz_periodic_orbit'
        file_ext = '.png'
        counter = 1
        output_file = f'images/{base_name}{file_ext}'
        
        # If file exists, append a number to the filename
        while os.path.exists(output_file):
            output_file = f'images/{base_name}_{counter}{file_ext}'
            counter += 1
            
        # Save the figure with high resolution
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as '{output_file}'")
        
        # ==================================================================
        # OPTIMIZATION HISTORY PLOT
        # ==================================================================
        print("  - Plotting optimization history...")
        
        # Create a figure with two subplots (stacked vertically)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # ------------------------------------------------------------------
        # Subplot 1: Cost and Residual Norm
        # ------------------------------------------------------------------
        # Plot cost (blue solid line) on left y-axis (log scale)
        ax1.semilogy(history['iter'], history['cost'], 'b-', label='Cost')
        # Plot residual norm (red dashed line) on same y-axis
        ax1.semilogy(history['iter'], history['residual_norm'], 'r--', label='Residual norm')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Value (log scale)')
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)  # Add grid for better readability
        
        # ------------------------------------------------------------------
        # Subplot 2: Period and Gradient Norm
        # ------------------------------------------------------------------
        # Create a second y-axis that shares the same x-axis
        ax2_twin = ax2.twinx()
        
        # Plot period (green solid line) on left y-axis (linear scale)
        ax2.plot(history['iter'], history['period'], 'g-', label='Period')
        # Plot gradient norm (magenta dashed line) on right y-axis (log scale)
        ax2_twin.semilogy(history['iter'], history['grad_norm'], 'm--', label='Gradient norm')
        
        # Configure axes labels and colors
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Period', color='g')
        ax2_twin.set_ylabel('Gradient norm', color='m')
        
        # Combine legends from both y-axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add grid to the second subplot
        ax2.grid(True, which='both', alpha=0.3)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        
        # ==================================================================
        # SAVE THE OPTIMIZATION HISTORY PLOT
        # ==================================================================        # Save the optimization history plot with a unique filename
        counter = 1
        history_file = f'images/{base_name}_history.png'
        
        # If file exists, append a number to the filename
        while os.path.exists(history_file):
            history_file = f'images/{base_name}_history_{counter}.png'
            counter += 1
            
        plt.savefig(history_file, dpi=150, bbox_inches='tight')
        print(f"Optimization history plot saved as '{history_file}'")
        
        # Display all plots
        plt.show()
        
    # Handle missing matplotlib installation
    except ImportError as e:
        print(f"\nWarning: Could not generate plots - {str(e)}")
        print("Make sure you have matplotlib installed: pip install matplotlib")
    # Handle any other plotting errors
    except Exception as e:
        print(f"\nWarning: Error generating plots - {str(e)}")
    
    # Return all relevant results
    return final_pos, final_T, final_cost, history, traj_3T, t_3T

if __name__ == '__main__':
    """
    Main entry point of the script when executed directly.
    Sets up command-line argument parsing and runs the main optimization routine.
    """
    # Initialize the argument parser with a description of the script's purpose
    parser = argparse.ArgumentParser(
        description='Find Unstable Periodic Orbits (UPOs) of the Lorenz system.\n\n'
                    'This script uses numerical optimization to find periodic solutions\n'
                    'to the Lorenz system of differential equations.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ======================================================================
    # COMMAND-LINE ARGUMENTS
    # ======================================================================
    
    # ODE Solver Tolerances
    parser.add_argument('--rtol', type=float, default=DEFAULT_RTOL,
                      help='Relative tolerance for the ODE solver. Controls the relative error '
                           'tolerance for the numerical integration. Smaller values increase '
                           'accuracy but may slow down computation. (default: %(default).1e)')
    
    parser.add_argument('--atol', type=float, default=DEFAULT_ATOL,
                      help='Absolute tolerance for the ODE solver. Controls the absolute error '
                           'tolerance. Should be set relative to the scale of the solution. '
                           '(default: %(default).1e)')
    
    # Optimization Parameters
    parser.add_argument('--lambda', dest='lambda_', type=float, default=DEFAULT_LAMBDA,
                      help='Penalty weight for the equilibrium avoidance term in the cost function. '
                           'Higher values make the optimizer more aggressive at avoiding equilibrium '
                           'points. (default: %(default).1e)')
    
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER,
                      help='Maximum number of L-BFGS optimization iterations. The optimizer will '
                           'stop after this many iterations even if convergence is not achieved. '
                           '(default: %(default)d)')
    
    parser.add_argument('--tol', type=float, default=DEFAULT_TOL,
                      help=f'Convergence tolerance for the optimization process. The optimization '
                           f'stops when the relative change in the cost function falls below this '
                           f'value. (default: {DEFAULT_TOL:.1e})')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # ======================================================================
    # UPDATE GLOBAL PARAMETERS
    # ======================================================================
    # Update global tolerance parameters based on command-line arguments
    # These will be used throughout the script for ODE integration
    DEFAULT_RTOL = args.rtol
    DEFAULT_ATOL = args.atol
    
    # ======================================================================
    # EXECUTE MAIN OPTIMIZATION ROUTINE
    # ======================================================================
    # Call the main function with the parsed arguments
    main(args)

# =============================================================================
# DOCUMENTATION
# =============================================================================
"""
Lorenz System: Unstable Periodic Orbit (UPO) Finder
==================================================

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date:   May 2024

This script implements a numerical method to find Unstable Periodic Orbits (UPOs)
in the Lorenz system using gradient-based optimization with automatic differentiation.
The code uses PyTorch for efficient computation and automatic differentiation.

## Features
- Finds periodic orbits by minimizing a differentiable cost function
- Uses L-BFGS optimization with automatic differentiation
- Implements equilibrium point avoidance
- Provides comprehensive visualization of results
- Includes verification of periodicity
- Supports customizable solver tolerances and optimization parameters

## Methodology
1. **Initial Guess Generation**: Uses k-d tree nearest neighbor search to find
   good initial guesses for periodic orbits from a chaotic trajectory.
2. **Optimization**: Minimizes a cost function that measures the distance
   between initial and final states after one period, while avoiding equilibria.
3. **Verification**: Verifies periodicity by integrating the solution for
   multiple periods and checking return accuracy.
4. **Visualization**: Generates 3D plots of the periodic orbit on the Lorenz
   attractor and optimization history plots.

## Command-Line Options
  --rtol FLOAT     Relative tolerance for ODE solver (default: 1e-9)
                   Controls relative error tolerance in numerical integration.
                   Smaller values increase accuracy but slow computation.
                   
  --atol FLOAT     Absolute tolerance for ODE solver (default: 1e-9)
                   Controls absolute error tolerance in numerical integration.
                   Should be set relative to the solution scale.
                   
  --lambda FLOAT   Penalty weight for equilibrium avoidance (default: 1e-3)
                   Higher values make the optimizer avoid equilibrium points
                   more aggressively.
                   
  --max-iter INT   Maximum number of L-BFGS iterations (default: 300)
                   The optimizer will stop after this many iterations
                   even if convergence is not achieved.
                   
  --tol FLOAT      Convergence tolerance for optimization (default: 1e-5)
                   Optimization stops when the relative change in the
                   cost function falls below this value.

## Example Usage
```bash
# Basic usage with default parameters
python lorenz_periodic_orbit.py

# Customize optimization parameters
python lorenz_periodic_orbit.py --lambda 0.001 --max-iter 500 --tol 1e-6

# Increase solver accuracy (useful for more complex orbits)
python lorenz_periodic_orbit.py --rtol 1e-10 --atol 1e-10
```

## Output
- Saves 3D visualization of the periodic orbit as 'lorenz_periodic_orbit.png'
- Saves optimization history plot as 'lorenz_periodic_orbit_history.png'
- Prints detailed convergence and verification information to console

## Dependencies
- Python 3.6+
- PyTorch
- NumPy
- SciPy
- Matplotlib (for visualization)

## References
- Lorenz, E. N. (1963). Deterministic nonperiodic flow. Journal of the atmospheric
  sciences, 20(2), 130-141.
- Viswanath, D. (2003). The fractal property of the Lorenz attractor.
  Physica D: Nonlinear Phenomena, 190(1-2), 115-128.
"""