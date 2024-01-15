######################################################################
############################# Plotting ###############################
######################################################################

# prep data for plotting
x_star = tf.cast(x_star, dtype=_data_type)
uhb_pred = model(x_star)
f_pred = equations(x_star, model, drop_mass_balance=False)
x_star = x_star.numpy().flatten().flatten()
u_star = u_star.flatten()
h_star = h_star.flatten()
u_p = uhb_pred[:, 0:1].numpy().flatten()
h_p = uhb_pred[:, 1:2].numpy().flatten()
B_p = uhb_pred[:, 2:3].numpy().flatten()
mom_residue = f_pred[0].numpy().flatten()
mass_residue = f_pred[1].numpy().flatten()

fig = plt.figure(figsize=[15, 16])


# PREDICTION AND RESIDUE PLOTTING
ax = plt.subplot(331)
ax.plot(x_star, u_p, 'b-', linewidth=2, label='predict')
ax.plot(x_star, u_star, 'r--', linewidth=2, label='data')
ax.set_xlabel('x', fontsize=15)
ax.set_title('Velocity', fontsize=15, rotation=0)
ax.legend()

ax = plt.subplot(332)
ax.plot(x_star, h_p, 'b-', linewidth=2, label='predict')
ax.plot(x_star, h_star, 'r--', linewidth=2, label='data')
ax.set_xlabel('x', fontsize=15)
ax.set_title('Thickness', fontsize=15, rotation=0)
ax.legend()

ax = plt.subplot(334)
ax.plot(x_star, mom_residue, 'b-', linewidth=2)
ax.set_xlabel('x', fontsize=15)
ax.set_title('Momentum residue', fontsize=15, rotation=0)

ax = plt.subplot(335)
ax.plot(x_star, mass_residue, 'b-', linewidth=2)
ax.set_xlabel('x', fontsize=15)
ax.set_title('Mass residue', fontsize=15, rotation=0)


# MASS BALANCE PLOT
ax = plt.subplot(336)
ax.plot(x_star, u_p*h_p, 'b-', label='predicted')
ax.plot(x_star, u_star*h_star, 'g*', label='data')
ax.plot(x_star, x_star + q0 * np.ones_like(x_star), 'r--', label='analytic')
ax.set_xlabel('x', fontsize=15)
ax.legend()
plt.title('Mass Balance', fontsize=15)

# HARDNESS PLOT
ax = plt.subplot(333)
ax.plot(x_star, B_p, 'b-', linewidth=2, label='predict')
ax.plot(x_star, np.ones_like(x_star), 'r--', linewidth=2, label='actual')
ax.set_xlabel('x', fontsize=15)
ax.set_title('Hardness Profile', fontsize=15)

# LOSS OVER ITERATION PLOTS
data_losses = np.trim_zeros(data_losses, 'b')
equation_losses = np.trim_zeros(equation_losses, 'b')
total_losses = np.trim_zeros(total_losses, 'b')
iteration_nums = np.arange(0, num_iterations, 10)

ax= plt.subplot(337)
ax.plot(iteration_nums[20:len(data_losses)], data_losses[20:])
ax.set_title('Data Loss (Boundary conditions)', fontsize=15)
ax.set_yscale('log')

ax = plt.subplot(338)
ax.plot(iteration_nums[20:len(equation_losses)], equation_losses[20:])
ax.set_title('Equation Loss', fontsize=15)
ax.set_yscale('log')

ax = plt.subplot(339)
ax.plot(iteration_nums[20:len(total_losses)], total_losses[20:])
ax.set_title('Total Loss', fontsize=15)
ax.set_yscale('log')

title_str = 'Adam for ' + str(0) + ', LBFGS for ' + str(num_iterations) \
    + ', Gamma = ' + str(gamma) + ', Order = 1, Fractional = ' + str(fractional) \
        + ', Training Time = ' + str(round(elapsed, 3)) + ' s'
plt.suptitle(title_str, fontsize=18, fontweight='bold')
plt.savefig(output_figure)

# # MOMENTUM BALANCE PLOT
# fig_mom = plt.figure()
# u_p_x = np.gradient(u_p, x_star)
# h_p_x = np.gradient(h_p, x_star)
# momlhs = nu_star * np.gradient(B_func(x_star) * h_p * np.abs(u_p_x) ** (1/n - 1) * u_p_x, x_star)
# momrhs = h_p * h_p_x
# ax_mom = plt.plot(x_star, momlhs - momrhs, 'b-')
# plt.title('Momentum Balance')