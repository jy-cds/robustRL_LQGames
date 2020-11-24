import numpy as np
import matplotlib.pyplot as plt
SGD_Folder = 'nice_start/SGD_1e-5_2e-4/'
CMD_folder = 'nice_start/CMD_2e-3/'
NGD_folder = 'nice_start/NGD_1e-5_2e-3/'


SGD_Folder = 'SGD_1e-5_1e-3/'
CMD_folder = 'CMD_1e-3, 1e-3/'#'CMD_2e-4_2e-3/'
NGD_folder = 'NGD_1e-5_1e-3_outer/'


K_list = np.load(CMD_folder+'K_list.npy')
L_list = np.load(CMD_folder+'L_list.npy')
# constraint_list = np.load(CMD_folder+'constraint_list.npy')

K_SGD_list = np.load(SGD_Folder+'K_SGD_list.npy')
L_SGD_list = np.load(SGD_Folder+'L_SGD_list.npy')
# constraint_SGD_list = np.load(SGD_Folder+'constraint_SGD_list.npy')

K_NGD_list = np.load(NGD_folder+'K_NGD_list.npy')
L_NGD_list = np.load(NGD_folder+'L_NGD_list.npy')
# constraint_NGD_list = np.load(NGD_folder+'constraint_NGD_list.npy')


cutoff = 7000

p1 = plt.figure(1)
plt.subplot(121)
plt.plot(K_SGD_list[0,1:cutoff],label = 'PGDA: 1e-5, 1e-3')
plt.plot(K_list[0,1:cutoff],label = 'Proposed Method: 1e-3, 1e-3')
plt.plot(K_NGD_list[0,1:cutoff],alpha = 0.8,linestyle = 'dashed',label = 'PNGD: 1e-5, 1e-3')
# plt.plot(-0.4913*np.ones((20000,)),label = 'Nash Equilibrium')
plt.title('K(1)')
plt.xlabel('Iteration')
plt.legend(fontsize = 12)
p1.tight_layout()

plt.subplot(122)
plt.plot(K_SGD_list[1,1:cutoff],label = 'PGDA: 1e-5, 1e-3')
plt.plot(K_list[1,1:cutoff],label = 'Proposed Method: 1e-3, 1e-3')
plt.plot(K_NGD_list[1,1:cutoff],alpha = 0.8,linestyle = 'dashed',label = 'PNGD: 1e-5, 1e-3')
# plt.plot(-1.3599*np.ones((20000,)),label = 'Nash Equilibrium')
plt.title('K(2)')
plt.legend(fontsize = 12)
plt.xlabel('Iteration')
p1.tight_layout()

# p2 = plt.figure(2)
# plt.subplot(121)
# plt.plot(L_list[0,:], label = 'Proposed Method')
# plt.plot(L_SGD_list[0,:],label = 'SGD')
# plt.plot(L_NGD_list[0,:],alpha = 0.6,linestyle = 'dashed',label = 'NGD')
# # plt.plot(0.0757*np.ones((20000,)),label = 'Nash Equilibrium')
# plt.title('L(1)')
# plt.legend()

plt.show()
#
plt.subplot(122)
# plt.plot(L_list[1,:],label = 'Proposed Method')
# plt.plot(L_NGD_list[1,:],label = 'NGD')
# plt.plot(L_SGD_list[1,:],alpha = 0.6,linestyle = 'dashed', label = 'SGD')
# # plt.plot(0.1314*np.ones((20000,)),label = 'Nash Equilibrium')
# plt.title('L(2)')
# plt.legend()
# plt.suptitle('Adversary Policy: L')
#
#
# # p5 = plt.figure(5)
# # plt.plot(constraint_list,label = 'Proposed Method')
# # plt.plot(constraint_SGD_list[:,1],label = 'SGD')
# # plt.plot(constraint_NGD_list[:,1],alpha = 0.6,linestyle = 'dashed',label = 'NGD')
# # plt.plot(0.5401*np.ones((20000,)),label = 'Nash Equilibrium')
# # plt.legend()
#
# plt.show()
#
#
#
# l = 29000
# line_labels = ["Proposed Method", "True Nash Equilibrium"]
# fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
# # fig1.suptitle('Protagonist Policy: K')
#
# ax1.title.set_text('K(1)')
# ax1.set_xlabel('Iterations')
# l1 = ax1.plot(K_list[0,10:], color="green",alpha = 0.6)[0]
# l2 = ax1.plot(-0.4913*np.ones((l,)), color="blue",alpha = 1)[0]
#
# ax2.title.set_text('K(2)')
# ax2.set_xlabel('Iterations')
# l1 = ax2.plot(K_list[0,10:], color="green",alpha = 0.6)[0]
# l2 = ax2.plot(-1.3599*np.ones((l,)), color="blue",alpha = 1)[0]
#
# fig1.legend([l1, l2],     # The line objects
#            labels=line_labels,   # The labels for each line
#            loc="center right",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            # title="Legend Title"  # Title for the legend
#            )
# plt.subplots_adjust(right=0.8)
#
# plt.show()
#
#
# l = 29000
# line_labels = ["Proposed Method", "True Nash Equilibrium"]
# fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
# # fig1.suptitle('Adversary Policy: L')
#
# ax1.title.set_text('L(1)')
# ax1.set_xlabel('Iterations')
# l1 = ax1.plot(L_list[0,10:], color="green",alpha = 0.6)[0]
# l2 = ax1.plot(0.0757*np.ones((l,)), color="blue",alpha = 1)[0]
#
# ax2.title.set_text('L(2)')
# ax2.set_xlabel('Iterations')
# l1 = ax2.plot(L_list[1,10:], color="green",alpha = 0.6)[0]
# l2 = ax2.plot(0.1314*np.ones((l,)), color="blue",alpha = 1)[0]
#
# fig1.legend([l1, l2],     # The line objects
#            labels=line_labels,   # The labels for each line
#            loc="center right",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            # title="Legend Title"  # Title for the legend
#            )
# plt.subplots_adjust(right=0.8)
#
# plt.show()
#
#
# p1 = plt.figure(1)
# plt.subplot(121)
# plt.plot(K_list[0,10:],label = 'Proposed Method')
# plt.plot(-0.4913*np.ones((l,)),label = 'Nash Equilibrium')
# plt.title('K(1)')
# plt.legend()
#
# plt.subplot(122)
# plt.plot(K_list[1,:],label = 'Proposed Method')
# plt.plot(-1.3599*np.ones((l,)),label = 'Nash Equilibrium')
# plt.title('K(2)')
# plt.legend()
# plt.suptitle('Protagonist Policy: K')
#
# p2 = plt.figure(2)
# plt.subplot(121)
# plt.plot(L_list[0,:], label = 'Proposed Method')
# plt.plot(0.0757*np.ones((l,)),label = 'Nash Equilibrium')
# plt.title('L(1)')
# plt.legend()
#
# plt.subplot(122)
# plt.plot(L_list[1,:],label = 'Proposed Method')
# plt.plot(0.1314*np.ones((l,)),label = 'Nash Equilibrium')
# plt.title('L(2)')
# plt.legend()
# plt.suptitle('Adversary Policy: L')
#
#
# p5 = plt.figure(5)
# # plt.plot(constraint_list,label = 'Proposed Method')
# plt.plot(0.5401*np.ones((l,)),label = 'Nash Equilibrium')
# plt.legend()
#
# plt.show()
#
#
#
#
