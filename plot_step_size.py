import numpy as np
import matplotlib.pyplot as plt


# method = 'SGD'
# step_size = { '1e-5_1e-3', '2e-5'}

# method = 'NGD'
# step_size = { '1e-5_1e-3_outer','1e-5_5e-3_outer', '1e-5_outer'}

method = 'CMD'
step_size = {'1e-5, 1e-5','2e-4, 2e-4','1e-4, 1e-4','2e-4, 2e-3','1e-4, 1e-3', '1e-3, 1e-3'}

# name_list = ['1e-5, 1e-5','2e-4, 2e-4','1e-4, 1e-4','2e-4, 2e-3','1e-4, 1e-3', '1e-3, 1e-3']
# name_list.reverse()

var_holder = {}

for step in step_size:
    if method != 'CMD':
        K_list = np.load(method+'_'+step+'/K_'+ method +'_list.npy')
    else:
        K_list = np.load(method + '_' + step + '/K_list.npy')
    var_holder[step] = K_list



p1 = plt.figure(1)
ax1=p1.add_subplot(1,2,1)
ax1.set_xlabel('Iteration')
i = 0
for name, array in var_holder.items():
    if i %2 ==1:
        plt.plot(array[0, 0:], linewidth=2, alpha=0.7,  label=name)
    else:
        plt.plot(array[0, 0:], linewidth=1.5, alpha=1,linestyle='dashed', label=name)
    i += 1
# plt.plot(-0.4913*np.ones((3000,)),label = 'Nash Equilibrium')
# ax1.scatter(0, -0.4913, s=80, marker=(5, 1))

handles,labels=ax1.get_legend_handles_labels()
# sorted_legends= [x for x in zip(labels)] #sort the labels based on the average which is on a list
# sorted_handles=[x for x in zip(handles)] #sort the handles based on the average which is on a list
ax1.legend(handles,labels)
plt.title('K(1)')


# plt.subplot(122)
ax2=p1.add_subplot(1,2,2)

ax2.set_xlabel('Iteration')
i=0
for name, array in var_holder.items():
    if i %2 ==1:
        plt.plot(array[1, 0:], linewidth=2, alpha=0.7,  label=name)
    else:
        plt.plot(array[1, 0:], linewidth=1.5, alpha=1,linestyle='dashed', label=name)
    i += 1
# plt.plot(-1.3599*np.ones((3000,)),label = 'Nash Equilibrium')
plt.title('K(2)')
p1.tight_layout()


# line_labels = ['1e-5','2e-4_2e-3','1e-4_1e-3','2e-4', '1e-4','1e-3']

# plt.legend( labels=line_labels)
# plt.legend()
ax1.legend(fontsize=12)

ax2.legend(fontsize=12)


# plt.suptitle('Various Step Sizes for CMD')

plt.show()


