import numpy as np
import matplotlib.pyplot as plt
import csv

plotname1 = 'fc100-accuracy'
plotname2 = 'fc100-loss'

# plotname1 = 'fc10-100_comparison-val_acc'
# plotname2 = 'fc10-100_comparison-loss'

the_date = '20171027'
experiment = 'fc100_fc3_relu'
logfile = 'experiment/' + experiment + '.csv'

experiment2 = 'fc10_fc3_relu'
logfile2 = 'experiment/' + experiment2 + '.csv'

# load data
def load_logs(logfile, discard_header=False):
    iteration, train_accuracy, valid_accuracy, loss = [], [], [], []
    with open(logfile) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if discard_header:
                discard_header = False
                continue
            r_it, r_train_acc, r_valid_acc, r_loss = row
            iteration.append(int(r_it))
            train_accuracy.append(float(r_train_acc))
            valid_accuracy.append(float(r_valid_acc))
            loss.append(float(r_loss))
    return iteration, train_accuracy, valid_accuracy, loss

it_fc100, tr_acc_fc100, va_acc_fc100, loss_fc100 = load_logs(logfile, discard_header=True)
it_fc10, tr_acc_fc10, va_acc_fc10, loss_fc10 = load_logs(logfile2, discard_header=True)


# Colours
col_pink   = '#E13375'
col_blue   = '#095998'
col_orange = '#F17B18'
col_green  = '#29C78D'

col_red    = '#d91c22'
col_cyan   = '#1cd9cc'
col_green  = '#81d91c'

#--------------------------------------------------------------------------------
# Accuracy
#--------------------------------------------------------------------------------

# Set figure size
fig, ax = plt.subplots(figsize=(12,6.75))

# Plotting multiple lines
ax.plot(it_fc100, va_acc_fc100, linewidth=2, color=col_red, label='Validation Accuracy')
ax.plot(it_fc100, tr_acc_fc100, '--', linewidth=2, color=col_cyan, label='Training Accuracy')

# Configuration comparisons
# ax.plot(it_fc100, va_acc_fc100, linewidth=2, color=col_red, label='FC100 Validation')
# ax.plot(it_fc10, va_acc_fc10, '--', linewidth=2, color=col_cyan, label='FC10 Validation')

# Plot title
ax.set_title('Training Accuracy', y=0.93, x=0.14, fontsize=16)

# Axes labels, fontsize (offset by adding y=0.0 etc to arguments)
ax.set_ylabel('Accuracy', fontsize=16)
ax.set_xlabel('Iteration', fontsize=16)

# Ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', right='on', top='on', direction='in', labelsize=12, length=6)
ax.tick_params(axis='both', which='minor', right='on', top='on', direction='in', labelsize=12, length=4)

# Set range limit on axes
ax.set_ylim([0,1.2])
ax.set_xlim([0,2500])

# To set a grid
ax.grid(True)

# Legend, generated from plot details and labels
ax.legend(loc=1, prop={'size': 12})

plt.savefig('./experiment/' + the_date + '-' + plotname1 + '.png', bbox_inches='tight')


#--------------------------------------------------------------------------------
# Loss
#--------------------------------------------------------------------------------

# Set figure size
fig, ax = plt.subplots(figsize=(12,6.75))

ax.plot(it_fc100, loss_fc100, linewidth=2, color=col_red, label='Loss')

# Configuration comparisons
# ax.plot(it_fc100, loss_fc100, linewidth=2, color=col_red, label='FC100 Loss')
# ax.plot(it_fc10, loss_fc10, '--', linewidth=2, color=col_cyan, label='FC10 Loss')

# Plot title
ax.set_title('Training Loss', y=0.93, x=0.12, fontsize=16)

# Axes labels, fontsize (offset by adding y=0.0 etc to arguments)
ax.set_ylabel('Loss', fontsize=16)
ax.set_xlabel('Iteration', fontsize=16)

# Ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', right='on', top='on', direction='in', labelsize=12, length=6)
ax.tick_params(axis='both', which='minor', right='on', top='on', direction='in', labelsize=12, length=4)

# Set range limit on axes
# ax.set_ylim([0,1.2])
ax.set_xlim([0,2500])

# To set a grid
ax.grid(True)

# Legend, generated from plot details and labels
ax.legend(loc=1, prop={'size': 12})

plt.savefig('./experiment/' + the_date + '-' + plotname2 + '.png', bbox_inches='tight')
