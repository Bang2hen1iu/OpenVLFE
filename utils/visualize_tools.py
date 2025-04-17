from copyreg import dispatch_table
from pyexpat.errors import XML_ERROR_SUSPEND_PE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


# 将坐标缩放到[0,1]区间  
def plot_embedding(data):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    return data


def t_sne(data, target, savename='T-SNE', n_components=2):
    if n_components == 2:
        tsne_digits = TSNE(n_components=n_components, random_state=35).fit_transform(data)
        aim_data = plot_embedding(tsne_digits)
        plt.figure()
        plt.subplot(111)
        plt.scatter(aim_data[:, 0], aim_data[:, 1], s=10, marker='.', c=target, cmap='rainbow')
        plt.title("T-SNE")
        plt.savefig(savename+'-2d.png')
    elif n_components == 3:
        tsne_digits = TSNE(n_components=n_components, random_state=35).fit_transform(data)
        aim_data = plot_embedding(tsne_digits)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(aim_data[:, 0], aim_data[:, 1], aim_data[:, 2], c=target)
        plt.title("T-SNE")
        plt.savefig(savename+'-3d.png')
    else:
        print("The value of n_components can only be 2 or 3")

def line_graph(prob, savename='default'):

    y = prob.squeeze(0)
    x = np.arange(len(y))
    # print(y.shape, x.shape)

    plt.figure('discrete line map')
    plt.grid(True)
    plt.scatter(x, y)
    plt.plot(x,y,color='r')
    plt.vlines(x, 0, y)
    plt.title('probability map')
    plt.savefig(savename + '.png')
    plt.cla()

def hist_graph(prob, savename='default'):
    
    plt.figure('hist map')
    plt.grid(True)
    plt.hist(prob, range=(0, np.log(25)))
    plt.title('entropy hist map')
    plt.savefig(savename + '.png')
    plt.cla()

def rect_graph():
    pass
# def auto_text(rects):
#     for rect in rects:
#         ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='center', va='bottom')

#TODO: fix picturing functions
class Curtain:
    def __init__(self, id, save_name):
        self.fig, self.ax = plt.subplot()
        self.palatte = []
        self.filename = save_name

    def show(self):
        pass
    
    def save(self, dpi=300):
        plt.savefig(self.filename, dpi)

    def clean(self):
        self.fig.clean()
        self.ax.clean()

    def rect_graph(self, x_list, y_list, title="rectangle graph", labels=None, width=0.2):
        assert(len(x_list) == len(y_list[0])), 'Unpair length of x and volumn'

        num_y = len(y_list)
        index = np.arange(len(x_list))
        if labels is None:
            labels = ['class {}'.format(i) for i in range(num_y)]
        rect_list = []
        max_v = 0
        min_v = 0
        for i, y in enumerate(y_list):
            rect = self.ax.bar(index - (num_y - 2*i)*width/2, y, color=self.palatte[i], width=width, label=labels[i])
            rect_list.append(rect)
            max_v = max(max_v, max(y))
            min_v = max(min_v, min(y))

        self.ax.set_title(title)
        self.ax.set_xticks(ticks=index)
        self.ax.set_xticklabels(labels)
        self.ax.set_ylabel('')

        self.ax.set_ylim(min_v, max_v)
        for rect in rect_list:
            self.ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='center', va='bottom')




        
            

