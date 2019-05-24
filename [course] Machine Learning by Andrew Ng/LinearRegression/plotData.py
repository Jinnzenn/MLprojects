import matplotlib.pyplot as plt

'''
散点图绘制
#matplotlib.pyplot.scatter (x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)
参数的解释：
x，y：表示的是大小为(n)的数组，也就是我们即将绘制散点图的数据点
s:是一个实数或者是一个数组大小为(n,)，这个是一个可选的参数。
c:表示的是颜色，也是一个可选项。默认是蓝色'b',表示的是标记的颜色，或者可以是一个表示颜色的字符，或者是一个长度为n的表示颜色的序列等等。但是c不可以是一个单独的RGB数字，也不可以是一个RGBA的序列。可以是他们的2维数组（只有一行）。
marker:表示的是标记的样式，默认的是'o'。
cmap:Colormap实体或者是一个colormap的名字，cmap仅仅当c是一个浮点数数组的时候才使用。如果没有申明就是image.cmap
norm:Normalize实体来将数据亮度转化到0-1之间，也是只有c是一个浮点数的数组的时候才使用。如果没有申明，就是默认为colors.Normalize。
vmin,vmax:实数，当norm存在的时候忽略。用来进行亮度数据的归一化。
alpha：实数，0-1之间。
linewidths:也就是标记点的长度。
'''
def plot_data(x, y):
   
    plt.scatter(x,y,marker='o',s=50,cmap='Blues',alpha=0.3)  #绘制散点图
    plt.xlabel('population')  #设置x轴标题
    plt.ylabel('profits')   #设置y轴标题 

    plt.show()
