import cleaningtest1 as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#染色体长度
DNA_SIZE = 26
#种群规模
POP_SIZE = 100
#交叉率
CROSSOVER_RATE = 0.8
#变异率
MUTATION_RATE = 0.1
#种群迭代次数
N_GENERATIONS = 500
#自变量x和y的范围
X_BOUND = [-3, 3]
Y_BOUND = [-3, 3]

GET_MAX = 0

#定义函数
def F(x, y):
    return x ** 2 + y ** 2

#获取最大适应度
def get_Maxfitness(pop):
    x, y = translateDNA(pop)
    pred = F(x, y)
    # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度
    return (pred - np.min(pred)) + 1e-3    #np.min(pred) 获得pred中的最小值 相减是为了获得非负数 加上一个很小的数防止出现为0的适应度

#获取最小适应度
def get_Minfitness(pop):
    x, y = translateDNA(pop)
    pred = F(x, y)
    #减去最大值后fitness范围:[np.min(pred) - np.max(pred),0] 加上-号后范围为[0, np.max(pred) - np.min(pred)]   加上一个很小的数防止出现为0的适应度
    return -(pred - np.max(pred)) + 1e-3

def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    #友情提示：pop是二维的哦~
    x_pop = pop[:, 1::2]  # 奇数列表示X
    y_pop = pop[:, ::2]  # 偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)     dot() 矩阵乘法计算
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    #返回的是x与y染色体的实数值(解码值)
    return x, y


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因
        if np.random.rand() < CROSSOVER_RATE:  # 产生一个0~1随机值，如果小于0.8则交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点 [low, high)
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 后代变异
        new_pop.append(child) #加入下一代种群

    return new_pop #返回新一代种群

#变异
def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=(fitness) / (fitness.sum()))
    '''
    介绍以下choice方法的参数：
    numpy.random.choice(a, size=None, replace=True, p=None)
    #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
    #replace:True表示可以取相同数字，False表示不可以取相同数字
    #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
    也就是说，从种群中根据适应度函数的大小为挑选概率，挑选POP_SIZE个元素作为下一代
    '''
    return pop[idx]


def print_info(pop):
    if GET_MAX == 1:
        fitness = get_Maxfitness(pop)
    else:
        fitness = get_Minfitness(pop)
    #获取适应度最大的下标索引
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))

def plot_3d(ax):
    X = np.linspace(*X_BOUND, 100) #等价于 np.linspace(-3,3,100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y) #扩展为二维矩阵，方便建图
    Z = F(X, Y) #计算函数值
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)#设置绘图参数
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()

if __name__ == "__main__":
    #创建画布
    fig = plt.figure()
    ax = Axes3D(fig)
    #更改为交互模式
    plt.ion()  #交互模式，程序遇到plt.show不会暂停，而是继续执行
    #绘制3D图
    plot_3d(ax)

    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))# 生成随机数的范围是[0，2) 矩阵大小为 rows: POP_SIZE cols:DNA_SIZE*2,因为两个自变量

    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        if 'sca' in locals():#'sca'判断是不是一个全局变量，如果是，则将其移除
            sca.remove()
        #定义sca为scatter返回对象，scatter就是在函数图像上画出当前种群每个个体的位置
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o');
        plt.show();
        plt.pause(0.1)
        #交叉变异，产生新一代种群
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        #计算适应度
        if GET_MAX == 1:
            fitness = get_Maxfitness(pop)
        else:
            fitness = get_Minfitness(pop)
        #选择
        pop = select(pop, fitness)  # 选择生成新的种群
    #打印最后的全局最优信息
    print_info(pop)
    plt.ioff() #关闭交互模式
    plot_3d(ax)
