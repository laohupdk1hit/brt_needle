from pickle import TRUE
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
import sklearn.metrics.pairwise as mp
import imageio
from time import time



def  dist_to_surf(points,r):
    '''compute distances of the pointdata to the sphere surf
    surf:x^2+y^2+z^2=r^2
    todo: 其他表面怎么定义，深度怎么定义'''
    dist_o = []
    for i in range(points.shape[0]):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        d = np.sqrt((x**2+y**2+z**2))
        dist_o.append(d-r)
    return  dist_o


def draw_lines(line3,view3d):
        fig = plt.figure()
        fig.set_size_inches(10.5, 10.5)

        ax = Axes3D(fig)
        plt.xlabel('x')
        plt.ylabel('y')
     
        ax.scatter3D(line3[:,0], line3[:,1], line3[:,2],marker='1', s=14,c='b')

        fig.savefig('test.png', dpi=50)
        if view3d:
            for angle in range(0,360,8):
                ax.view_init(elev=10., azim=angle)
                plt.savefig("movie%d.png" % angle)   
            outfilename = "my.gif"           
            frames = []
            for i in range(0,360,8):              
                im = imageio.imread('movie'+str(i)+'.png')            

                frames.append(im)                      
            imageio.mimsave(outfilename, frames, 'GIF', duration=0.1) 

        else:
            
            plt.show()

def creat_line(a,b,noise,num_points,scale):
    ''' creat line in 3d: lin3D=a+b*scalar+noise, 
    in which l is 3d coordinates of the points on the line the shape is num_points*3
    a; a point on the line with shape 1*3
    b; the direction vector with shape 1*3
    noise: num_points*3
    ''' 
    scalar=scale*np.arange(0,1,1/num_points).reshape(-1,1)

    line3D=(a+np.dot(scalar,b)+noise*0.08)

    return line3D

def create_toy_data(num_lines,num_points,draw,noise):
    a_list=[]
    b_list=[]
    
    line3=[]
    for i in range(num_lines):
        a=np.random.randn(1,3) #高斯分布，均值为0
        a=a*20
        a_list.append(a)
        b=np.random.rand(1,3) #均匀分布 均值为正
        b_list.append(b)
        l=creat_line(a,b,noise,num_points,80) #5太小了，整个针道5mm,0.5cm
        line3.append(l)
    line3=np.array(line3).reshape(num_points*num_lines,3)
    if draw:
        draw_lines(line3,0)

    return line3,a_list,b_list

num_lines = 12
num_points= 10
draw=0
noise=np.random.rand(num_points,3)*2
pointdata,a_list,b_list = create_toy_data(num_lines ,num_points,draw,noise)
np.savetxt('point4.dat',pointdata,delimiter=',',newline='\n',fmt='%.4f')

#pointdata = np.load('kmpoints_naive.npy')
#np.save('kmpoints_naive.npy',pointdata)
print("b:",b_list)
print(pointdata.shape)
# depth = dist_to_surf(pointdata,6.2)
# seed = pointdata[np.argsort(depth)[:5],:]
# '''每个seed做原点做软聚类'''

# dv = np.zeros(shape = (len(seed),40,40))
# for ind in range(len(seed)):
#     cosv = mp.cosine_distances((pointdata - seed[ind,:]),(pointdata - seed[ind,:]))
#     print(cosv.shape)
#     dv[ind,:,:] = cosv
# #print(dv.shape) = (5,40,40)
# vec_dist = -1 * np.min(dv,axis=0)
# seed_select = np.argmin(dv,axis=0)
# #print(vec_dist)
# #print(seed_select)

# kernel = -1 * mp.rbf_kernel(pointdata,pointdata)
# cos = -1* mp.cosine_distances(pointdata,pointdata)
# print(cos.shape)

# km = KMeans(n_clusters=5,init='k-means++').fit_predict(seed_select)
# # print("KM",km)

# #db = DBSCAN(min_samples=2,eps = 0.004,metric='precomputed').fit(kernel) #max=1.1
# #while using metric='cosine',very sensitive to eps
# #If metric is “precomputed”, X is assumed to be a distance matrix and must be square. 
# #precompute kdtree cosdiatance TO-DO

# ap = AffinityPropagation(preference=-0.01,affinity='precomputed',random_state=0).fit(kernel)
# af= AffinityPropagation(random_state=0).fit(cos)
# apv= AffinityPropagation(random_state=0).fit(vec_dist)
# #clustera = AgglomerativeClustering(affinity='cosine',linkage='complete', n_clusters=None,compute_full_tree=True,distance_threshold=0.055).fit(pointdata)
# #print("dist",clustera.distances_)
# print("ap lables",ap.labels_)
# print("cos lables",af.labels_)
# print("cos lables",apv.labels_)

fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, elev=30, azim=20)
plt.xlabel('x')
plt.ylabel('y')
ax.scatter3D(pointdata[:, 0], pointdata[:, 1], pointdata[:, 2],s=25,c='b')
#ax.scatter3D(pointdata[:, 0], pointdata[:, 1], pointdata[:, 2],s=25,c=apv.labels_)
plt.show()
