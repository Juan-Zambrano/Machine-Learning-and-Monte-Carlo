import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import seaborn as sns 


def main():
    with open('HatsopoulosReachTask.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        reg_tim_tri_ls = [line for line in csv_reader]

        region_ls = []
        for i in range(143):
            for j in reg_tim_tri_ls[i]:
                region_ls.append(j)
                break

        for i in range(143):
            if (reg_tim_tri_ls[i][0] in region_ls):
                del reg_tim_tri_ls[i][0]

        # print(len(reg_tim_tri_ls))  # 143
        # print(len(reg_tim_tri_ls[0]))  # 5530
        # reading and parsing direction file   direction lenght = 158
        with open('direction.csv', 'r') as csv_filee:
            csv_reader = csv.reader(csv_filee)
            direction_ls = [line for line in csv_reader]
        direct_ls = []
        for i in range(158):
            for j in direction_ls[i]:
                direct_ls.append(int(j))
        matrix_direction = np.array(direct_ls)

        #  eliminate time values and work on neuron columns
        back_to_3d = [[neuron[35*i:35*(i+1)]for i in range(158)] for neuron in reg_tim_tri_ls]
        sum_spike = [[sum([int(time) for time in direction]) for direction in neuron] for neuron in back_to_3d]
        #print(sum_spike)
        matrix_neuron = np.array(sum_spike)
        #count number of directions in direction list: ugly ass code
        count_0 = 0
        count_45 = 0
        count_90 = 0
        count_135 = 0
        count_180 = 0
        count_225 = 0
        count_270 = 0
        count_315 = 0
        i = 0
        for i in matrix_direction:
            if(i == 1):
                count_0 +=1
            if(i == 2):
                count_45 +=1
            if(i == 3):
                count_90 +=1
            if(i == 4):
                count_135 +=1
            if(i == 5):
                count_180 +=1
            if(i == 6):
                count_225 +=1
            if(i == 7):
                count_270 +=1
            if(i == 8):
                count_315 +=1
        direct_key = ['0','45','90','135','180','225','270','315']
        direct_value = [count_0,count_90,count_135,count_180,count_225,count_270,count_315]
        Sum_of_direct_dict = dict(zip(direct_key,direct_value))


        #print('direct_dict', Sum_of_direct_dict)

        #Create neu_x label list:
        key4neuron_dict = [ ('neu_' + str(i)) for i in range(1,len(matrix_neuron) + 1)]
        #create dictionary with neu1 label with list of spike frequency list
        neu_spike = dict(zip(key4neuron_dict,(i for i in matrix_neuron)))


        #print('neu_1', neu_spike['neu_2'])
        #neu1 = [7,18,21 ....,11] -> lenght = 158
        #new_spike = {key = neu_x:value = [7,18,21 ...,11]}

        #swaping integer values in direction matrix with coordinate values
        direction_label = []
        for i in matrix_direction:
            if(i == 1):
                direction_label.append('0')
            if(i == 2):
                direction_label.append('45')
            if(i == 3):
                direction_label.append('90')
            if(i == 4):
                direction_label.append('135')
            if(i == 5):
                direction_label.append('180')
            if(i == 6):
                direction_label.append('225')
            if(i == 7):
                direction_label.append('270')
            if(i == 8):
                direction_label.append('315')
        direction_label_matrix = np.array(direction_label)
     

        #PCA CODE!!!!!

        PCA_COL = [str(i) for i in matrix_direction]
        df = pd.DataFrame(columns = [str(i) for i in matrix_direction]) # add directions to column section

        
        for i in range(143):
            df.loc[i] = matrix_neuron[i]
        df.head(143)

        # renaming matrix of 143 X 158: neu X num_trials
        X = df 
        # normalizing
        x_std = StandardScaler().fit_transform(X)

        # Constructing Covariance Matrix
        # Taking Columns from x_std
        covariance_matrix = np.cov(x_std.T)
        #Finding the Eigen Vectors and Eigen Values from Cov_matrix (Eigendecomposition)
        # Eigendecomposition of the covariance matrix was done after data was standardized
        eigen_val,eigen_vec = np.linalg.eig(covariance_matrix)
        #deciding which eigen vectors to use by picking the eigen vectors with the higest eigen-values
        
        eig_pairs = [(np.abs(eigen_val[i]), eigen_vec[:,i]) for i in range(len(eigen_vec))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        #print('Eigenvalues in descending order:')
        #use the top 3 eigen vectors from eig_pairs of the 3 PC
        #for i in eig_pairs:
            #print(i[0])
        

        #FINDING OUT HOW MANY COMPONENTS TO USE (Explained Variance)
        tot = sum(eigen_val)
        var_exp = [(i / tot)*100 for i in sorted(eigen_val, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        print("Explained Variance")
        #print(var_exp)
        

        #print('PC1_VAR = ',eigen_val[0]/sum(eigen_val))
        PC1 = x_std.dot(eig_pairs[0][1].T)
        PC2 = x_std.dot(eig_pairs[1][1].T)
        PC3 = x_std.dot(eig_pairs[2][1].T)
        #creating component data frame
        PCA_R = pd.DataFrame(PC1,columns = ['PC1'])
        PCA_R['PC2'] = PC2
        PCA_R['PC3'] = PC3
        PCA_R.head(143)
        PCA_R_Matrix_1 = PCA_R.as_matrix()


        
        #Normalize each component and remove imaginary parts, this code is super janky and not efficent 
        PCA_R_Matrix_1 =[[str(c).replace('+0j','').replace('(','').replace(')','') for c in r] for r in PCA_R_Matrix_1]
        PCA_data = np.array(PCA_R_Matrix_1)
        df_PCA = pd.DataFrame(columns = ['PC1','PC2','PC3']) # add directions to column section
        for i in range(143):
            df_PCA.loc[i] = PCA_data[i]
        df_PCA.head(143)
        #Normalize direction dataframe
        PCA_no_j = df_PCA
        PCA_std = StandardScaler().fit_transform(PCA_no_j)
        #new dataframe of no j values and normalized after covariance matrix was made
        PCA_final = pd.DataFrame(columns = ['PC1','PC2','PC3']) # add directions to column section
        
        for j in range(143):
            PCA_final.loc[j] = PCA_std[j]
        PCA_final.head(143)

        #Displaying 2 principal components and the neurons projected
        fig1 = plt.figure(figsize = (10,10))
        plt.scatter(PCA_final['PC1'] ,PCA_final['PC2'] , color = ('r'), alpha = 0.5, edgecolor = 'k')
        plt.title('Principal Component Analysis of 143 neurons on 2 components')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

        #Displaying 3 principal components and the neurons projected
        fig2 = plt.figure()
        ax = fig2.add_subplot(111,projection = '3d')
        ax.scatter(PCA_final['PC1'] ,PCA_final['PC2'],PCA_final['PC3'], color = ('b'), alpha = 0.8, edgecolor = 'k')
        ax.set_title('Principal Component Analysis of 143 neurons on 3 components')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()


        #Factor analysis
        PCA_MATRIX = PCA_final.as_matrix()
        #solve for Y = XA + b , eliminate b cause its error and not doing error analysis
        # Y =  matrix_neuron (158 X 143)
        # X =  PCA_MATRIX
        # A = Direction Matrix(regressed)

        #Fig3 projecting onto 3 principal components
        A = (PCA_MATRIX.T.dot(PCA_MATRIX)).dot(PCA_MATRIX.T.dot(matrix_neuron))
        fig3 = plt.figure()
        ax = fig3.add_subplot(1,1,1, projection='3d')
        colors =['#630C3A', '#6a38ff','#1bc42f']
        for i in range(len(A)):
            ax.scatter(np.real(A[0][:]), np.real(A[1][:]), np.real(A[2][:]), color = colors[i])
        ax.set_title('PCA of Direction matrix on 3 components')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show() 

        #Fig4 projecting the first 2 principal components
        colors2 = ['#630C3A', '#6a38ff']
        for i in range(2):
            plt.scatter(np.real(A[0][:]), np.real(A[1][:]), color = colors2[i])
        plt.title('PCA of Direction matrix on 2 components')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

        # Writing a file to input into the clustering script
        A2 = A.T
        A2_right_form = A2
        
        #K-MEANS CLUSTERING CODE
        cluster_data = np.array(A2_right_form)
        
        df1 = pd.DataFrame(columns = ['x','y','z']) # add directions to column section
        
        for i in range(158):
            df1.loc[i] = cluster_data[i]
        df1.head(158)
    
        #Normalize direction dataframe
        Y = df1
        direct_std = StandardScaler().fit_transform(Y)
        #normalized dataframe to implementent into clustering algorithm
        df2 = pd.DataFrame(columns = ['x','y','z']) # add directions to column section
        
        for j in range(158):
            df2.loc[j] = direct_std[j]
        df2.head(158)

      
        kmeans = KMeans(n_clusters = 8)
        kmeans.fit(direct_std)

        labels = kmeans.predict(direct_std)
        centroids = kmeans.cluster_centers_

        colmap = {1: 'r', 2: 'g', 3: 'b',4:'c',5:'m',6:'y',7:'k',8:'teal'}

        #2D Cluster plot
        fig = plt.figure(figsize = (10,10))
        colors = map(lambda x: colmap[x+1],labels)

        plt.scatter(df2['x'] ,df2['y'] , color = colors , alpha = 0.5, edgecolor = 'k')
        for idk,centroid in enumerate(centroids):
            plt.scatter(centroid[0],centroid[1],color = colmap[idk + 1])
        plt.title('2D-Cluster Plot (Euclidean)')
        plt.xlabel('X-Direction')
        plt.ylabel('Y-Direction')
        plt.show()


        #3D Cluster plot
        fig_3D = plt.figure(figsize = (10,10))
        colors = map(lambda x: colmap[x+1],labels)

        ax = fig_3D.add_subplot(111,projection = '3d')

        ax.scatter(df2['x'] ,df2['y'] , df2['z'] , color = colors , alpha = 0.5, edgecolor = 'k')
        for idk,centroid in enumerate(centroids):
            plt.scatter(centroid[0],centroid[1],centroid[2],color = colmap[idk + 1])
        ax.set_title('3D-Cluster Plot (Euclidean)')
        ax.set_xlabel('X-Direction')
        ax.set_ylabel('Y-Direction')
        ax.set_zlabel('Z-Direction')
        plt.show()
       


main()












