import numpy as np

class FeatureUniform:


    def libsvm_form(self,features,labels,name):
        svm_save_file = name + '.data.nonscaled'
        m,n = features.shape

        f = open(svm_save_file,'w')
        for i in range(m):
            f.write(str(labels[i])+' ')
            for j in range(n):
                if j == n-1:
                    f.write(str(j + 1) + ':' + str(features[i, j]))
                else:
                    f.write(str(j+1)+':'+str(features[i,j])+' ')
            f.write('\n')

    def libsvm_form2feas(self,filename):
        features = []
        f = open(filename,'r')
        for line in f.readlines():
            one_feature = []
            lines = line.split()
            for i in range(1, len(lines)):
                value = float(lines[i].split(":")[1])
                one_feature.append(value)
            features.append(one_feature)
        return np.array(features)



