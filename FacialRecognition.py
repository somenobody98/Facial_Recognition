'''
Hi all! The code below is just a draft while MT and YK were coming up
with ideas. There is no need to follow because we had issues in figuring
out how these workshop should be taught. Hence, feel free to just revamp
everything if necessary.
'''

# These codes are the backend code which simplifies certain long codes
# such as the plotting of the pictures

from matplotlib.pyplot import imshow, subplots, show

from math import sqrt, ceil

from random import randint

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people, clear_data_home

from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import classification_report

class WrongDataInput(Exception):
    def __init__(self, msg = "Incorrect data entered into function"):
        super().__init__(msg)

class Extract_Data():
    def __init__(self):
        d = fetch_lfw_people(min_faces_per_person = 50)
        self.raw = d
        self.X = d.data
        self.y = d.target
        self.names = d.target_names
        self.img = d.images

    def show_img_data(self):
        print("Image data is")
        print(self.X)
        print(f"No. of Image Data: {self.X.shape[0]}")
        print(f"No. of Features: {self.X.shape[1]}")
        
    def show_img_category(self):
        print("Images' categories are")
        print(self.y)
        print(f"No. of Images' Categories: {self.y.shape[0]}")

    def show_names(self):
        print("Names of people in images are")
        print(self.names)
        print(f"No. of Names: {self.names.shape[0]}")

    def show_n_images(self, n):
        if type(n) == int:
            rows, cols = round(sqrt(n)), ceil(sqrt(n))        
            f, ax = subplots(rows,cols)
            f.suptitle("Raw Images")
            f.tight_layout(h_pad = 0.5)
            lst = []
            for i in range(rows):
                for j in range(cols):
                    k = randint(0, len(self.img)-1)
                    while k in lst:
                        k = randint(0, len(self.img)-1)
                    lst.append(k)
                    if rows == 1:
                        ax[j].imshow(self.img[k])
                        ax[j].axis('off')
                        ax[j].set_title(self.names[self.y[k]], {'fontsize': 10})
                    else:
                        ax[i,j].imshow(self.img[k])
                        ax[i,j].axis('off')
                        ax[i,j].set_title(self.names[self.y[k]], {'fontsize': 10})
            show()
        else:
            raise WrongDataInput()

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = randint(1,100))
        return X_train, X_test, y_train, y_test


class Principal_Component_Analysis():
    def __init__(self, num_of_components, test = False):
        self.test = test
        self.xt = None
        self.yt = None
        if type(num_of_components) == int:
            if num_of_components <= 1036:
                self.pca = PCA(n_components = num_of_components, svd_solver = 'randomized', whiten = True)
                if not self.test:
                    print("PCA created!")
            else:
                raise WrongDataInput("Number of components must be less than number of images and number of features")
        else:
            raise WrongDataInput()  

    def train(self, x_train, y_train):
        self.xt = x_train
        self.yt = y_train
        self.pca.fit(x_train)
        if not self.test:
            print("PCA trained!")

    def show_n_new_images(self, n):
        if type(n) == int:
            new = self.pca.components_.reshape((self.pca.n_components, 62, 47))
            f, ax = subplots(n, 2)
            f.suptitle("Before PCA vs After PCA")
            f.tight_layout(h_pad = 0.5)
            lst = []
            for i in range(n):
                k = randint(0, len(new)-1)
                while k in lst:
                    k = randint(0, len(new)-1)
                lst.append(k)
                if n == 1:
                    ax[0].imshow(self.xt[k].reshape((62, 47)))
                    ax[0].set_title(names[self.yt[k]], {'fontsize': 10})
                    ax[0].axis('off')
                    ax[1].axis('off')
                    ax[1].imshow(new[k])
                    ax[1].set_title(names[self.yt[k]], {'fontsize': 10})
                else:
                    ax[i,0].set_title(names[self.yt[k]], {'fontsize': 10})
                    ax[i,0].imshow(self.xt[k].reshape((62, 47)))
                    ax[i,0].axis('off')
                    ax[i,1].set_title(names[self.yt[k]], {'fontsize': 10})
                    ax[i,1].imshow(new[k])
                    ax[i,1].axis('off')
            show()
        else:
            raise WrongDataInput()

    def transform(self, data):
        return self.pca.transform(data)
    

class Support_Vector_Machine():
    def __init__(self):
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', verbose = True), param_grid)
        self.svm = clf
        print("SVM created!")

    def train(self, X_data, y_data):
        print("SVM training...")
        self.svm.fit(X_data, y_data)
        print("\nSVM trained!")

    def predict(self, X_data):
        print("SVM predicting...")
        prediction = self.svm.predict(X_data)
        print("SVM predicted!")
        return prediction
    
# Functions

names = ['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush',
         'Gerhard Schroeder', 'Hugo Chavez', 'Jacques Chirac', 'Jean Chretien',
         'John Ashcroft', 'Junichiro Koizumi', 'Serena Williams', 'Tony Blair']

def download_data():
    print("Data downloading...")
    fetch_lfw_people(min_faces_per_person = 50)
    print("Data downloaded!")

def results(x_test, y_test, pred):
    report = classification_report(y_test, pred, target_names = names, output_dict = True)
    for name in names:
        score = round(report[name]["precision"] * 100, 1)
        print(f"{name}'s pictures was {score}% accurately predicted")
    f, ax = subplots(3, 3)
    f.suptitle("Predictions")
    f.tight_layout(h_pad = 0.9)
    lst = []
    for i in range(3):
        for j in range(3):
            k = randint(0, len(pred)-1)
            while k in lst:
                k = randint(0, len(pred)-1)
            lst.append(k)
            ax[i,j].axis('off')
            act = names[y_test[k]]
            pre = names[pred[k]]
            ax[i,j].set_title(f"Actual: {act}\nPredicted: {pre}", {'fontsize': 8})
            ax[i,j].imshow(x_test[k].reshape((62, 47)))
    show()

def delete_data():
    print("Data deleting...")
    clear_data_home()
    print("Data deleted!")
