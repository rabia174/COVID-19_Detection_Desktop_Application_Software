import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
import smtplib, ssl

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from random import randint
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QPixmap
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
import tensorflow.keras.backend as K
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.models import load_model

qtcreator_file  = "/Users/rabiayapicioglu/Desktop/PY2/mainwindow4.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

qtcreator_file2  = "/Users/rabiayapicioglu/Desktop/PY2/signup1.ui"
Ui_MainWindow2, QtBaseClass = uic.loadUiType(qtcreator_file2)

qtcreator_file3  = "/Users/rabiayapicioglu/Desktop/PY2/uploadimages.ui"
Ui_MainWindow3, QtBaseClass = uic.loadUiType(qtcreator_file3)
    
class Member():
    def __init__(self, name, surname, email, passcode):
        self.name = name
        self.surname = surname
        self.email = email
        self.passcode = passcode

class UploadImages(QtWidgets.QMainWindow, Ui_MainWindow3):
    def __init__(self,parent=None):
        super(UploadImages,self).__init__(parent)
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow3.__init__(self)
        self.setupUi(self)
        self.actionUpload_MRI_Images.triggered.connect(self.choosefile)
        self.pushButton_2.clicked.connect(self.model_1)
        
    def get_dataset_partitions_tf(self,ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1

        if shuffle:
            # Specify seed to always have the same split distribution between runs
            ds = ds.shuffle(shuffle_size, seed=12)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)    
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
    

        return train_ds, val_ds, test_ds

    def model_1(self):
        
        BATCH_SIZE = 32
        IMG_SIZE = (224, 224)

        dir = "/Users/rabiayapicioglu/Desktop/covid19-pneumonia-normal-chest-xraypa-dataset/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset"
        
        restored_model = load_model("/Users/rabiayapicioglu/Desktop/PY2/model.h5")
        
        print("Successfull!")
        self.single_image_test(restored_model,self.file_name)
        
        #self.img_3.setPixmap(QPixmap(fileName))
        
    def produce_heatmap(self,model,path):
 
        image_f = image.load_img(path, target_size = (224, 224))
        x = image.img_to_array(image_f)
        x = np.expand_dims(x, axis = 0)
        x = tf.keras.applications.densenet.preprocess_input(x)

        preds = model.predict(x)
        preds = tf.nn.softmax(preds)
        preds = np.argmax(preds, 1)

        with tf.GradientTape() as tape:
              last_conv_layer = model.get_layer('densenet121').get_layer('conv5_block16_concat')
              iterate = tf.keras.models.Model([model.get_layer('densenet121').inputs], [model.get_layer('densenet121').output, last_conv_layer.output])
              model_out, last_conv_layer = iterate(x)
              class_out = model_out[:,:,:,:np.argmax(model_out[0])]
              grads = tape.gradient(class_out, last_conv_layer)
              pooled_grads = K.mean(grads, axis=(0, 1, 2))
  
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((7, 7))
        plt.matshow(heatmap)
        plt.show()

        img=cv2.imread(path)

        INTENSITY = 0.5

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        img = heatmap * INTENSITY + img

        #cv2.imshow("test",cv2.imread(path))
        #cv2.imshow("test2",img)
        
        #plt.imshow(cv2.imread(path).astype('uint8'))
        cv2.imwrite('produced_image2.png',img)

        self.img_3.setPixmap(QPixmap('produced_image2.png'))
        
#Unspecified error) could not find a writer for the specified extension in function 'imwrite_'
    def single_image_test(self,model,path):
        class_indices = ['COVID-19', 'Healthy','Pneumonia']
        test_image = image.load_img(path, target_size = (224, 224))
        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        predictions_x = tf.nn.softmax(result)
        predictions = np.argmax(predictions_x, 1)
        predictions_percent=np.array(predictions_x)*100

        self.prob= np.max(predictions_percent[0])
        self.ind = np.argmax(predictions_percent[0])
        self.patient_name = self.text_name.toPlainText()
        self.patient_surname = self.text_surname.toPlainText()
        self.patient_age = self.text_age.toPlainText()
        
        self.diagnosis = class_indices[self.ind]
        
        self.covid_prob = predictions_percent[0][0] 
        self.healthy_prob = predictions_percent[0][1]
        self.pneumonia_prob = predictions_percent[0][2]
        
        self.progressBar.setValue(self.covid_prob)
        self.label_8.setText("%"+str(self.covid_prob))
        
        
        self.progressBar_2.setValue(self.healthy_prob)
        self.label_13.setText("%"+str(self.healthy_prob))
   
        self.progressBar_3.setValue(self.pneumonia_prob)
        self.label_15.setText("%"+str(self.pneumonia_prob))

        
        print(predictions_percent)
        print(self.diagnosis)
        
        if self.female.isChecked():
            self.gender = "Woman"
        else: 
            self.gender = "Man"
        
        self.result= 'Our Prediction for you is:\n\n'+'Name:  '+self.patient_name+'\nSurname:  '+self.patient_surname+'\nAge: '+self.patient_age+'\nGender:  '+self.gender+'\n\nModels Prediction for you is:  \n'+ self.diagnosis + ' with probability ' + str(self.prob) + '\n\n Wish you healthy days!'
        self.results_label.setText(self.result)
        
        #self.model_n=model
        self.produce_heatmap(model,path)
        
        return predictions,predictions_percent
    
    def choosefile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*)", options=options)
        if fileName:
           self.img.setPixmap(QPixmap(fileName))
        self.file_name=fileName
        print(self.file_name)   
            
class MyWindowSignUp(QtWidgets.QMainWindow, Ui_MainWindow2):
    def __init__(self,parent=None):
        super(MyWindowSignUp,self).__init__(parent)
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow2.__init__(self)
        self.setupUi(self)
        self.continueb.clicked.connect(self.save_data)
        self.apply_button.clicked.connect(self.apply)
        
    def save_data(self):
        self.m_name=str(self.name.text())
        self.m_surname=str(self.surname.text())
        self.m_email=str(self.email.text())
        self.m_passcode=str(self.password.text())
        
        
        
        self.results_output.setText("We saved your information,enter the passcode in your e-mail and please verify your account!")
        
        self.sent_to_pass=self.send_email(self.m_email,self.m_name,self.m_surname)
        
        
    def apply(self):
        if(str(self.sent_to_pass)==self.verification_code.text()):
           self.results_output.setText("Successfull! Close this window and Login!")
           m1=Member(self.m_name,self.m_surname,self.m_email,self.m_passcode)
           members.append(m1)
        else:
           self.results_output.setText("We couldn't match your data, please try again!")
           
        
          
    def send_email(self,receiver_email,name,surname):
        sender_email = "diagnosistmj@gmail.com"
        password = "refY.174"
        self.to_pass= str(randint(10000,99999))
        message = MIMEMultipart("alternative")
        message["Subject"] = "Your Verification Password to DiagnosisTMJ"
        message["From"] = sender_email
        message["To"] = receiver_email

        # Create the plain-text and HTML version of your message
        text = """\
        Hi,
        How are you?
        This is DiagnosisTMJ team, we hope you have a nice day!"""
        html = """\
        <html>
          <body>
            <p>DiagnosisTMJ,<br>
                Dear """+name+""" """+surname+""",<br> We are so glad that you are with us!<br>
               <a href="https://yapicioglurabia.com">Visit Our Website for  more information!</a> 
               <br><br>Your verified licence password to the application is 6 digit number below <br>"""+self.to_pass+"""
               <br>Please do not share it with anyone, this is specific to you. 
               <br>------------<br>
               With our Best Regards.<br>
               
               Diagnosis TMJ team.<br>
               F.Rabia Yapicioglu.
               
            </p>
          </body>
        </html>
        """

        # Turn these into plain/html MIMEText objects
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")

        # Add HTML/plain-text parts to MIMEMultipart message
        # The email client will try to render the last part first
        message.attach(part1)
        message.attach(part2)

        # Create secure connection with server and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(
                sender_email, receiver_email, message.as_string()
            )
   
        return self.to_pass
        
            
class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,parent=None):
        super(MyWindow,self).__init__(parent)
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.login)
        self.signup.clicked.connect(self.open_signUp_page)
        self.dialog= MyWindowSignUp()
        self.dialogUpload=UploadImages()
    def login(self):
        self.email = str(self.email.text())
        self.password = str(self.password.text())
        lem=[]    
        for i in range(0,len(members)):
            lem.append(members[i].email)
        if( self.email in lem ):
             i=lem.index(self.email)
             if( members[i].passcode== self.password ):
                self.open_uploadImg_page()
                
    def open_uploadImg_page(self):
        print("Here")
        self.dialogUpload.show()
        window.close()
      
    def open_signUp_page(self):
        self.dialog.show()
        window.close()
        
    

        
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    
    members=[]
    admin=Member("Rabia","Yapicioglu","y","r")
    members.append(admin)
    window = MyWindow()
    window.show()
 

    sys.exit(app.exec_())
    

    




